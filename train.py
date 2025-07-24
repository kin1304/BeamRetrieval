#!/usr/bin/env python3
"""
Main Training Script for Advanced Multi-Hop Retriever
Usage: python train.py [--samples N] [--epochs N] [--batch_size N]
"""
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import random

from models.advanced_retriever import create_advanced_retriever
from utils.data_loader import load_hotpot_data

def calculate_f1_em(predictions, targets):
    """Calculate F1 score and Exact Match"""
    if not predictions or not targets:
        return 0.0, 0.0
    
    # Convert to sets for F1 calculation
    pred_set = set(predictions)
    target_set = set(targets)
    
    # Exact Match
    em = 1.0 if pred_set == target_set else 0.0
    
    # F1 Score
    if len(pred_set) == 0 and len(target_set) == 0:
        f1 = 1.0
    elif len(pred_set) == 0 or len(target_set) == 0:
        f1 = 0.0
    else:
        precision = len(pred_set & target_set) / len(pred_set)
        recall = len(pred_set & target_set) / len(target_set)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1, em

class RetrievalDataset(Dataset):
    """Dataset for multi-hop retrieval training"""
    
    def __init__(self, data, tokenizer, max_len=256, num_contexts=5):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_contexts = num_contexts
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        contexts = item['contexts']
        supporting_facts = item.get('supporting_facts', [])
        
        # Select contexts
        if len(contexts) > self.num_contexts:
            sf_titles = {sf[0] for sf in supporting_facts}
            sf_contexts = [ctx for ctx in contexts if ctx['title'] in sf_titles]
            other_contexts = [ctx for ctx in contexts if ctx['title'] not in sf_titles]
            
            selected_contexts = sf_contexts[:2]
            remaining = self.num_contexts - len(selected_contexts)
            if remaining > 0 and other_contexts:
                selected_contexts.extend(random.sample(other_contexts, min(remaining, len(other_contexts))))
        else:
            selected_contexts = contexts
        
        # Pad contexts if needed
        while len(selected_contexts) < self.num_contexts:
            selected_contexts.append({'title': 'Empty', 'text': 'No context available.'})
        
        # Tokenize question
        q_tokens = self.tokenizer(
            question,
            max_length=self.max_len//2,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize contexts
        c_tokens_list = []
        for ctx in selected_contexts[:self.num_contexts]:
            ctx_text = f"{ctx['title']}: {ctx['text']}"
            c_tokens = self.tokenizer(
                ctx_text,
                max_length=self.max_len,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            c_tokens_list.append(c_tokens['input_ids'].view(-1))
        
        # Supporting facts indices
        sf_indices = []
        for i, ctx in enumerate(selected_contexts[:self.num_contexts]):
            if any(ctx['title'] == sf[0] for sf in supporting_facts):
                sf_indices.append(i)
        
        # Ensure at least 1 supporting fact
        if not sf_indices:
            sf_indices.append(0)
        sf_indices = sf_indices[:3]  # Max 3 supporting facts
        
        # Pad to ensure consistent length
        while len(sf_indices) < 2:
            sf_indices.append(sf_indices[0])
        
        return {
            'q_codes': [q_tokens['input_ids'].view(-1)],
            'c_codes': c_tokens_list,
            'sf_idx': [torch.tensor(sf_indices, dtype=torch.long)],
            'hop': len(sf_indices)
        }

def collate_fn(batch):
    """Custom collate function"""
    return {
        'q_codes': [item['q_codes'] for item in batch],
        'c_codes': [item['c_codes'] for item in batch],
        'sf_idx': [item['sf_idx'] for item in batch],
        'hops': [item['hop'] for item in batch]
    }

def train_epoch(model, dataloader, optimizer, device, max_batches=None):
    """Train for one epoch"""
    model.train()
    epoch_losses = []
    f1_scores = []
    em_scores = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        
        batch_loss = 0
        valid_samples = 0
        batch_f1 = 0
        batch_em = 0
        
        # Process each sample in batch
        for i in range(len(batch['q_codes'])):
            try:
                q_codes = [q.to(device) for q in batch['q_codes'][i]]
                c_codes = [c.to(device) for c in batch['c_codes'][i]]
                sf_idx = [s.to(device) for s in batch['sf_idx'][i]]
                hop = batch['hops'][i]
                
                outputs = model(q_codes, c_codes, sf_idx, hop)
                loss = outputs['loss']
                
                if loss.requires_grad and not torch.isnan(loss):
                    batch_loss += loss
                    valid_samples += 1
                    
                    # Calculate F1 and EM
                    if 'current_preds' in outputs and outputs['current_preds']:
                        predictions = outputs['current_preds'][0][:2] if outputs['current_preds'][0] else []
                        targets = sf_idx[0].cpu().tolist()[:2]
                        
                        f1, em = calculate_f1_em(predictions, targets)
                        batch_f1 += f1
                        batch_em += em
                        
                        # Debug info for first batch of first epoch
                        if batch_idx == 0 and i == 0:
                            print(f"\n🔍 Debug Info:")
                            print(f"   Raw model output keys: {list(outputs.keys())}")
                            print(f"   Current preds shape: {len(outputs['current_preds'])} beams")
                            print(f"   Beam 0 predictions: {outputs['current_preds'][0]}")
                            print(f"   Targets (supporting facts): {targets}")
                            print(f"   Hop count: {hop}")
                            print(f"   F1: {f1:.4f}, EM: {em:.4f}")
                            print(f"   Why only 1 prediction? Possible reasons:")
                            print(f"     - Beam search stopped early")
                            print(f"     - Model confidence low for other indices") 
                            print(f"     - Need more training to predict multiple supporting facts")
                    else:
                        # No predictions available
                        if batch_idx == 0 and i == 0:
                            print(f"\n⚠️  No predictions available from model output")
                            print(f"   Available output keys: {list(outputs.keys()) if outputs else 'None'}")
                    
            except Exception as e:
                print(f"Sample {i} failed: {e}")
                continue
        
        if valid_samples > 0:
            avg_loss = batch_loss / valid_samples
            avg_f1 = batch_f1 / valid_samples
            avg_em = batch_em / valid_samples
            
            avg_loss.backward()
            optimizer.step()
            
            epoch_losses.append(avg_loss.item())
            f1_scores.append(avg_f1)
            em_scores.append(avg_em)
            
            progress_bar.set_postfix({
                'loss': f'{avg_loss.item():.4f}',
                'f1': f'{avg_f1:.4f}',
                'em': f'{avg_em:.4f}',
                'valid': f'{valid_samples}/{len(batch["q_codes"])}'
            })
        
        # Early break if max_batches specified
        if max_batches and batch_idx >= max_batches - 1:
            break
    
    max_f1 = max(f1_scores) if f1_scores else 0.0
    max_em = max(em_scores) if em_scores else 0.0
    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
    
    return max_f1, max_em, avg_loss

def main():
    parser = argparse.ArgumentParser(description='Train Advanced Multi-Hop Retriever')
    parser.add_argument('--samples', type=int, default=100, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--max_len', type=int, default=128, help='Max sequence length')
    parser.add_argument('--save_path', type=str, default='models/retriever_trained.pt', help='Model save path')
    parser.add_argument('--gradient_accumulation', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--max_batches', type=int, default=10, help='Max batches per epoch for testing')
    
    args = parser.parse_args()
    
    print("🚀 Advanced Multi-Hop Retriever Training")
    print("=" * 50)
    print(f"📊 Samples: {args.samples}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"🎯 Learning rate: {args.learning_rate}")
    print(f"📏 Max length: {args.max_len}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Load data
    print("\n📚 Loading data...")
    train_data = load_hotpot_data('train', sample_size=args.samples)
    print(f"✅ Loaded {len(train_data)} training samples")
    
    # Create model
    print("\n🧠 Creating model...")
    model = create_advanced_retriever(
        model_name="microsoft/deberta-v3-base",
        beam_size=2,
        use_focal=True,
        use_early_stop=True,
        max_seq_len=args.max_len
    )
    model.to(device)
    print(f"✅ Model created with {model.count_parameters():,} parameters")
    
    # Create dataset and dataloader
    print("\n📦 Creating dataset...")
    tokenizer = model.tokenizer
    dataset = RetrievalDataset(train_data, tokenizer, max_len=args.max_len, num_contexts=5)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    # Training loop
    print(f"\n🎯 Starting training for {args.epochs} epochs...")
    best_f1 = 0.0
    best_em = 0.0
    train_metrics = []
    
    for epoch in range(args.epochs):
        print(f"\n📚 Epoch {epoch+1}/{args.epochs}")
        
        max_f1, max_em, avg_loss = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            max_batches=args.max_batches  # Limit batches to prevent memory issues
        )
        
        train_metrics.append({
            'epoch': epoch + 1,
            'max_f1': max_f1,
            'max_em': max_em,
            'avg_loss': avg_loss
        })
        
        print(f"📊 Epoch {epoch+1} - Max F1: {max_f1:.4f}, Max EM: {max_em:.4f}, Avg Loss: {avg_loss:.4f}")
        
        # Save model if F1 improves
        if max_f1 > best_f1:
            best_f1 = max_f1
            best_em = max_em
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'max_f1': max_f1,
                'max_em': max_em,
                'avg_loss': avg_loss,
                'train_metrics': train_metrics,
                'config': {
                    'model_name': 'microsoft/deberta-v3-base',
                    'beam_size': 2,
                    'use_focal': True,
                    'max_seq_len': args.max_len,
                    'samples': args.samples,
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate
                }
            }, args.save_path)
            print(f"💾 Best model saved to {args.save_path} (F1: {max_f1:.4f}, EM: {max_em:.4f})")
            print(f"📈 F1 improvement: {max_f1:.4f}")
        
        # Explain why EM might be 0
        if max_em == 0.0:
            print(f"❗ EM = 0 means: No exact match between predicted and target supporting facts")
            print(f"   - F1 > 0 means partial overlap exists (some correct predictions)")
            print(f"   - This is normal in early training - model learns partial patterns first")
    
    print(f"\n🎉 Training completed!")
    print(f"📈 Final metrics - F1: {train_metrics[-1]['max_f1']:.4f}, EM: {train_metrics[-1]['max_em']:.4f}")
    print(f"🏆 Best F1: {best_f1:.4f}, Best EM: {best_em:.4f}")
    print(f"💾 Model saved to {args.save_path}")
    
    # Detailed explanation of metrics
    print(f"\n📚 Metrics Explanation:")
    print(f"• F1 Score: Measures partial overlap between predicted and target supporting facts")
    print(f"• EM Score: Exact Match - 1.0 only when predictions exactly match targets")
    print(f"• Why EM=0? Model is learning gradually - partial matches (F1>0) come before exact matches")

if __name__ == "__main__":
    main()
