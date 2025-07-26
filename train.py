#!/usr/bin/env python3
"""
Main Training Script for Advanced Multi-Hop Retriever
Usage: python train.py [--dataset train/dev] [--samples N] [--epochs N] [--batch_size N] [--gpu]
"""
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaV2Tokenizer
from tqdm import tqdm
import random

from models.advanced_retriever import create_advanced_retriever
from utils.data_loader import load_hotpot_data

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        print(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🍎 Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    return device

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
        
        # Tokenize question and contexts together with proper format: [CLS] + Q + C + [SEP]
        q_tokens_list = []
        c_tokens_list = []
        
        for ctx in selected_contexts[:self.num_contexts]:
            ctx_text = f"{ctx['title']}: {ctx['text']}"
            
            # Create input with format: [CLS] + Q + C + [SEP] (no SEP between Q and C)
            combined_text = question + " " + ctx_text
            combined_tokens = self.tokenizer(
                combined_text,
                max_length=self.max_len,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Extract tokens for backward compatibility
            input_ids = combined_tokens['input_ids'].view(-1)
            
            # For backward compatibility, split roughly at question length
            question_tokens = self.tokenizer(
                question,
                max_length=self.max_len // 2,
                truncation=True,
                add_special_tokens=False,
                return_tensors='pt'
            )['input_ids'].view(-1)
            
            # Format: [CLS] Q C [SEP] - use the combined tokens directly
            # Extract question part for compatibility (first part of combined)
            q_len = len(question_tokens) + 1  # +1 for CLS
            question_part = input_ids[:q_len]
            
            # Extract context part (remaining part)
            context_part = input_ids[q_len:]
            
            # Pad question part to consistent length
            q_max_len = self.max_len // 2
            if len(question_part) > q_max_len:
                question_part = question_part[:q_max_len]
            else:
                padding_len = q_max_len - len(question_part)
                question_part = torch.cat([question_part, torch.full((padding_len,), self.tokenizer.pad_token_id)])
            
            q_tokens_list.append(question_part)
            c_tokens_list.append(input_ids)  # Full sequence: [CLS] + Q + C + [SEP]
        
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
            'q_codes': q_tokens_list,  # List of question tokens for each context
            'c_codes': c_tokens_list,  # List of full sequence tokens
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

def evaluate_model(model, dataloader, device, max_batches=None):
    """Evaluate model and return average F1, EM, and loss"""
    model.eval()
    eval_losses = []
    eval_f1_scores = []
    eval_em_scores = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for batch_idx, batch in enumerate(progress_bar):
            batch_f1 = 0
            batch_em = 0
            valid_samples = 0
            
            # Process each sample in batch individually
            for i in range(len(batch['q_codes'])):
                try:
                    # Move data to device efficiently
                    q_codes = [q.to(device, non_blocking=True) for q in batch['q_codes'][i]]
                    c_codes = [c.to(device, non_blocking=True) for c in batch['c_codes'][i]]
                    sf_idx = [s.to(device, non_blocking=True) for s in batch['sf_idx'][i]]
                    hop = batch['hops'][i]
                    
                    outputs = model(q_codes, c_codes, sf_idx, hop)
                    loss = outputs['loss']
                    
                    if not torch.isnan(loss):
                        valid_samples += 1
                        eval_losses.append(loss.item())
                        
                        # Calculate F1 and EM
                        if 'final_preds' in outputs and outputs['final_preds']:
                            predictions = outputs['final_preds'][0] if outputs['final_preds'][0] else []
                        elif 'current_preds' in outputs and outputs['current_preds']:
                            predictions = outputs['current_preds'][0] if outputs['current_preds'][0] else []
                        else:
                            predictions = []
                        
                        targets = sf_idx[0].cpu().tolist()
                        f1, em = calculate_f1_em(predictions, targets)
                        batch_f1 += f1
                        batch_em += em
                        eval_f1_scores.append(f1)
                        eval_em_scores.append(em)
                        
                except Exception as e:
                    print(f"Evaluation sample {i} failed: {e}")
                    continue
            
            if valid_samples > 0:
                avg_f1 = batch_f1 / valid_samples
                avg_em = batch_em / valid_samples
                
                progress_bar.set_postfix({
                    'loss': f'{eval_losses[-1]:.4f}' if eval_losses else '0.0000',
                    'f1': f'{avg_f1:.4f}',
                    'em': f'{avg_em:.4f}'
                })
            
            # Early break if max_batches specified
            if max_batches and batch_idx >= max_batches - 1:
                break
    
    # Calculate averages
    avg_f1 = sum(eval_f1_scores) / len(eval_f1_scores) if eval_f1_scores else 0.0
    avg_em = sum(eval_em_scores) / len(eval_em_scores) if eval_em_scores else 0.0
    avg_loss = sum(eval_losses) / len(eval_losses) if eval_losses else 0.0
    
    return avg_f1, avg_em, avg_loss

def train_epoch(model, dataloader, optimizer, device, max_batches=None, scaler=None):
    """Train for one epoch with GPU optimization and comprehensive metrics"""
    model.train()
    epoch_losses = []
    f1_scores = []
    em_scores = []
    
    # Calculate checkpoint intervals for reporting
    total_batches = max_batches if max_batches else len(dataloader)
    checkpoint_intervals = [int(total_batches * p) for p in [0.25, 0.5, 0.75]]
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(progress_bar):
        
        batch_f1 = 0
        batch_em = 0
        valid_samples = 0
        
        # Process each sample in batch individually
        for i in range(len(batch['q_codes'])):
            try:
                # Zero gradients for each sample
                optimizer.zero_grad()
                
                # Move data to device efficiently
                q_codes = [q.to(device, non_blocking=True) for q in batch['q_codes'][i]]
                c_codes = [c.to(device, non_blocking=True) for c in batch['c_codes'][i]]
                sf_idx = [s.to(device, non_blocking=True) for s in batch['sf_idx'][i]]
                hop = batch['hops'][i]
                
                # Mixed precision forward pass
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(q_codes, c_codes, sf_idx, hop)
                        loss = outputs['loss']
                else:
                    outputs = model(q_codes, c_codes, sf_idx, hop)
                    loss = outputs['loss']
                
                if loss.requires_grad and not torch.isnan(loss):
                    # Backward pass for individual sample
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    
                    valid_samples += 1
                    epoch_losses.append(loss.item())
                    
                    # Calculate F1 and EM only after ALL hops are completed
                    if 'final_preds' in outputs and outputs['final_preds']:
                        # Use final predictions after all hops are completed
                        final_predictions = outputs['final_preds'][0] if outputs['final_preds'][0] else []
                        targets = sf_idx[0].cpu().tolist()
                        
                        f1, em = calculate_f1_em(final_predictions, targets)
                        batch_f1 += f1
                        batch_em += em
                        f1_scores.append(f1)
                        em_scores.append(em)
                        
                    elif 'current_preds' in outputs and outputs['current_preds']:
                        # Fallback to current predictions if final not available
                        predictions = outputs['current_preds'][0] if outputs['current_preds'][0] else []
                        targets = sf_idx[0].cpu().tolist()
                        
                        f1, em = calculate_f1_em(predictions, targets)
                        batch_f1 += f1
                        batch_em += em
                        f1_scores.append(f1)
                        em_scores.append(em)
                    
            except Exception as e:
                print(f"Sample {i} failed: {e}")
                continue
        
        if valid_samples > 0:
            avg_f1 = batch_f1 / valid_samples
            avg_em = batch_em / valid_samples
            
            progress_bar.set_postfix({
                'loss': f'{epoch_losses[-1]:.4f}' if epoch_losses else '0.0000',
                'f1': f'{avg_f1:.4f}',
                'em': f'{avg_em:.4f}',
                'valid': f'{valid_samples}/{len(batch["q_codes"])}'
            })
        
        # Clear GPU cache periodically
        if torch.cuda.is_available() and batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        # Report metrics at checkpoints (25%, 50%, 75%)
        if batch_idx in checkpoint_intervals and f1_scores and em_scores:
            current_avg_f1 = sum(f1_scores) / len(f1_scores)
            current_avg_em = sum(em_scores) / len(em_scores)
            current_avg_loss = sum(epoch_losses) / len(epoch_losses)
            checkpoint_pct = int((batch_idx / total_batches) * 100)
            
            print(f"\n📊 {checkpoint_pct}% Checkpoint ({batch_idx+1}/{total_batches} batches):")
            print(f"   Average Loss: {current_avg_loss:.4f}")
            print(f"   Average F1: {current_avg_f1:.4f}")
            print(f"   Average EM: {current_avg_em:.4f}")
            print(f"   Max F1 so far: {max(f1_scores):.4f}")
            print(f"   Max EM so far: {max(em_scores):.4f}")
            print(f"   Samples processed: {len(f1_scores)}")
        
        # Early break if max_batches specified
        if max_batches and batch_idx >= max_batches - 1:
            break
    
    # Calculate final averages and max values
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    max_f1 = max(f1_scores) if f1_scores else 0.0
    max_em = max(em_scores) if em_scores else 0.0
    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
    
    return avg_f1, avg_em, max_f1, max_em, avg_loss

def main():
    parser = argparse.ArgumentParser(description='Train Advanced Multi-Hop Retriever - Paper Configuration')
    # ========== PAPER CONFIGURATION (DEFAULT VALUES) ==========
    parser.add_argument('--dataset', type=str, default='train', choices=['train', 'dev'], 
                       help='Dataset to use for training (train or dev)')
    parser.add_argument('--samples', type=int, default=None, help='Number of training samples (None = FULL DATASET)')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs (Paper: 16, Test: 2)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (Paper: 1)')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate (Paper: 2e-5)')
    parser.add_argument('--max_len', type=int, default=512, help='Max sequence length (Paper: 512)')
    parser.add_argument('--save_path', type=str, default='models/deberta_v3_paper_full.pt', help='Model save path')
    parser.add_argument('--gradient_accumulation', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--max_batches', type=int, default=None, help='Max batches per epoch (None = FULL DATASET)')
    parser.add_argument('--gpu', action='store_true', help='Force GPU usage')
    parser.add_argument('--mixed_precision', action='store_true', default=False, help='Use mixed precision (typically False with gradient checkpointing)')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False, help='Use gradient checkpointing for memory efficiency (recommended)')
    
    args = parser.parse_args()
    
    # Display configuration based on dataset
    dataset_info = {
        'train': ('FULL HOTPOT QA TRAIN (~90K samples)', 'training'),
        'dev': ('FULL HOTPOT QA DEV (~7.4K samples)', 'development/validation')
    }
    
    dataset_name, dataset_purpose = dataset_info[args.dataset]
    
    print(f"📖 TRAINING ON {args.dataset.upper()} SET")
    print("=" * 60)
    print(f"📊 Dataset: {dataset_name if args.samples is None else f'{args.samples} {args.dataset} samples'}")
    print(f"🎯 Purpose: Using {dataset_purpose} data for training")
    if args.dataset == 'dev':
        print(f"⚠️  NOTE: Training on DEV set - typically used for testing/validation")
        print(f"📈 This can be useful for quick experiments or debugging")
    print(f"🔄 Epochs: {args.epochs} {'✅ (Paper: 16)' if args.epochs == 16 else '⚠️ (Paper: 16)'}")
    print(f"📦 Batch size: {args.batch_size} {'✅ (Paper: 1)' if args.batch_size == 1 else '⚠️ (Paper: 1)'}")
    print(f"🎯 Learning rate: {args.learning_rate} {'✅ (Paper: 2e-5)' if args.learning_rate == 2e-5 else '⚠️ (Paper: 2e-5)'}")
    print(f"📏 Max length: {args.max_len} {'✅ (Paper: 512)' if args.max_len == 512 else '⚠️ (Paper: 512)'}")
    print(f"⚡ Mixed precision: {args.mixed_precision}")
    print(f"🔄 Gradient checkpointing: {args.gradient_checkpointing}")
    print(f"🔢 Max batches: {'UNLIMITED' if args.max_batches is None else args.max_batches}")
    
    # Auto-adjust save path based on dataset
    if args.save_path == 'models/deberta_v3_paper_full.pt':  # Default path
        if args.dataset == 'dev':
            args.save_path = 'models/deberta_v3_paper_dev.pt'
        else:
            args.save_path = 'models/deberta_v3_paper_train.pt'
    
    print(f"💾 Save path: {args.save_path}")
    print("=" * 60)
    
    # Paper compliance check
    is_paper_config = (
        args.samples is None and
        args.epochs == 16 and
        args.batch_size == 1 and
        args.learning_rate == 2e-5 and
        args.max_len == 512
    )
    
    if is_paper_config:
        print(f"🎉 EXACT PAPER CONFIGURATION DETECTED! (using {args.dataset.upper()} set)")
        print(f"📖 Training with exact settings from the paper on {args.dataset} data")
    else:
        print("⚠️  Custom configuration - differs from paper settings")
        print("📝 To use exact paper config, run without arguments")
    print("=" * 60)
    
    # Device selection
    device = get_device()
    if args.gpu and not torch.cuda.is_available():
        print("⚠️  GPU requested but not available, using CPU")
    
    # Load data - using the selected dataset
    print(f"\n📚 Loading {args.dataset.upper()} data for training...")
    train_data = load_hotpot_data(args.dataset, sample_size=args.samples)
    print(f"✅ Loaded {len(train_data)} {args.dataset.upper()} samples for training")
    if args.dataset == 'dev':
        print(f"⚠️  NOTE: Using DEV set as training data!")
    
    # Create model
    print("\n🧠 Creating model...")
    model = create_advanced_retriever(
        model_name="microsoft/deberta-v3-base",
        beam_size=2,
        use_focal=True,
        use_early_stop=True,
        max_seq_len=args.max_len,
        gradient_checkpointing=args.gradient_checkpointing
    )
    model.to(device)
    print(f"✅ Model created with {model.count_parameters():,} parameters")
    
    # Mixed precision setup
    scaler = None
    if args.mixed_precision and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        print("⚡ Using mixed precision training")
    
    # Create dataset and dataloader
    print("\n📦 Creating dataset...")
    tokenizer = model.tokenizer
    # FULL CONTEXT PROCESSING: Use all 10 contexts from HotpotQA dataset
    num_contexts = 10  # HotpotQA provides 10 contexts per sample - use them all
    print(f"🔧 Using {num_contexts} contexts per sample for FULL context processing")
    dataset = RetrievalDataset(train_data, tokenizer, max_len=args.max_len, num_contexts=num_contexts)
    
    # Memory optimization for 10 contexts - disable pin_memory and multiprocessing
    pin_memory = False  # Disable to save GPU memory
    num_workers = 0  # Disable multiprocessing to save memory
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    
    # Optimizer with better settings for GPU
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    # Training loop
    print(f"\n🎯 Starting training for {args.epochs} epochs...")
    if torch.cuda.is_available():
        print(f"🔥 GPU Memory before training: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    
    best_avg_f1 = 0.0
    best_avg_em = 0.0
    best_max_f1 = 0.0
    best_max_em = 0.0
    train_metrics = []
    
    for epoch in range(args.epochs):
        print(f"\n📚 Epoch {epoch+1}/{args.epochs}")
        
        # GPU memory info
        if torch.cuda.is_available():
            print(f"🔥 GPU Memory before epoch: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        avg_f1, avg_em, max_f1, max_em, avg_loss = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            max_batches=args.max_batches,
            scaler=scaler
        )
        
        # Store detailed metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'avg_f1': avg_f1,
            'avg_em': avg_em,
            'max_f1': max_f1,
            'max_em': max_em,
            'avg_loss': avg_loss
        }
        train_metrics.append(epoch_metrics)
        
        print(f"\n🎯 EPOCH {epoch+1}/{args.epochs} - FINAL METRICS")
        print(f"=" * 50)
        print(f"📊 Average Metrics (Trung bình toàn epoch):")
        print(f"   • Average F1 Score: {avg_f1:.4f}")
        print(f"   • Average EM Score: {avg_em:.4f}")
        print(f"   • Average Loss: {avg_loss:.4f}")
        print(f"📈 Best Individual Predictions This Epoch:")
        print(f"   • Max F1 Score: {max_f1:.4f}")
        print(f"   • Max EM Score: {max_em:.4f}")
        print(f"=" * 50)
        
        # GPU memory info
        if torch.cuda.is_available():
            print(f"🔥 GPU Memory after epoch: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        # Save checkpoint after every epoch with complete metrics
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'metrics': epoch_metrics,
            'best_metrics': {
                'best_avg_f1': max(best_avg_f1, avg_f1),
                'best_avg_em': max(best_avg_em, avg_em),
                'best_max_f1': max(best_max_f1, max_f1),
                'best_max_em': max(best_max_em, max_em)
            },
            'train_metrics_history': train_metrics,
            'config': {
                'model_name': 'microsoft/deberta-v3-base',
                'beam_size': 2,
                'use_focal': True,
                'max_seq_len': args.max_len,
                'dataset': args.dataset,
                'samples': args.samples,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'num_contexts': num_contexts
            }
        }
        
        # Always save checkpoint after each epoch
        torch.save(checkpoint_data, args.save_path)
        print(f"💾 Checkpoint saved to {args.save_path}")
        
        # Update best metrics if improved
        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            best_avg_em = avg_em
            best_max_f1 = max_f1
            best_max_em = max_em
            print(f"🏆 NEW BEST! Average F1 improved: {avg_f1:.4f}")
        elif avg_f1 == best_avg_f1 and max_f1 > best_max_f1:
            # Tie-breaker: if avg F1 is same, prefer higher max F1
            best_avg_f1 = avg_f1
            best_avg_em = avg_em
            best_max_f1 = max_f1
            best_max_em = max_em
            print(f"🏆 NEW BEST! Max F1 improved: {max_f1:.4f} (avg F1 tied)")
        
        print(f"📈 Current Best Metrics So Far:")
        print(f"   • Best Avg F1: {max(best_avg_f1, avg_f1):.4f}")
        print(f"   • Best Avg EM: {max(best_avg_em, avg_em):.4f}")
        print(f"   • Best Max F1: {max(best_max_f1, max_f1):.4f}")
        print(f"   • Best Max EM: {max(best_max_em, max_em):.4f}")
        
        # Explain metrics for clarity
        print(f"\n📝 Giải thích Metrics:")
        print(f"   • F1 Score = Độ chính xác tổng hợp (precision + recall)")
        print(f"   • EM Score = Exact Match (1.0 chỉ khi dự đoán hoàn toàn chính xác)")
        print(f"   • Average = Trung bình trên toàn bộ samples")
        print(f"   • Max = Điểm số cao nhất trong epoch này")
        if avg_em == 0.0 and avg_f1 > 0.0:
            print(f"   • EM=0 nhưng F1>0: Mô hình đang học từng phần, chưa dự đoán hoàn toàn chính xác")
        
        print(f"-" * 60)
    
    print(f"\n🎉 TRAINING HOÀN THÀNH!")
    print(f"=" * 60)
    print(f"📈 Metrics Epoch Cuối Cùng:")
    print(f"   • Average F1: {train_metrics[-1]['avg_f1']:.4f}")
    print(f"   • Average EM: {train_metrics[-1]['avg_em']:.4f}")
    print(f"   • Average Loss: {train_metrics[-1]['avg_loss']:.4f}")
    print(f"🏆 Metrics Tốt Nhất Đạt Được:")
    print(f"   • Best Average F1: {max(best_avg_f1, train_metrics[-1]['avg_f1']):.4f}")
    print(f"   • Best Average EM: {max(best_avg_em, train_metrics[-1]['avg_em']):.4f}")
    print(f"   • Best Max F1: {max(best_max_f1, train_metrics[-1]['max_f1']):.4f}")
    print(f"   • Best Max EM: {max(best_max_em, train_metrics[-1]['max_em']):.4f}")
    print(f"💾 Model và checkpoint đã lưu: {args.save_path}")
    print(f"=" * 60)
    
    # Training summary in Vietnamese
    dataset_summary = f"Trained on {args.dataset.upper()} set"
    if args.dataset == 'dev':
        dataset_summary += " (sử dụng development data để training)"
    
    print(f"\n📚 Tóm Tắt Training:")
    print(f"• Dataset: {dataset_summary}")
    print(f"• Số samples: {len(train_data):,}")
    print(f"• Số epochs: {args.epochs}")
    print(f"• Checkpoint bao gồm: model weights, optimizer state, metrics history, config")
    print(f"• Có thể load lại model bằng: python evaluate_checkpoint.py --checkpoint_path {args.save_path}")
    print(f"\n📊 Ý Nghĩa Metrics:")
    print(f"• F1 Score: Đo độ chính xác khi có partial match giữa dự đoán và target")
    print(f"• EM Score: Exact Match - chỉ = 1.0 khi dự đoán hoàn toàn chính xác")
    print(f"• Average: Hiệu suất trung bình trên toàn bộ dataset")
    print(f"• Max: Hiệu suất tốt nhất trong từng epoch")
    print(f"• Tại sao EM=0? Model học dần dần - partial matches (F1>0) xuất hiện trước exact matches")

if __name__ == "__main__":
    main()
