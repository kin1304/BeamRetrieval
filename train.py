#!/usr/bin/env python3
"""
Script Training Chính cho Advanced Multi-Hop Retriever
Cách dùng: python train.py [--dataset train/dev] [--samples N] [--epochs N] [--batch_size N] [--gpu]
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
    """Lấy thiết bị tốt nhất có sẵn"""
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
    """Tính toán F1 score và Exact Match"""
    if not predictions or not targets:
        return 0.0, 0.0
    
    # Chuyển đổi thành sets để tính F1
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
    """Dataset cho training multi-hop retrieval"""
    
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def _split_context_to_paragraphs(self, context_item):
        """
        🚀 TỐI ƯU: HotpotQA format: [title, [sentence1, sentence2, ...]]
        Gộp title + tất cả sentences thành 1 paragraph lớn duy nhất
        """
        title = context_item[0]        # Title của context
        sentences = context_item[1]    # List các câu
        
        # Gộp tất cả sentences thành 1 đoạn văn lớn
        combined_text = ' '.join(sentence.strip() for sentence in sentences if sentence.strip())
        
        # Tạo paragraph hoàn chỉnh: Title + Combined text
        if combined_text:
            paragraph = f"{title}. {combined_text}"
        else:
            paragraph = f"{title}. No content available."
        
        # Return list với 1 paragraph duy nhất
        return [paragraph]
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        contexts = item['contexts']
        supporting_facts = item.get('supporting_facts', [])
        
        # 🆕 SỬ DỤNG TẤT CẢ CONTEXTS - không giới hạn
        selected_contexts = contexts  # Sử dụng tất cả contexts có sẵn
        num_contexts = len(selected_contexts)  # Dynamic cho mỗi sample
        
        # 🚀 TỐI ƯU: Chia đoạn văn TRƯỚC khi tokenization (không decode/re-tokenize!)
        q_tokens_list = []
        p_tokens_list = []  # Mới: chuỗi đoạn văn trực tiếp
        context_to_paragraph_mapping = []  # Ánh xạ chỉ số đoạn văn tới chỉ số context gốc
        
        # Tokenize câu hỏi một lần (token sạch không có special tokens)
        question_tokens = self.tokenizer(
            question,
            max_length=self.max_len // 2,
            truncation=True,
            add_special_tokens=False,  # Không có [CLS], [SEP] cho token sạch
            return_tensors='pt'
        )['input_ids'].view(-1)
        
        # Đệm question tokens tới độ dài nhất quán
        q_max_len = self.max_len // 2
        if len(question_tokens) > q_max_len:
            question_tokens = question_tokens[:q_max_len]
        else:
            padding_len = q_max_len - len(question_tokens)
            question_tokens = torch.cat([question_tokens, torch.full((padding_len,), self.tokenizer.pad_token_id)])
        
        # Xử lý từng context và chia thành đoạn văn
        for ctx_idx, ctx in enumerate(selected_contexts):  # 🆕 Sử dụng tất cả selected_contexts
            # HotpotQA format: ctx = [title, [sentences]]
            # Không cần gộp title + text, xử lý trực tiếp!
            paragraphs = self._split_context_to_paragraphs(ctx)
            
            # Tokenize từng paragraph: [CLS] + Q + P + [SEP]
            for paragraph_text in paragraphs:
                combined_text = question + " " + paragraph_text
                para_tokens = self.tokenizer(
                    combined_text,
                    max_length=self.max_len,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )['input_ids'].view(-1)
                
                p_tokens_list.append(para_tokens)
                context_to_paragraph_mapping.append(ctx_idx)
        
        # Lưu question tokens một lần (dùng lại cho tất cả đoạn văn)
        q_tokens_list.append(question_tokens)
        
        # Chỉ số supporting facts (ánh xạ tới chỉ số context gốc)
        sf_indices = []
        for i, ctx in enumerate(selected_contexts):  # 🆕 Dùng tất cả selected_contexts
            if any(ctx[0] == sf[0] for sf in supporting_facts):  # ctx[0] = title
                sf_indices.append(i)
        
        # Đảm bảo ít nhất 1 supporting fact
        if not sf_indices:
            sf_indices.append(0)
        sf_indices = sf_indices[:3]  # Tối đa 3 supporting facts
        
        # 🆕 KHÔNG ĐỆM - để dynamic length
        # while len(sf_indices) < 2:
        #     sf_indices.append(sf_indices[0])
        
        return {
            'q_codes': q_tokens_list,  # Token câu hỏi sạch đơn (không có [CLS], [SEP])
            'p_codes': p_tokens_list,  # MỚI: Chuỗi đoạn văn trực tiếp [CLS] + Q + P + [SEP]
            'context_mapping': context_to_paragraph_mapping,  # MỚI: Ánh xạ Đoạn văn → Context
            'sf_idx': [torch.tensor(sf_indices, dtype=torch.long)],
            'hop': len(sf_indices)
        }

def collate_fn(batch):
    """🚀 TỐI ƯU: Hàm collate tùy chỉnh cho định dạng paragraph-based mới"""
    return {
        'q_codes': [item['q_codes'] for item in batch],
        'p_codes': [item['p_codes'] for item in batch],  # MỚI: Chuỗi đoạn văn
        'context_mapping': [item['context_mapping'] for item in batch],  # MỚI: Ánh xạ Đoạn văn → Context
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

    """Train một epoch với tối ưu hóa GPU"""
    model.train()
    epoch_losses = []
    f1_scores = []
    em_scores = []
    
    # Tính toán checkpoint 30% để báo cáo

    total_batches = max_batches if max_batches else len(dataloader)
    checkpoint_intervals = [int(total_batches * p) for p in [0.25, 0.5, 0.75]]
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(progress_bar):
        
        batch_f1 = 0
        batch_em = 0
        valid_samples = 0
        
        # Xử lý từng sample trong batch riêng lẻ
        for i in range(len(batch['q_codes'])):
            try:
                # Zero gradients cho từng sample
                optimizer.zero_grad()
                
                # 🚀 TỐI ƯU: Di chuyển dữ liệu lên device hiệu quả (định dạng mới)
                q_codes = [q.to(device, non_blocking=True) for q in batch['q_codes'][i]]
                p_codes = [p.to(device, non_blocking=True) for p in batch['p_codes'][i]]  # MỚI: Chuỗi đoạn văn
                context_mapping = batch['context_mapping'][i]  # MỚI: Thông tin ánh xạ
                sf_idx = [s.to(device, non_blocking=True) for s in batch['sf_idx'][i]]
                hop = batch['hops'][i]
                
                # Mixed precision forward pass
                if scaler is not None:

                    with torch.cuda.amp.autocast('cuda'):
                        # 🚀 TỐI ƯU: Sử dụng p_codes (chuỗi đoạn văn) và context_mapping
                        outputs = model(q_codes, p_codes, sf_idx, hop, context_mapping=context_mapping)
                        loss = outputs['loss']
                else:
                    outputs = model(q_codes, p_codes, sf_idx, hop, context_mapping=context_mapping)
                    loss = outputs['loss']
                
                if loss.requires_grad and not torch.isnan(loss):
                    # Backward pass cho sample riêng lẻ
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    
                    valid_samples += 1
                    epoch_losses.append(loss.item())
                    
                    # Tính toán F1 và EM chỉ sau khi TẤT CẢ hops đã hoàn thành
                    predictions = []
                    targets = sf_idx[0].cpu().tolist()
                    
                    if 'final_preds' in outputs and outputs['final_preds']:
                        # Sử dụng final predictions sau khi tất cả hops hoàn thành
                        predictions = outputs['final_preds'][0] if len(outputs['final_preds']) > 0 else []
                        
                    elif 'current_preds' in outputs and outputs['current_preds']:
                        # Fallback tới current predictions nếu final không có
                        predictions = outputs['current_preds'][0] if len(outputs['current_preds']) > 0 else []
                    
                    # Luôn tính F1/EM kể cả khi predictions rỗng (để debug)
                    f1, em = calculate_f1_em(predictions, targets)
                    batch_f1 += f1
                    batch_em += em
                    f1_scores.append(f1)
                    em_scores.append(em)
                    
                    # Thông tin debug cho vài sample đầu
                    if len(f1_scores) <= 3:
                        print(f"\n🔍 Debug Sample {len(f1_scores)}:")
                        print(f"   Predictions: {predictions}")
                        print(f"   Targets: {targets}")
                        print(f"   F1: {f1:.4f}, EM: {em:.4f}")
                        print(f"   Available outputs: {list(outputs.keys())}")
                    
            except Exception as e:
                print(f"Sample {i} thất bại: {e}")
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
        
        # Xóa GPU cache định kỳ
        if torch.cuda.is_available() and batch_idx % 10 == 0:
            torch.cuda.empty_cache()

        # Báo cáo metrics theo chu kỳ (mỗi 25% và 50%)
        if f1_scores and em_scores:
            current_avg_f1 = sum(f1_scores) / len(f1_scores)
            current_avg_em = sum(em_scores) / len(em_scores)
            current_avg_loss = sum(epoch_losses) / len(epoch_losses)
            current_max_f1 = max(f1_scores)
            current_max_em = max(em_scores)
            
            # Báo cáo tại 25%, 50%, 75% 
            progress_checkpoints = [int(total_batches * p) for p in [0.25, 0.5, 0.75]]
            
            if batch_idx in progress_checkpoints:
                progress_pct = int((batch_idx / total_batches) * 100)
                print(f"{progress_pct}% ({batch_idx+1}/{total_batches}) - Loss: {current_avg_loss:.4f}, F1: {current_max_f1:.4f}, EM: {current_max_em:.4f}")

        
        # Dừng sớm nếu max_batches được chỉ định
        if max_batches and batch_idx >= max_batches - 1:
            break
    
    # Calculate final averages and max values
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    max_f1 = max(f1_scores) if f1_scores else 0.0
    max_em = max(em_scores) if em_scores else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
    

    return max_f1, max_em, avg_f1, avg_em, avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train Advanced Multi-Hop Retriever - Cấu hình Paper')
    # ========== CẤU HÌNH PAPER (GIÁ TRỊ MẶC ĐỊNH) ==========
    parser.add_argument('--dataset', type=str, default='train', choices=['train', 'dev'], 
                       help='Dataset sử dụng cho training (train hoặc dev)')
    parser.add_argument('--samples', type=int, default=None, help='Số lượng training samples (None = FULL DATASET)')
    parser.add_argument('--epochs', type=int, default=2, help='Số epochs (Paper: 16, Test: 2)')
    parser.add_argument('--batch_size', type=int, default=1, help='Kích thước batch (Paper: 1)')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate (Paper: 2e-5)')
    parser.add_argument('--max_len', type=int, default=512, help='Độ dài chuỗi tối đa (Paper: 512)')
    parser.add_argument('--save_path', type=str, default='models/deberta_v3_paper_full.pt', help='Đường dẫn lưu model')
    parser.add_argument('--gradient_accumulation', type=int, default=1, help='Bước tích lũy gradient')
    parser.add_argument('--max_batches', type=int, default=None, help='Số batch tối đa mỗi epoch (None = FULL DATASET)')
    parser.add_argument('--gpu', action='store_true', help='Ép buộc sử dụng GPU')
    parser.add_argument('--mixed_precision', action='store_true', default=False, help='Sử dụng mixed precision (thường False với gradient checkpointing)')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False, help='Sử dụng gradient checkpointing để tiết kiệm bộ nhớ (khuyến nghị)')
    
    args = parser.parse_args()
    
    # Hiển thị cấu hình dựa trên dataset
    dataset_info = {
        'train': ('FULL HOTPOT QA TRAIN (~90K samples)', 'training'),
        'dev': ('FULL HOTPOT QA DEV (~7.4K samples)', 'development/validation')
    }
    
    dataset_name, dataset_purpose = dataset_info[args.dataset]
    
    # Lựa chọn thiết bị
    device = get_device()
    # Tự động điều chỉnh đường dẫn lưu dựa trên dataset
    if args.save_path == 'models/deberta_v3_paper_full.pt':  # Đường dẫn mặc định
        if args.dataset == 'dev':
            args.save_path = 'models/deberta_v3_paper_dev.pt'
        else:
            args.save_path = 'models/deberta_v3_paper_train.pt'
    
    # Tải dữ liệu
    train_data = load_hotpot_data(args.dataset, sample_size=args.samples)
    
    # 🆕 THỐNG KÊ CONTEXTS - chỉ để thông tin, không giới hạn
    if train_data:
        context_lengths = [len(item['contexts']) for item in train_data if 'contexts' in item]
        if context_lengths:
            print(f"📊 Dataset có {len(context_lengths)} items")
            print(f"📊 Context lengths: min={min(context_lengths)}, max={max(context_lengths)}, avg={sum(context_lengths)/len(context_lengths):.1f}")
            print(f"✨ Sử dụng ALL contexts cho mỗi sample (dynamic)")
        else:
            print(f"⚠️  No context data found")
    else:
        print(f"⚠️  No training data loaded")
    
    print(f"Training trên {len(train_data)} {args.dataset.upper()} samples")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.learning_rate}")
    print(f"Model sẽ lưu tại: {args.save_path}")
    print("=" * 60)
    
    # Tạo model
    model = create_advanced_retriever(
        model_name="microsoft/deberta-v3-base",
        beam_size=2,
        use_focal=True,
        use_early_stop=True,
        max_seq_len=args.max_len,
        gradient_checkpointing=args.gradient_checkpointing
    )
    model.to(device)
    
    # Thiết lập mixed precision
    scaler = None
    if args.mixed_precision and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        print("⚡ Sử dụng mixed precision training")
    
    # Tạo dataset và dataloader
    print("\n📦 Đang tạo dataset...")
    tokenizer = model.tokenizer

    dataset = RetrievalDataset(train_data, tokenizer, max_len=args.max_len)  # 🆕 Bỏ num_contexts
    
    # Pin memory để chuyển GPU nhanh hơn
    pin_memory = torch.cuda.is_available()
    num_workers = 2 if torch.cuda.is_available() else 0

    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    
    # Optimizer với cài đặt tốt hơn cho GPU
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    # Vòng lặp training
    print(f"\n🎯 Bắt đầu training cho {args.epochs} epochs...")
    if torch.cuda.is_available():
        print(f"🔥 GPU Memory trước khi training: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    
    best_avg_f1 = 0.0
    best_avg_em = 0.0
    best_max_f1 = 0.0
    best_max_em = 0.0
    train_metrics = []
    
    for epoch in range(args.epochs):
        print(f"\n📚 Epoch {epoch+1}/{args.epochs}")
        
        # Thông tin GPU memory
        if torch.cuda.is_available():
            print(f"🔥 GPU Memory trước epoch: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        

        max_f1, max_em, avg_f1, avg_em, avg_loss = train_epoch(

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
            'avg_f1': avg_f1,
            'avg_em': avg_em,
            'avg_loss': avg_loss
        }
        train_metrics.append(epoch_metrics)

        print(f"Epoch {epoch+1}: Max F1={max_f1:.4f}, Avg F1={avg_f1:.4f}, Max EM={max_em:.4f}, Avg EM={avg_em:.4f}, Loss={avg_loss:.4f}")

        
        # Thông tin GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated()/1024**2
            gpu_max = torch.cuda.max_memory_allocated()/1024**2
            print(f"🔥 GPU Memory: {gpu_mem:.1f}MB (Max: {gpu_max:.1f}MB)")
            torch.cuda.reset_peak_memory_stats()
        
        # Lưu model nếu F1 cải thiện
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
                    'dataset': args.dataset,
                    'samples': args.samples,
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate
                }
            }, args.save_path)
            print(f"Model mới tốt nhất lưu tại {args.save_path} (F1: {best_f1:.4f})")
    
    # Tóm tắt cuối training
    final_avg_f1 = train_metrics[-1]['avg_f1'] if train_metrics else 0.0
    final_avg_em = train_metrics[-1]['avg_em'] if train_metrics else 0.0
    print(f"\nTraining hoàn thành!")
    print(f"Best Max F1: {best_f1:.4f}, Best Max EM: {best_em:.4f}")
    print(f"Final Avg F1: {final_avg_f1:.4f}, Final Avg EM: {final_avg_em:.4f}")
    print(f"Model đã lưu tại: {args.save_path}")


if __name__ == "__main__":
    main()
