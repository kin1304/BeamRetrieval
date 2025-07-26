#!/usr/bin/env python3
"""
Script Test Retrieval cho Advanced Multi-Hop Retriever
Cách dùng: python test_retrieval.py --model_path models/checkpoint.pt [--dataset test] [--samples N]
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
import json
import random

from models.advanced_retriever import create_advanced_retriever
from utils.data_loader import load_hotpot_data
from train import RetrievalDataset, collate_fn, calculate_f1_em, get_device

def load_model_from_checkpoint(checkpoint_path, device):
    """Load model từ checkpoint file"""
    print(f"Đang load model từ {checkpoint_path}...")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint tại {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Tạo model với config từ checkpoint
    model = create_advanced_retriever(
        model_name=config.get('model_name', 'microsoft/deberta-v3-base'),
        beam_size=config.get('beam_size', 2),
        use_focal=config.get('use_focal', True),
        use_early_stop=True,
        max_seq_len=config.get('max_seq_len', 512),
        gradient_checkpointing=False  # Không cần gradient checkpointing khi test
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config

def evaluate_model(model, dataloader, device, max_batches=None, has_labels=True):
    """Evaluate model trên test set"""
    model.eval()
    all_f1_scores = []
    all_em_scores = []
    detailed_results = []
    
    total_batches = max_batches if max_batches else len(dataloader)
    
    print(f"Bắt đầu evaluation trên {total_batches} batches...")
    if not has_labels:
        print(f"📝 Mode: Prediction only (không có ground truth labels)")
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", total=total_batches)
        
        for batch_idx, batch in enumerate(progress_bar):
            batch_results = []
            
            # Xử lý từng sample trong batch
            for i in range(len(batch['q_codes'])):
                try:
                    # Di chuyển dữ liệu lên device
                    q_codes = [q.to(device, non_blocking=True) for q in batch['q_codes'][i]]
                    p_codes = [p.to(device, non_blocking=True) for p in batch['p_codes'][i]]
                    context_mapping = batch['context_mapping'][i]
                    sf_idx = [s.to(device, non_blocking=True) for s in batch['sf_idx'][i]]
                    hop = batch['hops'][i]
                    
                    # Forward pass
                    outputs = model(q_codes, p_codes, sf_idx, hop, context_mapping=context_mapping)
                    
                    # Lấy predictions
                    predictions = []
                    
                    if 'final_preds' in outputs and outputs['final_preds']:
                        predictions = outputs['final_preds'][0] if len(outputs['final_preds']) > 0 else []
                    elif 'current_preds' in outputs and outputs['current_preds']:
                        predictions = outputs['current_preds'][0] if len(outputs['current_preds']) > 0 else []
                    
                    # Chỉ tính F1/EM nếu có labels
                    f1, em = 0.0, 0.0
                    targets = []
                    
                    if has_labels:
                        targets = sf_idx[0].cpu().tolist()
                        f1, em = calculate_f1_em(predictions, targets)
                        all_f1_scores.append(f1)
                        all_em_scores.append(em)
                    
                    # Lưu kết quả chi tiết
                    result = {
                        'sample_idx': batch_idx * len(batch['q_codes']) + i,
                        'predictions': predictions,
                        'hop': hop
                    }
                    
                    if has_labels:
                        result.update({
                            'targets': targets,
                            'f1': f1,
                            'em': em
                        })
                    
                    batch_results.append(result)
                    
                except Exception as e:
                    print(f"Lỗi tại sample {batch_idx}-{i}: {e}")
                    continue
            
            detailed_results.extend(batch_results)
            
            # Cập nhật progress bar
            if has_labels and all_f1_scores:
                current_avg_f1 = sum(all_f1_scores) / len(all_f1_scores)
                current_avg_em = sum(all_em_scores) / len(all_em_scores)
                progress_bar.set_postfix({
                    'avg_f1': f'{current_avg_f1:.4f}',
                    'avg_em': f'{current_avg_em:.4f}',
                    'samples': len(all_f1_scores)
                })
            else:
                progress_bar.set_postfix({
                    'samples': len(detailed_results)
                })
            
            # Dừng sớm nếu max_batches được chỉ định
            if max_batches and batch_idx >= max_batches - 1:
                break
    
    return all_f1_scores, all_em_scores, detailed_results

def print_evaluation_results(f1_scores, em_scores, detailed_results, has_labels=True):
    """In kết quả evaluation chi tiết"""
    if not detailed_results:
        print("Không có kết quả để hiển thị!")
        return
    
    print(f"\n" + "="*60)
    print(f"📊 KẾT QUẢ EVALUATION")
    print(f"="*60)
    print(f"Tổng số samples: {len(detailed_results)}")
    
    if not has_labels:
        print(f"📝 Mode: Prediction only (test set không có ground truth labels)")
        print(f"💡 Predictions đã được tạo cho {len(detailed_results)} samples")
        
        # Hiển thị một vài ví dụ predictions
        print(f"\n📋 VÍ DỤ PREDICTIONS:")
        for i, result in enumerate(detailed_results[:5]):
            print(f"  {i+1}. Sample {result['sample_idx']}: {result['predictions']}")
        
        return
    
    if not f1_scores or not em_scores:
        print("Không có scores để tính toán!")
        return
    
    # Tính toán metrics tổng quan
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_em = sum(em_scores) / len(em_scores)
    max_f1 = max(f1_scores)
    max_em = max(em_scores)
    
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Average EM Score: {avg_em:.4f}")
    
    # Phân tích phân bố
    f1_perfect = sum(1 for f1 in f1_scores if f1 == 1.0)
    f1_good = sum(1 for f1 in f1_scores if f1 >= 0.5)
    f1_poor = sum(1 for f1 in f1_scores if f1 == 0.0)
    
    em_perfect = sum(1 for em in em_scores if em == 1.0)
    
    print(f"\n📈 PHÂN TÍCH PHÂN BỐ:")
    print(f"F1 = 1.0 (Perfect): {f1_perfect}/{len(f1_scores)} ({f1_perfect/len(f1_scores)*100:.1f}%)")
    print(f"F1 ≥ 0.5 (Good): {f1_good}/{len(f1_scores)} ({f1_good/len(f1_scores)*100:.1f}%)")
    print(f"F1 = 0.0 (Poor): {f1_poor}/{len(f1_scores)} ({f1_poor/len(f1_scores)*100:.1f}%)")
    print(f"EM = 1.0 (Exact): {em_perfect}/{len(em_scores)} ({em_perfect/len(em_scores)*100:.1f}%)")
    
    # Một vài ví dụ tốt nhất và tệ nhất
    sorted_results = sorted(detailed_results, key=lambda x: x.get('f1', 0), reverse=True)

def save_results(results, f1_scores, em_scores, output_path):
    """Lưu kết quả vào file JSON"""
    output_data = {
        'summary': {
            'total_samples': len(f1_scores),
            'avg_f1': sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            'avg_em': sum(em_scores) / len(em_scores) if em_scores else 0.0,
            'max_f1': max(f1_scores) if f1_scores else 0.0,
            'max_em': max(em_scores) if em_scores else 0.0,
        },
        'detailed_results': results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Kết quả đã lưu vào {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Test Advanced Multi-Hop Retriever')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Đường dẫn tới checkpoint model (.pt file)')
    parser.add_argument('--dataset', type=str, default='dev', choices=['train', 'dev', 'test'],
                       help='Dataset để test (mặc định: dev - có labels để tính F1/EM)')
    parser.add_argument('--samples', type=int, default=None,
                       help='Số lượng samples để test (None = toàn bộ dataset)')
    parser.add_argument('--max_batches', type=int, default=None,
                       help='Số batch tối đa để test (None = toàn bộ)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size cho testing')
    parser.add_argument('--output', type=str, default=None,
                       help='File output để lưu kết quả (JSON format)')
    
    args = parser.parse_args()
    
    # Lựa chọn thiết bị
    device = get_device()
    
    # Load model từ checkpoint
    model, config = load_model_from_checkpoint(args.model_path, device)
    
    # Load test data
    dataset_name = args.dataset
    
    if args.dataset == 'test':
        print(f"⚠️  Lưu ý: HotpotQA test set không có ground truth labels")
        print(f"📝 Sẽ tạo predictions mà không tính F1/EM scores")
    
    test_data = load_hotpot_data(dataset_name, sample_size=args.samples)
    
    # Tạo dataset và dataloader
    max_seq_len = config.get('max_seq_len', 512)
    dataset = RetrievalDataset(test_data, model.tokenizer, max_len=max_seq_len, num_contexts=5)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Không shuffle khi test
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        num_workers=0  # Không dùng multiprocessing khi test
    )
    
    # Evaluate model
    has_labels = args.dataset in ['train', 'dev']  # test set không có labels
    f1_scores, em_scores, detailed_results = evaluate_model(
        model, dataloader, device, max_batches=args.max_batches, has_labels=has_labels
    )
    
    # In kết quả
    print_evaluation_results(f1_scores, em_scores, detailed_results, has_labels=has_labels)
    
    # Lưu kết quả nếu có chỉ định output file
    if args.output:
        save_results(detailed_results, f1_scores, em_scores, args.output)
    
    print(f"\n🎉 Evaluation hoàn thành!")

if __name__ == "__main__":
    main()
