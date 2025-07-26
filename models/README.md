# 🧠 Thư mục Models - BeamRetrieval

## 📋 Tổng quan

Thư mục này chứa các model chính của hệ thống BeamRetrieval, bao gồm implementation của Advanced Multi-Hop Retriever với khả năng beam search và tối ưu hóa bộ nhớ.

## 📁 Cấu trúc Files

### `advanced_retriever.py`
**Model chính** - Implementation của Advanced Multi-Hop Retriever

#### 🎯 Các thành phần chính:
- **`FocalLoss`**: Triển khai Focal Loss để xử lý class imbalance
- **`Retriever`**: Model chính với multi-hop reasoning và beam search
- **`create_advanced_retriever()`**: Factory function để tạo model

#### 🚀 Tính năng nổi bật:
- **Multi-hop reasoning**: Hỗ trợ reasoning qua nhiều hop
- **Beam search**: Theo dõi nhiều hypothesis với beam_size=2
- **Paragraph-level processing**: Xử lý đoạn văn riêng biệt thay vì full context
- **Progressive concatenation**: Kết hợp thông tin từ các hop trước đó
- **Dual-level output**: Trả về cả paragraph-level và context-level predictions

#### ⚙️ Cấu hình Model:
```python
model = create_advanced_retriever(
    model_name="microsoft/deberta-v3-base",
    beam_size=2,
    use_focal=True,
    use_early_stop=True,
    max_seq_len=512,
    gradient_checkpointing=False  # Incompatible với multi-hop
)
```

#### 💾 Model Files (*.pt)
Các file model đã trained:
- `deberta_v3_full_dataset.pt` - Model train trên full dataset
- `deberta_v3_full_trained.pt` - Model hoàn chỉnh đã train
- `deberta_v3_optimized.pt` - Model tối ưu hóa
- `deberta_v3_paper_dev.pt` - Model train trên dev set (paper config)
- `deberta_v3_paper_full.pt` - Model train với config paper đầy đủ
- `deberta_v3_stable.pt` - Phiên bản ổn định
- `deberta_v3_trained.pt` - Model đã train cơ bản
- `retriever_trained.pt` - Model retriever đã train

## 🔧 API chính

### Khởi tạo Model
```python
from models.advanced_retriever import create_advanced_retriever

model = create_advanced_retriever(
    model_name="microsoft/deberta-v3-base",
    beam_size=2,
    use_focal=True,
    max_seq_len=512
)
```

### Forward Pass
```python
outputs = model(
    q_codes=question_tokens,      # Token câu hỏi sạch
    p_codes=paragraph_sequences,  # Chuỗi đoạn văn đã tokenized
    sf_idx=supporting_facts,      # Supporting fact indices
    hop=2,                       # Số hops
    context_mapping=mapping      # Ánh xạ paragraph→context
)
```

### Output Format
```python
{
    'current_preds': [[context_indices]],     # Context-level predictions
    'final_preds': [[context_indices]],       # Alias cho backward compatibility
    'paragraph_preds': [[paragraph_indices]], # Paragraph-level predictions (chi tiết)
    'loss': tensor(loss_value)                # Training loss
}
```

## 🏗️ Kiến trúc Algorithm

### Hop 1: Independent Paragraph Scoring
- Chấm điểm từng đoạn văn độc lập với câu hỏi
- Format: `[CLS] + Question + Paragraph + [SEP]`
- Chọn top `beam_size` đoạn văn có điểm cao nhất

### Hop 2+: Multi-hop Combination
- Mở rộng mỗi beam với tất cả đoạn văn chưa sử dụng
- Progressive concatenation: `[CLS] + Q + P1 + P2 + ... + [SEP]`
- Beam pruning: Chỉ giữ top candidates

### Output Generation
- Chuyển đổi paragraph predictions về context predictions
- Dual-level output cho flexibility

## 🔥 Tối ưu hóa

### Memory Optimization
- **Gradient checkpointing**: Disabled (incompatible với multi-hop)
- **Mixed precision**: Hỗ trợ automatic mixed precision
- **Paragraph splitting**: Pre-processing trong data loader
- **Device management**: Consistent GPU usage

### Performance Features
- **Vectorized operations**: Batch processing hiệu quả
- **Smart truncation**: Ưu tiên question và special tokens
- **Progressive reasoning**: Tích lũy context qua các hop

## 📊 Metrics và Evaluation

Model được đánh giá bằng:
- **F1 Score**: Đo overlap một phần giữa predicted và target
- **Exact Match (EM)**: Perfect match với supporting facts
- **Loss Tracking**: CrossEntropy + Focal Loss (optional)

## 🚀 Sử dụng

### Training
```bash
python train.py --dataset train --epochs 16 --batch_size 1 --learning_rate 2e-5
```

### Inference
```python
model.eval()
with torch.no_grad():
    outputs = model(q_codes, p_codes, sf_idx, hop=2)
    predictions = outputs['current_preds'][0]
```

## 📝 Lưu ý quan trọng

1. **Gradient Checkpointing**: Không tương thích với multi-hop reasoning do reuse computational graph
2. **Memory Management**: Sử dụng mixed precision hoặc smaller batch size thay vì gradient checkpointing  
3. **Device Consistency**: Đảm bảo tất cả tensors cùng device
4. **Paragraph Processing**: Paragraph splitting được thực hiện trong data loader để hiệu quả

## 🔗 Dependencies

- PyTorch ≥ 1.9.0
- Transformers ≥ 4.20.0
- DeBERTa-v3-base model từ Hugging Face

## 📈 Performance

Model đạt được:
- **F1 Score**: ~66.7% trên dev set
- **Exact Match**: ~40-50% tùy thuộc vào dataset
- **Training Speed**: ~2-3 samples/second trên Apple Silicon MPS
