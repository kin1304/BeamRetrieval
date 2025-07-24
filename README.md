# 🔍 DeBERTa v3 Multi-Hop Retriever

Hệ thống retrieval đa bước sử dụng DeBERTa v3 với beam search cho bài toán Question Answering trên dataset HotpotQA.

## 📋 Tổng quan

Model này thực hiện multi-hop reasoning để tìm kiếm các supporting facts từ nhiều documents liên quan đến một câu hỏi. Sử dụng kiến trúc DeBERTa v3 với 183M parameters và beam search algorithm.

### 🎯 Tính năng chính:
- **DeBERTa v3 Base**: 183,834,628 parameters cho hiệu suất cao
- **Beam Search**: Tìm kiếm đa đường với beam_size=2
- **Focal Loss**: Xử lý class imbalance trong training
- **Multi-hop Reasoning**: Kết hợp thông tin từ nhiều documents
- **F1 & Exact Match**: Đánh giá hiệu suất chi tiết

## 🏗️ Kiến trúc Model

```
DeBERTa v3 Encoder
├── Question + Context Encoding
├── Hop 1: First document selection
├── Hop 2+: Subsequent document selection  
├── Beam Search Expansion
└── Supporting Facts Prediction
```

### 📊 Thống kê Model:
- **Model**: microsoft/deberta-v3-base
- **Parameters**: 183,834,628
- **Max Sequence Length**: 128-256 tokens
- **Beam Size**: 2
- **Training Data**: HotpotQA

## 🚀 Cài đặt

### Yêu cầu hệ thống:
- Python 3.8+
- PyTorch 1.9+
- Transformers 4.20+
- 8GB+ RAM

### Cài đặt dependencies:
```bash
# Clone repository
git clone https://github.com/kin1304/BeamRetrieval.git
cd BeamRetrieval

# Tạo virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Cài đặt packages
pip install -r requirements.txt
```

### Dependencies chính:
```
torch>=1.9.0
transformers>=4.20.0
datasets>=2.0.0
tqdm>=4.64.0
tiktoken>=0.4.0
sentencepiece>=0.1.97
```

## 📈 Training

### Quick Start:
```bash
# Training cơ bản với tham số mặc định
python train.py

# Training với cấu hình tùy chỉnh
python train.py --samples 100 --epochs 3 --batch_size 1 --max_len 128
```

### Tham số Training:

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--samples` | 100 | Số lượng training samples |
| `--epochs` | 3 | Số epochs training |
| `--batch_size` | 2 | Batch size (khuyến nghị 1-2 cho CPU) |
| `--learning_rate` | 1e-5 | Learning rate |
| `--max_len` | 128 | Độ dài sequence tối đa |
| `--max_batches` | 10 | Số batches tối đa mỗi epoch |
| `--save_path` | models/retriever_trained.pt | Đường dẫn lưu model |

### Ví dụ Training Commands:

```bash
# Training nhanh cho testing
python train.py --samples 50 --epochs 2 --batch_size 1 --max_batches 5

# Training ổn định
python train.py --samples 100 --epochs 3 --batch_size 1 --max_len 128

# Training đầy đủ (cần nhiều thời gian)
python train.py --samples 500 --epochs 5 --batch_size 2 --max_len 256
```

## 📊 Kết quả Training

### Metrics theo dõi:
- **F1 Score**: Đo lường độ chính xác partial overlap
- **Exact Match (EM)**: Đo lường exact match hoàn toàn
- **Loss**: Cross-entropy hoặc Focal loss

### Kết quả mẫu:
```
📊 Epoch 3 - Max F1: 0.6667, Max EM: 0.0000, Avg Loss: 0.6787
🏆 Best F1: 0.6667, Best EM: 0.0000
💾 Model saved to models/deberta_v3_stable.pt
```

### Giải thích Metrics:
- **F1 = 0.6667**: Model dự đoán đúng 2/3 supporting facts
- **EM = 0.0000**: Chưa có exact match hoàn toàn (bình thường ở early training)
- **Loss giảm dần**: Model đang học hiệu quả

## 🔧 Cấu trúc Code

```
BeamRetrieval/
├── train.py                    # Main training script
├── models/
│   ├── advanced_retriever.py   # Core model architecture
│   └── *.pt                    # Trained model checkpoints
├── utils/
│   └── data_loader.py          # Data loading utilities
├── data/                       # HotpotQA dataset
└── requirements.txt            # Dependencies
```

### Core Components:

#### 1. **Retriever Class** (`models/advanced_retriever.py`):
```python
class Retriever(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base", beam_size=2, ...)
    def forward(self, q_codes, c_codes, sf_idx, hop=0)
    def encode_texts(self, texts, max_length=None)
```

#### 2. **Training Script** (`train.py`):
```python
def calculate_f1_em(predictions, targets)  # Metrics calculation
def train_epoch(model, dataloader, optimizer, device)  # Training loop
class RetrievalDataset(Dataset)  # Data preparation
```

#### 3. **Data Loader** (`utils/data_loader.py`):
```python
def load_hotpot_data(split='train', sample_size=None)  # HotpotQA loading
```

## 🎯 Sử dụng Model

### Load trained model:
```python
from models.advanced_retriever import create_advanced_retriever
import torch

# Load model
model = create_advanced_retriever(
    model_name="microsoft/deberta-v3-base",
    beam_size=2,
    use_focal=True,
    max_seq_len=128
)

# Load trained weights
checkpoint = torch.load('models/deberta_v3_stable.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Inference example:
```python
# Prepare inputs
question = "What is the capital of France?"
contexts = [
    {"title": "France", "text": "France is a country in Europe..."},
    {"title": "Paris", "text": "Paris is the capital city of France..."},
    # ... more contexts
]

# Tokenize and predict
# (Implementation depends on your specific use case)
```

## 🐛 Troubleshooting

### Vấn đề thường gặp:

#### 1. **Memory Issues**:
```bash
# Giảm batch_size
python train.py --batch_size 1

# Giảm sequence length
python train.py --max_len 64

# Giảm số samples
python train.py --samples 50
```

#### 2. **Training quá chậm**:
```bash
# Giới hạn batches per epoch
python train.py --max_batches 5

# Sử dụng ít samples
python train.py --samples 20 --epochs 2
```

#### 3. **Model chỉ predict 1 supporting fact**:
- Đây là hiện tượng bình thường ở early training
- Model học dần từ single-hop sang multi-hop
- Tăng epochs và samples để cải thiện

#### 4. **Resource Tracker Warning**:
```
UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects
```
- Warning này không ảnh hưởng đến training
- Do memory management của PyTorch multiprocessing

## 📈 Performance Tips

### 🚀 Tối ưu Training:
1. **Batch Size**: Sử dụng 1-2 cho CPU, 4-8 cho GPU
2. **Sequence Length**: 128 tokens optimal cho speed/accuracy
3. **Gradient Accumulation**: Tăng effective batch size
4. **Mixed Precision**: Sử dụng fp16 nếu có GPU

### 💾 Memory Optimization:
1. **Workspace cleanup**: Đã xóa files không cần thiết
2. **Model checkpointing**: Chỉ lưu best models
3. **Data batching**: Xử lý small batches

## 📚 Dataset

### HotpotQA:
- **Training**: 90,447 samples
- **Dev**: 7,405 samples  
- **Format**: Multi-hop questions với supporting facts
- **Location**: `data/hotpotqa/`

### Data Structure:
```python
{
    "question": "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
    "contexts": [
        {"title": "Document 1", "text": "Content..."},
        {"title": "Document 2", "text": "Content..."}
    ],
    "supporting_facts": [
        ["Document 1", 0],  # Document title, sentence index
        ["Document 2", 3]
    ]
}
```

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## 📄 License

MIT License - xem file LICENSE để biết chi tiết.

## 🙏 Acknowledgments

- **DeBERTa v3**: Microsoft Research
- **HotpotQA**: Yang et al., 2018
- **Transformers**: Hugging Face
- **PyTorch**: Facebook Research

---

## 📞 Support

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra [Troubleshooting](#-troubleshooting)
2. Tạo issue trên GitHub
3. Đảm bảo đã cài đặt đúng dependencies

**Happy Multi-Hop Reasoning! 🔍✨**