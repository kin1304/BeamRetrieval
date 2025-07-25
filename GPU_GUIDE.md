# 🚀 Hướng dẫn chạy trên GPU

## 🍎 Apple Silicon (MPS)

Nếu bạn có Mac với chip Apple Silicon (M1/M2/M3):

```bash
# Training tối ưu cho Apple Silicon
python run_mps_training.py

# Hoặc manual với batch size lớn hơn
python train.py --batch_size=6 --samples=1000 --epochs=3
```

## 🔥 NVIDIA GPU (CUDA)

Nếu bạn có GPU NVIDIA với CUDA:

```bash
# Training với GPU và mixed precision
python train.py --gpu --mixed_precision --batch_size=8 --samples=5000 --epochs=3

# GPU lớn (>8GB VRAM)
python train.py --gpu --mixed_precision --batch_size=16 --gradient_accumulation=2

# GPU nhỏ (<4GB VRAM) 
python train.py --gpu --mixed_precision --batch_size=4 --gradient_accumulation=8
```

## ⚙️ Tối ưu hóa Memory

### Giảm memory sử dụng:
```bash
# Giảm max_len
python train.py --max_len=64 --batch_size=8

# Gradient accumulation thay vì batch size lớn
python train.py --batch_size=2 --gradient_accumulation=8

# Giảm số contexts
python train.py --batch_size=4  # Default 5 contexts
```

### Tăng hiệu suất:
```bash
# Mixed precision (chỉ CUDA)
python train.py --mixed_precision --batch_size=8

# Gradient checkpointing (tiết kiệm memory)
python train.py --batch_size=4  # Tự động bật trong model
```

## 🧠 Mixed Precision là gì?

**Mixed Precision** là kỹ thuật sử dụng cả **FP16** (16-bit) và **FP32** (32-bit) floating point trong cùng một model:

### 📊 So sánh độ chính xác:
- **FP32** (Single): 32-bit, độ chính xác cao, memory nhiều
- **FP16** (Half): 16-bit, độ chính xác thấp hơn, memory ít hơn 50%
- **Mixed**: Dùng FP16 cho forward pass, FP32 cho gradients

### ✅ Ưu điểm:
- **Tăng tốc độ**: 1.5-2x nhanh hơn trên GPU Tensor Core
- **Tiết kiệm memory**: Giảm 50% VRAM usage
- **Batch size lớn hơn**: Có thể train với batch size gấp đôi
- **Độ chính xác**: Vẫn giữ được quality như FP32

### ⚙️ Cách hoạt động:
```python
# Forward pass: FP16 (nhanh, ít memory)
with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

# Backward pass: FP32 (chính xác)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 🎯 Khi nào sử dụng:
- ✅ **NVIDIA GPU** với Tensor Cores (RTX 20xx, 30xx, 40xx, V100, A100)
- ✅ **Large models** (>100M parameters)
- ✅ **Memory limited** (GPU <8GB)
- ❌ **Apple Silicon** (chưa hỗ trợ)
- ❌ **CPU training**

### 📈 Performance boost:
```bash
# Không mixed precision
python train.py --batch_size=4    # ~6GB VRAM, 45s/epoch

# Với mixed precision  
python train.py --mixed_precision --batch_size=8  # ~6GB VRAM, 25s/epoch
```

## 📊 Monitoring GPU

```bash
# Theo dõi GPU usage (NVIDIA)
nvidia-smi -l 1

# Theo dõi Apple Silicon
sudo powermetrics -n 1 --samplers gpu_power
```

## 🔧 Troubleshooting

### CUDA Out of Memory:
```bash
# Giảm batch size
python train.py --batch_size=2 --gradient_accumulation=8

# Giảm sequence length
python train.py --max_len=64

# Dọn cache
python -c "import torch; torch.cuda.empty_cache()"
```

### MPS Issues:
```bash
# Fallback to CPU nếu MPS có vấn đề
export PYTORCH_ENABLE_MPS_FALLBACK=1
python train.py --batch_size=4
```

## 🏆 Recommended Settings

### Apple Silicon M1/M2/M3:
```bash
python train.py --batch_size=6 --samples=2000 --epochs=3 --max_len=128
```

### RTX 3090/4090 (24GB):
```bash
python train.py --gpu --mixed_precision --batch_size=16 --samples=10000 --epochs=5
```

### RTX 3080/4080 (12GB):
```bash
python train.py --gpu --mixed_precision --batch_size=8 --samples=5000 --epochs=3
```

### RTX 3070/4070 (8GB):
```bash
python train.py --gpu --mixed_precision --batch_size=6 --gradient_accumulation=2
```

### GTX 1660/RTX 3060 (6GB):
```bash
python train.py --gpu --mixed_precision --batch_size=4 --gradient_accumulation=4
```

## 🎯 Format Input mới

Model hiện sử dụng format: `[CLS] + Question + Context + [SEP]`

- Không còn [SEP] giữa Question và Context
- Đơn giản và hiệu quả hơn
- Tương thích với BERT standard practices

## ✅ Verification

Sau khi training, kiểm tra:

```bash
# Test model
python test_format.py

# Kiểm tra saved model
ls -la models/retriever_trained.pt
```
