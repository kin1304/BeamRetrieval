#!/usr/bin/env python3
"""
GPU Test Script - Check GPU/CUDA availability and performance
"""

import torch
import time
from transformers import AutoTokenizer, AutoModel

def test_gpu_setup():
    """Test GPU setup and availability"""
    print("🧪 GPU/CUDA Test")
    print("=" * 50)
    
    # Basic CUDA info
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  Device {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1024**3:.1f}GB")
            print(f"    Compute capability: {props.major}.{props.minor}")
    
    # MPS (Apple Silicon) info
    print(f"\nMPS available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print("Apple Silicon GPU detected")
    
    # Choose device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name()
        print(f"\n✅ Using CUDA GPU: {device_name}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("\n✅ Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("\n💻 Using CPU")
    
    return device

def test_model_loading(device):
    """Test loading model on GPU"""
    print(f"\n🧠 Testing model loading on {device}")
    
    try:
        # Load tokenizer and model
        model_name = "microsoft/deberta-v3-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Move to device
        start_time = time.time()
        model = model.to(device)
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded on {device} in {load_time:.2f}s")
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        if torch.cuda.is_available():
            print(f"🔥 GPU Memory used: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def test_inference_speed(model, tokenizer, device):
    """Test inference speed on GPU vs CPU"""
    if model is None:
        return
    
    print(f"\n⚡ Testing inference speed on {device}")
    
    # Sample text
    texts = [
        "What is the capital of France? Paris is the capital and most populous city of France.",
        "Who wrote Romeo and Juliet? William Shakespeare wrote the famous play Romeo and Juliet.",
        "What is machine learning? Machine learning is a subset of artificial intelligence."
    ] * 10  # Repeat for better timing
    
    model.eval()
    
    # Warmup
    sample_input = tokenizer(texts[0], return_tensors='pt').to(device)
    with torch.no_grad():
        _ = model(**sample_input)
    
    # Actual timing
    start_time = time.time()
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding=True).to(device)
            outputs = model(**inputs)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(texts)
    
    print(f"📈 Total time: {total_time:.3f}s")
    print(f"📈 Average time per sample: {avg_time*1000:.2f}ms")
    print(f"📈 Throughput: {len(texts)/total_time:.1f} samples/sec")
    
    if torch.cuda.is_available():
        print(f"🔥 Peak GPU Memory: {torch.cuda.max_memory_allocated()/1024**2:.1f}MB")

def main():
    """Main test function"""
    # Test GPU setup
    device = test_gpu_setup()
    
    # Test model loading
    model, tokenizer = test_model_loading(device)
    
    # Test inference speed
    test_inference_speed(model, tokenizer, device)
    
    # Training recommendations
    print(f"\n💡 Training Recommendations:")
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU Memory: {gpu_memory:.1f}GB")
        
        if gpu_memory >= 8:
            print("   ✅ Use batch_size=8-16 for training")
            print("   ✅ Enable mixed precision: --mixed_precision")
            print("   ✅ Use gradient_accumulation=2-4")
        elif gpu_memory >= 4:
            print("   ⚠️  Use batch_size=4-8 for training")
            print("   ✅ Enable mixed precision: --mixed_precision")
            print("   ✅ Use gradient_accumulation=4-8")
        else:
            print("   ⚠️  Use batch_size=2-4 for training")
            print("   ✅ Enable mixed precision: --mixed_precision")
            print("   ⚠️  Use gradient_accumulation=8-16")
            
        print(f"\n🚀 Run training with GPU:")
        print(f"   python train.py --gpu --mixed_precision --batch_size=8")
        
    elif torch.backends.mps.is_available():
        print("   ✅ Use batch_size=4-8 for training")
        print("   ⚠️  Mixed precision not available on MPS")
        print(f"\n🍎 Run training with MPS:")
        print(f"   python train.py --batch_size=4")
        
    else:
        print("   ⚠️  CPU only - use smaller batch_size=2")
        print("   ⚠️  Training will be slow")
        print(f"\n💻 Run training with CPU:")
        print(f"   python train.py --batch_size=2")

if __name__ == "__main__":
    main()
