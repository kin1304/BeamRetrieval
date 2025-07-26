# 🧠 THUẬT TOÁN CHI TIẾT - LỚP RETRIEVER

## 🎯 **TỔNG QUAN**

Lớp `Retriever` thực hiện **multi-hop reasoning** với **paragraph-level processing** để tìm supporting facts cho câu hỏi đa hop. Thuật toán sử dụng **beam search** để track multiple hypotheses và **progressive concatenation** để build multi-hop reasoning.

---

## 🔄 **FLOW CHÍNH - FORWARD METHOD**

### **INPUT:**
- `q_codes`: List chứa clean question tokens `[question_tokens]`  
- `p_codes`: List paragraph sequences `[CLS] + Q + P + [SEP]` (đã được pre-tokenized)
- `sf_idx`: Supporting fact indices (context-level)
- `hop`: Số hops cho inference mode
- `context_mapping`: Mapping từ paragraph index → original context index

### **OUTPUT:**
- `current_preds`: Context-level predictions (main output)
- `final_preds`: Alias cho backward compatibility  
- `paragraph_preds`: Paragraph-level predictions (detailed)
- `loss`: Training loss

---

## 📋 **BƯỚC 1: INITIALIZATION**

```python
# 1.1 Thiết lập device và các hàm loss
device = q_codes[0].device
total_loss = torch.tensor(0.0, device=device, requires_grad=True)
loss_function = nn.CrossEntropyLoss()
focal_loss_function = FocalLoss() if self.use_focal else None

# 1.2 Trích xuất đầu vào (🚀 ĐÃ TỐI ƯU - không cần trích xuất context!)
question_tokens = q_codes[0]  # Token câu hỏi đã làm sạch
all_paragraph_sequences = p_codes  # Các đoạn văn đã được tokenize trước!
context_to_paragraph_mapping = context_mapping or list(range(len(p_codes)))

# 1.3 Xác định số hop
if self.training:
    sf_idx = sf_idx[0]
    hops = len(sf_idx)  # Số lượng supporting facts
else:
    hops = hop if hop > 0 else 2  # Mặc định 2 hop cho inference
```

**Điểm chính:**
- 🚀 **TỐI ƯU HÓA**: Không còn trích xuất context! Sử dụng các đoạn văn đã tách sẵn trực tiếp
- **Hop động**: Chế độ training dựa trên số supporting facts, chế độ inference dùng tham số

---

## 🥇 **BƯỚC 2: HOP 1 - INDEPENDENT PARAGRAPH SCORING**

### **2.1 Chuẩn bị Tensor**
```python
max_len = max(len(seq) for seq in all_paragraph_sequences)
num_paragraphs = len(all_paragraph_sequences)

# Tạo tensor theo batch cho tất cả đoạn văn
hop1_qp_ids = torch.zeros([num_paragraphs, max_len], device=device, dtype=torch.long)
hop1_qp_attention_mask = torch.zeros([num_paragraphs, max_len], device=device, dtype=torch.long)

# Điền tensor với các chuỗi đoạn văn
for i, paragraph_seq in enumerate(all_paragraph_sequences):
    seq_len = len(paragraph_seq)
    hop1_qp_ids[i, :seq_len] = paragraph_seq
    hop1_qp_attention_mask[i, :seq_len] = (paragraph_seq != pad_token_id).long()
```

### **2.2 Tạo nhãn (Chỉ khi Training)**
```python
if self.training:
    hop1_label = torch.zeros([num_paragraphs], dtype=torch.long, device=device)
    
    for i, paragraph_seq in enumerate(all_paragraph_sequences):
        original_ctx_idx = context_to_paragraph_mapping[i]
        if original_ctx_idx in sf_idx:  # Context hỗ trợ
            hop1_label[i] = 1  # Nhãn tích cực
```

### **2.3 Forward Pass**
```python
# Encoder forward: [num_paragraphs, seq_len] → [num_paragraphs, hidden_size]
hop1_encoder_outputs = self.encoder(hop1_qp_ids, hop1_qp_attention_mask)[0][:, 0, :]

# Phân loại: [num_paragraphs, hidden_size] → [num_paragraphs, 2]
hop1_projection = self.hop_classifier_layer(hop1_encoder_outputs)

# Tính toán loss
if self.training:
    total_loss += CrossEntropyLoss(hop1_projection, hop1_label)
```

### **2.4 Chọn Beam**
```python
# Chọn top beam_size đoạn văn dựa trên điểm số lớp tích cực
_, hop1_pred_paragraphs = hop1_projection[:, 1].topk(self.beam_size, dim=-1)

# Khởi tạo theo dõi beam
current_preds = [[idx.item()] for idx in hop1_pred_paragraphs]  # Mỗi beam = [para_idx]

# Trích xuất token đoạn văn đã chọn cho multi-hop
selected_paragraph_tokens = []
for pred_idx in hop1_pred_paragraphs:
    selected_seq = all_paragraph_sequences[pred_idx.item()]
    paragraph_tokens = _extract_paragraph_tokens(selected_seq, question_tokens)
    selected_paragraph_tokens.append([paragraph_tokens])  # Mỗi beam = [para_tokens]
```

**Thuật toán chính:**
- **Chấm điểm độc lập**: Mỗi đoạn văn được chấm điểm độc lập với câu hỏi
- **Chọn TopK**: Chọn `beam_size=2` đoạn văn có điểm cao nhất
- **Trích xuất token**: Trích xuất token đoạn văn để chuẩn bị cho multi-hop

---

## 🔗 **BƯỚC 3: HOP 2+ - MULTI-HOP COMBINATION**

### **3.1 Mở rộng Beam**
```python
for hop_idx in range(1, hops):  # Bỏ qua hop 0 (đã hoàn thành)
    next_sequences = []      # Chuỗi ứng viên cho hop này
    next_labels = []         # Nhãn training
    next_pred_mapping = []   # Ánh xạ ngược về chỉ số đoạn văn
    
    # Với mỗi beam từ hop trước
    for beam_idx in range(self.beam_size):
        beam_selected_paragraphs = selected_paragraph_tokens[beam_idx]  # Lựa chọn trước đó
        beam_used_indices = set(current_preds[beam_idx])                # Chỉ số đoạn văn đã dùng
        
        # Thử mỗi đoạn văn chưa sử dụng làm ứng viên tiếp theo
        for para_idx, paragraph_seq in enumerate(all_paragraph_sequences):
            if para_idx in beam_used_indices:
                continue  # Bỏ qua các đoạn văn đã chọn
```

### **3.2 Tạo chuỗi Multi-hop**
```python
# Trích xuất token đoạn văn mới
new_paragraph_tokens = _extract_paragraph_tokens(paragraph_seq, question_tokens)

# Tạo kết nối tiến bộ: [CLS] + Q + P1 + P2 + ... + Pnew + [SEP]
multi_hop_seq = _create_multi_hop_sequence(
    question_tokens,           # Token câu hỏi sạch
    beam_selected_paragraphs,  # Lựa chọn hop trước  
    new_paragraph_tokens       # Đoạn văn ứng viên mới
)

next_sequences.append(multi_hop_seq)
next_pred_mapping.append(current_preds[beam_idx] + [para_idx])  # Theo dõi lựa chọn
```

**Định dạng chuỗi Multi-hop:**
```
Hop 1: [CLS] + Q + P1 + [SEP]
Hop 2: [CLS] + Q + P1 + P2 + [SEP]  
Hop 3: [CLS] + Q + P1 + P2 + P3 + [SEP]
```

### **3.3 Tạo nhãn (Training)**
```python
if self.training:
    new_pred_set = set(current_preds[beam_idx] + [para_idx])
    target_contexts = set()
    
    # Ánh xạ chỉ số đoạn văn về chỉ số context
    for p_idx in new_pred_set:
        target_contexts.add(context_to_paragraph_mapping[p_idx])
    
    # Kiểm tra xem tổ hợp có khớp với supporting facts mục tiêu
    if target_contexts == set(sf_idx[:hop_idx+1]):
        next_labels.append(1)  # Tổ hợp đúng
    else:
        next_labels.append(0)  # Tổ hợp sai
```

### **3.4 Forward Pass**
```python
# Chuẩn bị tensor
max_len = max(len(seq) for seq in next_sequences)
num_candidates = len(next_sequences)

hop_qp_ids = torch.zeros([num_candidates, max_len], device=device, dtype=torch.long)
hop_qp_attention_mask = torch.zeros([num_candidates, max_len], device=device, dtype=torch.long)

# Điền tensor
for i, seq in enumerate(next_sequences):
    seq_len = len(seq)
    hop_qp_ids[i, :seq_len] = seq
    hop_qp_attention_mask[i, :seq_len] = (seq != pad_token_id).long()

# Forward pass
hop_encoder_outputs = self.encoder(hop_qp_ids, hop_qp_attention_mask)[0][:, 0, :]
hop_projection = self.hop_n_classifier_layer(hop_encoder_outputs)

# Tính toán loss
if self.training:
    if self.use_focal:
        total_loss += FocalLoss(hop_projection, hop_label)
    else:
        total_loss += CrossEntropyLoss(hop_projection, hop_label)
```

### **3.5 Cập nhật Beam**
```python
# Chọn top beam_size ứng viên
_, hop_pred_indices = hop_projection[:, 1].topk(self.beam_size, dim=-1)

# Cập nhật theo dõi beam
new_current_preds = []
new_selected_paragraph_tokens = []

for pred_idx in hop_pred_indices:
    selected_prediction = next_pred_mapping[pred_idx.item()]
    new_current_preds.append(selected_prediction)
    
    # Xây dựng lại danh sách token đoạn văn cho beam này
    beam_paragraph_tokens = []
    for para_idx in selected_prediction:
        para_seq = all_paragraph_sequences[para_idx]
        para_tokens = _extract_paragraph_tokens(para_seq, question_tokens)
        beam_paragraph_tokens.append(para_tokens)
    
    new_selected_paragraph_tokens.append(beam_paragraph_tokens)

# Cập nhật cho vòng lặp tiếp theo
current_preds = new_current_preds
selected_paragraph_tokens = new_selected_paragraph_tokens
```

**Thuật toán chính:**
- **Mở rộng tổ hợp**: Mỗi beam thử với mọi đoạn văn chưa sử dụng
- **Kết nối tiến bộ**: Xây dựng chuỗi multi-hop từng bước
- **Beam pruning**: Chỉ giữ top `beam_size` ứng viên mỗi hop

---

## 📤 **BƯỚC 4: SINH ĐẦU RA**

### **4.1 Chuyển đổi Paragraph → Dự đoán Context**
```python
final_context_preds = []
for beam_paragraphs in current_preds:
    context_indices = []
    for para_idx in beam_paragraphs:
        ctx_idx = context_to_paragraph_mapping[para_idx]
        if ctx_idx not in context_indices:  # Tránh trùng lặp
            context_indices.append(ctx_idx)
    final_context_preds.append(context_indices)
```

### **4.2 Trả về Kết quả**
```python
return {
    'current_preds': final_context_preds,    # Đầu ra chính (cấp context)
    'final_preds': final_context_preds,      # Tương thích ngược
    'paragraph_preds': current_preds,        # Chi tiết (cấp đoạn văn)
    'loss': total_loss                       # Loss training
}
```

---

## 🔧 **CÁC PHƯƠNG THỨC HỖ TRỢ**

### **_extract_paragraph_tokens()**
```python
def _extract_paragraph_tokens(sequence, question_tokens):
    """Trích xuất token đoạn văn từ chuỗi [CLS] + Q + P + [SEP]"""
    # Định dạng chuỗi: [CLS] + Q + P + [SEP]
    question_start = 1  # Bỏ qua [CLS]
    question_end = question_start + len(question_tokens)
    paragraph_start = question_end
    paragraph_end = len(sequence) - 1  # Bỏ qua [SEP] cuối
    
    if paragraph_start < paragraph_end:
        return sequence[paragraph_start:paragraph_end]
    else:
        return torch.tensor([])  # Rỗng nếu không có nội dung đoạn văn
```

### **_create_multi_hop_sequence()**
```python
def _create_multi_hop_sequence(question_tokens, selected_paragraphs, new_paragraph):
    """Tạo chuỗi multi-hop tiến bộ"""
    # Xây dựng chuỗi: [CLS] + Q + P1 + P2 + ... + Pnew + [SEP]
    sequence_parts = [
        torch.tensor([cls_token_id]),
        question_tokens
    ]
    
    # Thêm các đoạn văn đã chọn từ hop trước
    for paragraph_tokens in selected_paragraphs:
        if len(paragraph_tokens) > 0:
            sequence_parts.append(paragraph_tokens)
    
    # Thêm đoạn văn mới
    if len(new_paragraph) > 0:
        sequence_parts.append(new_paragraph)
    
    # Thêm [SEP] cuối
    sequence_parts.append(torch.tensor([sep_token_id]))
    
    # Kết hợp và xử lý cắt ngắn
    combined_sequence = torch.cat(sequence_parts)
    if len(combined_sequence) > max_seq_len:
        # Cắt ngắn thông minh: giữ [CLS] + Q + [SEP], cắt ngắn đoạn văn
        combined_sequence = combined_sequence[:max_seq_len-1]
        combined_sequence = torch.cat([combined_sequence, torch.tensor([sep_token_id])])
    
    return combined_sequence
```

---

## 🎯 **ĐẶC ĐIỂM THUẬT TOÁN**

### **1. Lý luận Multi-hop Tiến bộ**
- **Hop 1**: Chấm điểm đoạn văn độc lập `[CLS] + Q + P + [SEP]`
- **Hop 2**: Chấm điểm tổ hợp `[CLS] + Q + P1 + P2 + [SEP]`  
- **Hop 3**: Lý luận mở rộng `[CLS] + Q + P1 + P2 + P3 + [SEP]`

### **2. Theo dõi Beam Search**
- **Nhiều giả thuyết**: Duy trì `beam_size=2` ứng viên
- **Ngăn trùng lặp**: Không chọn lại đoạn văn đã sử dụng
- **Chọn tối ưu**: Chọn TopK tại mỗi hop

### **3. Đầu ra Hai cấp**
- **Cấp Context**: `current_preds` cho các tác vụ downstream
- **Cấp Đoạn văn**: `paragraph_preds` cho phân tích chi tiết

### **4. Chiến lược Training**
- **Hop 1**: CrossEntropy loss cho chấm điểm độc lập
- **Hop 2+**: Focal Loss cho class imbalance (tùy chọn)
- **Nhãn tiến bộ**: Khớp chính xác với supporting facts mục tiêu

### **5. Tối ưu hóa Bộ nhớ**
- **Phép toán véc tơ hóa**: Xử lý batch cho hiệu quả
- **Cắt ngắn thông minh**: Ưu tiên câu hỏi và token đặc biệt
- **Quản lý thiết bị**: Sử dụng thiết bị nhất quán

---

## 📊 **PHÂN TÍCH ĐỘ PHỨC TẠP**

### **Độ phức tạp Thời gian**
- **Hop 1**: O(P × L) với P=đoạn văn, L=độ dài chuỗi
- **Hop 2+**: O(B × P × L) với B=beam_size  
- **Tổng cộng**: O(H × B × P × L) với H=số hop

### **Độ phức tạp Không gian**
- **Lưu trữ đoạn văn**: O(P × L)
- **Theo dõi beam**: O(H × B × P)
- **Trạng thái encoder**: O(P × D) với D=hidden_size

---

## 🔍 **VÍ DỤ THỰC HIỆN**

### **Đầu vào:**
- Câu hỏi: "Thủ đô của Pháp là gì?"
- Đoạn văn: [P1: "Thông tin Pháp...", P2: "Thông tin Paris...", P3: "Thông tin Đức..."]
- Mục tiêu: [0, 1] (context 0 và 1 là hỗ trợ)

### **Hop 1:**
- Chấm điểm tất cả: P1=0.8, P2=0.9, P3=0.3
- Chọn top-2: [P2, P1] 
- Beam: [[P2], [P1]]

### **Hop 2:**
- Beam 1: Thử P2+P1, P2+P3 → Điểm: 0.95, 0.4
- Beam 2: Thử P1+P2, P1+P3 → Điểm: 0.95, 0.2  
- Chọn top-2: [P2+P1, P1+P2] (cùng tổ hợp)
- Cuối: [[P1, P2], [P1, P2]]

### **Đầu ra:**
- Dự đoán context: [[1, 0], [1, 0]]
- Dự đoán đoạn văn: [[2, 1], [1, 2]]

Thuật toán này đảm bảo lý luận multi-hop chính xác với khả năng xử lý các context phức tạp! 🚀
