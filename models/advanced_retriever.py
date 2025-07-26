#!/usr/bin/env python3
"""
Model Advanced Multi-Hop Retriever với Focal Loss
Dựa trên kiến trúc bạn cung cấp cho hệ thống BeamRetrieval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from typing import List, Dict, Any, Optional, Tuple
import logging
from transformers import DebertaV2Tokenizer, AutoModel, AutoConfig

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Triển khai Focal Loss để xử lý class imbalance
    """
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Tính cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Tính weight (alpha)
        p_t = torch.exp(-ce_loss)
        alpha_t = self.alpha * (1 - p_t)

        # Tính Focal Loss
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss

        # Chọn reduction method
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError("Unsupported reduction mode. Use 'mean', 'sum', or 'none'.")

class Retriever(nn.Module):
    """
    Multi-hop Retriever với beam search và focal loss
    Dựa trên kiến trúc bạn chỉ định
    """
    
    def __init__(self,
                 config=None,
                 model_name="microsoft/deberta-v3-small",
                 encoder_class=None,
                 max_seq_len=512,
                 mean_passage_len=70,
                 beam_size=2,  # Khớp với yêu cầu beam_width=2 của bạn
                 gradient_checkpointing=False,
                 use_focal=True,  # Bật focal loss mặc định
                 use_early_stop=True,
                 ):
        super().__init__()
        
        # Tải config nếu không được cung cấp
        if config is None:
            config = AutoConfig.from_pretrained(model_name)
        self.config = config
        
        # Tải encoder
        if encoder_class is None:
            from transformers import AutoModel
            encoder_class = AutoModel
        
        self.encoder = encoder_class.from_pretrained(model_name, config=config)
        
        # Lưu trữ parameters
        self.max_seq_len = max_seq_len
        self.mean_passage_len = mean_passage_len
        self.beam_size = beam_size
        # Gradient checkpointing cơ bản không tương thích với multi-hop reasoning
        # do tái sử dụng computational graph giữa các hop gây lỗi "backward second time"
        # Thay thế tối ưu bộ nhớ: sử dụng batch size nhỏ hơn, model parallelism, hoặc mixed precision
        self.gradient_checkpointing = False
        self.use_focal = use_focal
        self.use_early_stop = use_early_stop
        self.use_label_order = False  # Thêm thuộc tính này
        
        # Các lớp phân loại cho hop khác nhau
        self.hop_classifier_layer = nn.Linear(config.hidden_size, 2)
        self.hop_n_classifier_layer = nn.Linear(config.hidden_size, 2)
        
        # Lưu ý: Gradient checkpointing bị vô hiệu hóa để tương thích multi-hop
        # Các thay thế tối ưu bộ nhớ: mixed precision, model parallelism, batch sizes nhỏ hơn
        
        # Khởi tạo tokenizer (ngăn chặn cảnh báo fast tokenizer cho DeBERTa)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*sentencepiece tokenizer.*")
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        
        logger.info(f"🧠 Retriever khởi tạo với {self.count_parameters():,} parameters")
        logger.info(f"📊 Beam size: {beam_size}, Max seq len: {max_seq_len}")
        logger.info(f"🎯 Sử dụng focal loss: {use_focal}, Early stop: {use_early_stop}")
    
    def count_parameters(self):
        """Đếm các parameters có thể train"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def prepare_question_tokens(self, question_text: str):
        """
        Phương thức helper để chuẩn bị token câu hỏi sạch từ raw text
        
        Args:
            question_text: Raw text câu hỏi
            
        Returns:
            torch.Tensor: Token câu hỏi sạch (không có [CLS], [SEP], padding)
        """
        # Tokenize câu hỏi không có special tokens
        question_tokens = self.tokenizer(
            question_text,
            add_special_tokens=False,
            return_tensors='pt'
        )['input_ids'].squeeze(0)
        
        return question_tokens
    
    # 🗑️ DEPRECATED: Các phương thức bên dưới không còn được sử dụng sau khi tối ưu hóa
    # Chia paragraph bây giờ được thực hiện trong data loader (train.py) để hiệu quả
    
    def _split_context_to_paragraphs(self, context_text: str, max_paragraph_len: int = 200):
        """
        ⚠️ DEPRECATED: Chia context text thành paragraphs để xử lý tốt hơn
        
        🚀 TỐI ƯU HÓA: Điều này bây giờ được thực hiện trong RetrievalDataset._split_context_to_paragraphs()
        để tránh luồng tokenize → decode → split → tokenize kém hiệu quả.
        
        Args:
            context_text: Full context text
            max_paragraph_len: Độ dài tối đa mỗi paragraph
            
        Returns:
            List các paragraph strings
        """
        # Chia theo các dấu phân cách paragraph phổ biến
        paragraphs = []
        
        # Try splitting by double newlines first
        parts = context_text.split('\n\n')
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # If part is too long, split by sentences
            if len(part) > max_paragraph_len:
                sentences = part.split('. ')
                current_paragraph = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    # Add period back if not present
                    if not sentence.endswith('.') and not sentence.endswith('!') and not sentence.endswith('?'):
                        sentence += '.'
                    
                    # Check if adding this sentence exceeds limit
                    if len(current_paragraph + sentence) > max_paragraph_len and current_paragraph:
                        paragraphs.append(current_paragraph.strip())
                        current_paragraph = sentence + " "
                    else:
                        current_paragraph += sentence + " "
                
                # Add remaining content
                if current_paragraph.strip():
                    paragraphs.append(current_paragraph.strip())
            else:
                paragraphs.append(part)
        
        # If no paragraphs found, treat entire text as one paragraph
        if not paragraphs:
            paragraphs = [context_text.strip()]
            
        return paragraphs

    def _tokenize_paragraph(self, question_tokens: torch.Tensor, paragraph_text: str):
        """
        ⚠️ DEPRECATED: Tokenize a single paragraph with question to create [CLS] + Q + P + [SEP]
        
        🚀 OPTIMIZATION: This is now done directly in RetrievalDataset.__getitem__()
        to avoid decode/re-tokenize inefficiency.
        
        Args:
            question_tokens: Token câu hỏi (không có [CLS] và [SEP])
            paragraph_text: Text đoạn văn để tokenize
            
        Returns:
            torch.Tensor: Chuỗi đã tokenized [CLS] + Q + P + [SEP]
        """
        device = question_tokens.device
        
        # Tokenize paragraph text thôi (không có special tokens)
        paragraph_tokens = self.tokenizer(
            paragraph_text,
            add_special_tokens=False,
            return_tensors='pt'
        )['input_ids'].squeeze(0)
        
        # Di chuyển lên device
        paragraph_tokens = paragraph_tokens.to(device)
        
        # Tạo chuỗi đầy đủ: [CLS] + Q + P + [SEP]
        sequence = torch.cat([
            torch.tensor([self.tokenizer.cls_token_id], device=device),
            question_tokens,
            paragraph_tokens,
            torch.tensor([self.tokenizer.sep_token_id], device=device)
        ])
        
        # Cắt ngắn nếu quá dài
        if len(sequence) > self.max_seq_len:
            # Giữ [CLS] + Q + [SEP], cắt ngắn paragraph
            question_with_special = len(question_tokens) + 2  # +2 cho [CLS] và [SEP]
            if question_with_special < self.max_seq_len:
                max_paragraph_len = self.max_seq_len - question_with_special
                truncated_paragraph = paragraph_tokens[:max_paragraph_len]
                sequence = torch.cat([
                    torch.tensor([self.tokenizer.cls_token_id], device=device),
                    question_tokens,
                    truncated_paragraph,
                    torch.tensor([self.tokenizer.sep_token_id], device=device)
                ])
            else:
                # Câu hỏi quá dài, cắt ngắn tất cả
                sequence = sequence[:self.max_seq_len]
                
        return sequence

    def _extract_paragraph_tokens(self, sequence: torch.Tensor, question_tokens: torch.Tensor):
        """
        Trích xuất token đoạn văn từ chuỗi [CLS] + Q + P + [SEP]
        
        Args:
            sequence: Chuỗi đã tokenized đầy đủ
            question_tokens: Token câu hỏi (không có [CLS] và [SEP])
            
        Returns:
            torch.Tensor: Chỉ các token đoạn văn (phần P)
        """
        # Định dạng chuỗi: [CLS] + Q + P + [SEP]
        question_start = 1  # Bỏ qua [CLS]
        question_end = question_start + len(question_tokens)
        paragraph_start = question_end
        
        # Tìm vị trí [SEP] (nên ở cuối)
        paragraph_end = len(sequence) - 1  # Bỏ qua [SEP] cuối
        
        # Trích xuất token đoạn văn
        if paragraph_start < paragraph_end:
            paragraph_tokens = sequence[paragraph_start:paragraph_end]
        else:
            paragraph_tokens = torch.tensor([], device=sequence.device, dtype=sequence.dtype)
            
        return paragraph_tokens

    def _create_multi_hop_sequence(self, question_tokens: torch.Tensor, selected_paragraphs: List[torch.Tensor], new_paragraph: torch.Tensor):
        """
        Tạo chuỗi multi-hop: [CLS] + Q + P1 + P2 + ... + Pnew + [SEP]
        
        Args:
            question_tokens: Token câu hỏi (không có special tokens)
            selected_paragraphs: Danh sách tensor token đoạn văn đã chọn từ hop trước
            new_paragraph: Token đoạn văn mới để thêm vào
            
        Returns:
            torch.Tensor: Chuỗi kết hợp
        """
        device = question_tokens.device
        
        # Xây dựng các phần của chuỗi
        sequence_parts = [
            torch.tensor([self.tokenizer.cls_token_id], device=device),
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
        sequence_parts.append(torch.tensor([self.tokenizer.sep_token_id], device=device))
        
        # Kết hợp tất cả các phần
        combined_sequence = torch.cat(sequence_parts)
        
        # Cắt ngắn nếu quá dài
        if len(combined_sequence) > self.max_seq_len:
            # Giữ [CLS] + Q + [SEP], cắt ngắn paragraphs tỷ lệ
            question_with_special = len(question_tokens) + 2  # +2 cho [CLS] và [SEP]
            available_space = self.max_seq_len - question_with_special
            
            if available_space > 0:
                # Tính tổng độ dài paragraph
                total_paragraph_len = sum(len(p) for p in selected_paragraphs) + len(new_paragraph)
                
                if total_paragraph_len > available_space:
                    # Cắt ngắn từ cuối
                    combined_sequence = combined_sequence[:self.max_seq_len-1]
                    combined_sequence = torch.cat([
                        combined_sequence, 
                        torch.tensor([self.tokenizer.sep_token_id], device=device)
                    ])
        
        return combined_sequence

    def encode_texts(self, texts: List[str], max_length: int = None):
        """
        Encode danh sách texts sử dụng tokenizer và encoder
        """
        if max_length is None:
            max_length = self.max_seq_len
        
        # Tokenize tất cả texts
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )
        
        # Di chuyển lên cùng device với model
        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        return encoded
    
    def _hop1_forward(self, hop1_qp_ids, hop1_qp_attention_mask):
        """Forward pass for first hop - simplified without gradient checkpointing"""
        hop1_encoder_outputs = self.encoder(
            input_ids=hop1_qp_ids, 
            attention_mask=hop1_qp_attention_mask
        )[0][:, 0, :]  # [doc_num, hidden_size]
        
        hop1_projection = self.hop_classifier_layer(hop1_encoder_outputs)
        return hop1_projection
    
    def _hop_n_forward(self, hop_qp_ids, hop_qp_attention_mask):
        """Forward pass for subsequent hops - simplified without gradient checkpointing"""
        hop_encoder_outputs = self.encoder(
            input_ids=hop_qp_ids, 
            attention_mask=hop_qp_attention_mask
        )[0][:, 0, :]  # [vec_num, hidden_size]
        
        hop_projection = self.hop_n_classifier_layer(hop_encoder_outputs)
        return hop_projection

    def forward(self, q_codes, p_codes, sf_idx, hop=0, context_mapping=None):
        """
        🚀 TỐI ƯU: Forward pass với đoạn văn đã chia sẵn (không decode/re-tokenize!)
        
        Pipeline:
        1. Sử dụng chuỗi đoạn văn đã chia sẵn trực tiếp từ data loader
        2. Áp dụng multi-hop reasoning với concatenation phù hợp
        3. Chuyển đổi dự đoán đoạn văn ngược về dự đoán context
        
        Args:
            q_codes: List chứa token câu hỏi sạch (không có [CLS], [SEP])
            p_codes: List các chuỗi đoạn văn [CLS] + Q + P + [SEP] (đã pre-tokenized)
            sf_idx: Chỉ số supporting fact (cấp context)
            hop: Số hops cho inference mode
            context_mapping: List ánh xạ chỉ số đoạn văn → chỉ số context gốc
        """
        device = q_codes[0].device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Khởi tạo biến
        loss_function = nn.CrossEntropyLoss()
        
        # 🚀 TỐI ƯU: Sử dụng đoạn văn đã chia sẵn trực tiếp (không trích xuất context!)
        question_tokens = q_codes[0]  # Token câu hỏi sạch
        all_paragraph_sequences = p_codes  # Đoạn văn đã tokenized!
        context_to_paragraph_mapping = context_mapping if context_mapping else list(range(len(p_codes)))
        focal_loss_function = None
        
        if self.use_focal:
            focal_loss_function = FocalLoss()
        
        # 🚀 TỐI ƯU: Không còn trích xuất context! Sử dụng đoạn văn đã chia sẵn trực tiếp
        # BƯỚC 1: Đã có chuỗi đoạn văn từ data loader (không decode/re-tokenize!)
        
        # BƯỚC 2: Xác định tham số training
        if self.training:
            sf_idx = sf_idx[0]
            hops = len(sf_idx)
        else:
            hops = hop if hop > 0 else (len(sf_idx[0]) if sf_idx and len(sf_idx) > 0 else 2)
        
        if len(all_paragraph_sequences) <= hops or hops < 1:
            return {
                'current_preds': [list(range(min(hops, len(all_paragraph_sequences))))],
                'loss': total_loss
            }
        
        # BƯỚC 3: Multi-hop reasoning
        current_preds = []  # Mỗi beam theo dõi chỉ số đoạn văn
        selected_paragraph_tokens = []  # Mỗi beam theo dõi token đoạn văn đã chọn
        
        for hop_idx in range(hops):
            if hop_idx == 0:
                # HOP ĐẦU TIÊN: Xử lý tất cả đoạn văn độc lập
                max_len = max(len(seq) for seq in all_paragraph_sequences)
                num_paragraphs = len(all_paragraph_sequences)
                
                hop1_qp_ids = torch.zeros([num_paragraphs, max_len], device=device, dtype=torch.long)
                hop1_qp_attention_mask = torch.zeros([num_paragraphs, max_len], device=device, dtype=torch.long)
                
                if self.training:
                    hop1_label = torch.zeros([num_paragraphs], dtype=torch.long, device=device)
                
                # Điền tensor với chuỗi đoạn văn
                for i, paragraph_seq in enumerate(all_paragraph_sequences):
                    seq_len = len(paragraph_seq)
                    hop1_qp_ids[i, :seq_len] = paragraph_seq
                    hop1_qp_attention_mask[i, :seq_len] = (paragraph_seq != self.tokenizer.pad_token_id).long()
                    
                    if self.training:
                        # Kiểm tra xem đoạn văn này có thuộc supporting context không
                        original_ctx_idx = context_to_paragraph_mapping[i]
                        if original_ctx_idx in sf_idx:
                            hop1_label[i] = 1
                
                # Forward pass cho hop đầu tiên
                hop1_projection = self._hop1_forward(hop1_qp_ids, hop1_qp_attention_mask)
                
                if self.training:
                    total_loss = total_loss + loss_function(hop1_projection, hop1_label)
                
                # Chọn top beam_size đoạn văn
                _, hop1_pred_paragraphs = hop1_projection[:, 1].topk(self.beam_size, dim=-1)
                
                # Khởi tạo theo dõi beam
                current_preds = [[idx.item()] for idx in hop1_pred_paragraphs]
                selected_paragraph_tokens = []
                
                for pred_idx in hop1_pred_paragraphs:
                    pred_idx = pred_idx.item()
                    # Trích xuất token đoạn văn từ chuỗi đã chọn
                    selected_seq = all_paragraph_sequences[pred_idx]
                    paragraph_tokens = self._extract_paragraph_tokens(selected_seq, question_tokens)
                    selected_paragraph_tokens.append([paragraph_tokens])
            
            else:
                # HOP TIẾP THEO: Kết hợp lựa chọn trước với ứng viên mới
                next_sequences = []
                next_labels = []
                next_pred_mapping = []
                
                for beam_idx in range(self.beam_size):
                    beam_selected_paragraphs = selected_paragraph_tokens[beam_idx]
                    beam_used_indices = set(current_preds[beam_idx])
                    
                    # Thử từng đoạn văn chưa sử dụng làm ứng viên tiếp theo
                    for para_idx, paragraph_seq in enumerate(all_paragraph_sequences):
                        if para_idx in beam_used_indices:
                            continue
                        
                        # Trích xuất token đoạn văn mới
                        new_paragraph_tokens = self._extract_paragraph_tokens(paragraph_seq, question_tokens)
                        
                        # Tạo chuỗi multi-hop
                        multi_hop_seq = self._create_multi_hop_sequence(
                            question_tokens, 
                            beam_selected_paragraphs, 
                            new_paragraph_tokens
                        )
                        
                        next_sequences.append(multi_hop_seq)
                        next_pred_mapping.append(current_preds[beam_idx] + [para_idx])
                        
                        # Nhãn cho training
                        if self.training:
                            new_pred_set = set(current_preds[beam_idx] + [para_idx])
                            target_contexts = set()
                            for p_idx in new_pred_set:
                                target_contexts.add(context_to_paragraph_mapping[p_idx])
                            
                            # Kiểm tra xem tổ hợp này có khớp với supporting facts mục tiêu không
                            if target_contexts == set(sf_idx[:hop_idx+1]):
                                next_labels.append(1)
                            else:
                                next_labels.append(0)
                
                if not next_sequences:
                    break
                
                # Chuẩn bị tensor cho hop này
                max_len = max(len(seq) for seq in next_sequences)
                num_candidates = len(next_sequences)
                
                hop_qp_ids = torch.zeros([num_candidates, max_len], device=device, dtype=torch.long)
                hop_qp_attention_mask = torch.zeros([num_candidates, max_len], device=device, dtype=torch.long)
                
                if self.training:
                    hop_label = torch.tensor(next_labels, dtype=torch.long, device=device)
                
                # Điền tensor
                for i, seq in enumerate(next_sequences):
                    seq_len = len(seq)
                    hop_qp_ids[i, :seq_len] = seq
                    hop_qp_attention_mask[i, :seq_len] = (seq != self.tokenizer.pad_token_id).long()
                
                # Forward pass cho hop tiếp theo
                hop_projection = self._hop_n_forward(hop_qp_ids, hop_qp_attention_mask)
                
                if self.training:
                    if self.use_focal:
                        total_loss = total_loss + focal_loss_function(hop_projection, hop_label)
                    else:
                        total_loss = total_loss + loss_function(hop_projection, hop_label)
                
                # Chọn top beam_size ứng viên
                _, hop_pred_indices = hop_projection[:, 1].topk(self.beam_size, dim=-1)
                
                # Cập nhật theo dõi beam
                new_current_preds = []
                new_selected_paragraph_tokens = []
                
                for pred_idx in hop_pred_indices:
                    pred_idx = pred_idx.item()
                    selected_prediction = next_pred_mapping[pred_idx]
                    new_current_preds.append(selected_prediction)
                    
                    # Xây dựng danh sách token đoạn văn cho beam này
                    beam_paragraph_tokens = []
                    for para_idx in selected_prediction:
                        para_seq = all_paragraph_sequences[para_idx]
                        para_tokens = self._extract_paragraph_tokens(para_seq, question_tokens)
                        beam_paragraph_tokens.append(para_tokens)
                    
                    new_selected_paragraph_tokens.append(beam_paragraph_tokens)
                
                current_preds = new_current_preds
                selected_paragraph_tokens = new_selected_paragraph_tokens
        
        # Chuyển đổi dự đoán đoạn văn ngược về dự đoán context
        final_context_preds = []
        for beam_paragraphs in current_preds:
            context_indices = []
            for para_idx in beam_paragraphs:
                ctx_idx = context_to_paragraph_mapping[para_idx]
                if ctx_idx not in context_indices:
                    context_indices.append(ctx_idx)
            final_context_preds.append(context_indices)
        
        # Trả về kết quả
        return {
            'current_preds': final_context_preds,
            'final_preds': final_context_preds,
            'paragraph_preds': current_preds,  # Cũng trả về dự đoán cấp đoạn văn
            'loss': total_loss
        }
    
    def retrieve_contexts(self, question: str, contexts: List[Dict[str, Any]], max_hops: int = 3):
        """
        Interface cấp cao cho retrieval
        """
        self.eval()
        
        # Chuẩn bị đầu vào (đơn giản hóa cho demo)
        # Trong thực tế, bạn cần tokenize và chuẩn bị input đúng cách
        with torch.no_grad():
            # Đây là phiên bản đơn giản - bạn cần implement tokenization phù hợp
            # dựa trên yêu cầu cụ thể của bạn
            pass
        
        # Trả về top contexts (placeholder)
        return contexts[:self.beam_size]
    
    def get_model_summary(self):
        """
        Get comprehensive model summary with parameters and configuration
        """
        summary = {
            'model_info': {
                'total_parameters': self.count_parameters(),
                'model_name': 'microsoft/deberta-v3-base',
                'architecture': 'Advanced Multi-Hop Retriever with Focal Loss'
            },
            'configuration': {
                'max_seq_len': self.max_seq_len,
                'mean_passage_len': self.mean_passage_len,
                'beam_size': self.beam_size,
                'use_focal': self.use_focal,
                'use_early_stop': self.use_early_stop,
                'gradient_checkpointing': self.gradient_checkpointing
            },
            'capabilities': {
                'multi_hop_reasoning': True,
                'beam_search': True,
                'focal_loss': self.use_focal,
                'paragraph_level_processing': True,
                'context_combination': True
            }
        }
        return summary
    
    def print_model_summary(self):
        """Print formatted model summary"""
        summary = self.get_model_summary()
        
        print(f"\n🧠 Advanced Multi-Hop Retriever Model Summary")
        print(f"=" * 60)
        
        # Model info
        print(f"📊 Model Information:")
        for key, value in summary['model_info'].items():
            if key == 'total_parameters':
                print(f"   • {key.replace('_', ' ').title()}: {value:,}")
            else:
                print(f"   • {key.replace('_', ' ').title()}: {value}")
        
        # Configuration
        print(f"\n⚙️  Configuration:")
        for key, value in summary['configuration'].items():
            print(f"   • {key.replace('_', ' ').title()}: {value}")
        
        # Capabilities
        print(f"\n🚀 Capabilities:")
        for key, value in summary['capabilities'].items():
            status = "✅ Enabled" if value else "❌ Disabled"
            print(f"   • {key.replace('_', ' ').title()}: {status}")
        
        print(f"=" * 60)


# Factory function
def create_advanced_retriever(model_name="microsoft/deberta-v3-small", **kwargs):
    """
    Hàm factory để tạo model advanced retriever
    """
    try:
        config = AutoConfig.from_pretrained(model_name)
        
        # Lọc bỏ các arguments không hợp lệ cho Retriever
        valid_args = {}
        retriever_params = {
            'encoder_class', 'max_seq_len', 'mean_passage_len', 
            'beam_size', 'gradient_checkpointing', 'use_focal', 'use_early_stop'
        }
        
        for key, value in kwargs.items():
            if key in retriever_params:
                valid_args[key] = value
        
        # Đảm bảo có beam_size mặc định
        if 'beam_size' not in valid_args:
            valid_args['beam_size'] = 2
        
        model = Retriever(
            config=config,
            model_name=model_name,
            **valid_args
        )
        return model
    except Exception as e:
        print(f"Lỗi khi tạo model: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Create model
    model = create_advanced_retriever(
        model_name="microsoft/deberta-v3-base",  # Fallback model
        beam_size=2,
        use_focal=True,
        use_early_stop=True
    )
    
    print(f"🧠 Model created with {model.count_parameters():,} parameters")
    print(f"📊 Configuration: beam_size={model.beam_size}, focal_loss={model.use_focal}")
