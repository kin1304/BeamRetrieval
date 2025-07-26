#!/usr/bin/env python3
"""
Advanced Multi-Hop Retriever Model with Focal Loss
Based on the architecture you provided for BeamRetrieval system
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
    Focal Loss implementation for handling class imbalance
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
    Multi-hop Retriever with beam search and focal loss
    Based on your specified architecture
    """
    
    def __init__(self,
                 config=None,
                 model_name="microsoft/deberta-v3-small",
                 encoder_class=None,
                 max_seq_len=512,
                 mean_passage_len=70,
                 beam_size=2,  # Matching your beam_width=2 requirement
                 gradient_checkpointing=False,
                 use_focal=True,  # Enable focal loss by default
                 use_early_stop=True,
                 ):
        super().__init__()
        
        # Load config if not provided
        if config is None:
            config = AutoConfig.from_pretrained(model_name)
        self.config = config
        
        # Load encoder
        if encoder_class is None:
            from transformers import AutoModel
            encoder_class = AutoModel
        
        self.encoder = encoder_class.from_pretrained(model_name, config=config)
        
        # Store parameters
        self.max_seq_len = max_seq_len
        self.mean_passage_len = mean_passage_len
        self.beam_size = beam_size
        # Gradient checkpointing is fundamentally incompatible with multi-hop reasoning
        # due to computational graph reuse between hops causing "backward second time" errors
        # Alternative memory optimization: use smaller batch size, model parallelism, or mixed precision
        self.gradient_checkpointing = False
        self.use_focal = use_focal
        self.use_early_stop = use_early_stop
        self.use_label_order = False  # Add this attribute
        
        # Classification layers for different hops
        self.hop_classifier_layer = nn.Linear(config.hidden_size, 2)
        self.hop_n_classifier_layer = nn.Linear(config.hidden_size, 2)
        
        # Note: Gradient checkpointing disabled for multi-hop compatibility
        # Memory optimization alternatives: mixed precision, model parallelism, smaller batch sizes
        
        # Initialize tokenizer (suppress fast tokenizer warning for DeBERTa)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*sentencepiece tokenizer.*")
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        
        logger.info(f"🧠 Retriever initialized with {self.count_parameters():,} parameters")
        logger.info(f"📊 Beam size: {beam_size}, Max seq len: {max_seq_len}")
        logger.info(f"🎯 Using focal loss: {use_focal}, Early stop: {use_early_stop}")
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _split_context_to_paragraphs(self, context_text: str, max_paragraph_len: int = 200):
        """
        Split context text into paragraphs for better processing
        
        Args:
            context_text: Full context text
            max_paragraph_len: Maximum length per paragraph
            
        Returns:
            List of paragraph strings
        """
        # Split by common paragraph separators
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
        Tokenize a single paragraph with question to create [CLS] + Q + P + [SEP]
        
        Args:
            question_tokens: Question tokens (without [CLS] and [SEP])
            paragraph_text: Paragraph text to tokenize
            
        Returns:
            torch.Tensor: Tokenized sequence [CLS] + Q + P + [SEP]
        """
        device = question_tokens.device
        
        # Tokenize paragraph text only (no special tokens)
        paragraph_tokens = self.tokenizer(
            paragraph_text,
            add_special_tokens=False,
            return_tensors='pt'
        )['input_ids'].squeeze(0)
        
        # Move to device
        paragraph_tokens = paragraph_tokens.to(device)
        
        # Create full sequence: [CLS] + Q + P + [SEP]
        sequence = torch.cat([
            torch.tensor([self.tokenizer.cls_token_id], device=device),
            question_tokens,
            paragraph_tokens,
            torch.tensor([self.tokenizer.sep_token_id], device=device)
        ])
        
        # Truncate if too long
        if len(sequence) > self.max_seq_len:
            # Keep [CLS] + Q + [SEP], truncate paragraph
            question_with_special = len(question_tokens) + 2  # +2 for [CLS] and [SEP]
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
                # Question too long, truncate everything
                sequence = sequence[:self.max_seq_len]
                
        return sequence

    def _extract_paragraph_tokens(self, sequence: torch.Tensor, question_tokens: torch.Tensor):
        """
        Extract paragraph tokens from [CLS] + Q + P + [SEP] sequence
        
        Args:
            sequence: Full tokenized sequence
            question_tokens: Question tokens (without [CLS] and [SEP])
            
        Returns:
            torch.Tensor: Only the paragraph tokens (P part)
        """
        # Sequence format: [CLS] + Q + P + [SEP]
        question_start = 1  # Skip [CLS]
        question_end = question_start + len(question_tokens)
        paragraph_start = question_end
        
        # Find [SEP] position (should be at the end)
        paragraph_end = len(sequence) - 1  # Skip final [SEP]
        
        # Extract paragraph tokens
        if paragraph_start < paragraph_end:
            paragraph_tokens = sequence[paragraph_start:paragraph_end]
        else:
            paragraph_tokens = torch.tensor([], device=sequence.device, dtype=sequence.dtype)
            
        return paragraph_tokens

    def _create_multi_hop_sequence(self, question_tokens: torch.Tensor, selected_paragraphs: List[torch.Tensor], new_paragraph: torch.Tensor):
        """
        Create multi-hop sequence: [CLS] + Q + P1 + P2 + ... + Pnew + [SEP]
        
        Args:
            question_tokens: Question tokens (without special tokens)
            selected_paragraphs: List of selected paragraph token tensors from previous hops
            new_paragraph: New paragraph tokens to add
            
        Returns:
            torch.Tensor: Combined sequence
        """
        device = question_tokens.device
        
        # Build sequence parts
        sequence_parts = [
            torch.tensor([self.tokenizer.cls_token_id], device=device),
            question_tokens
        ]
        
        # Add selected paragraphs from previous hops
        for paragraph_tokens in selected_paragraphs:
            if len(paragraph_tokens) > 0:
                sequence_parts.append(paragraph_tokens)
        
        # Add new paragraph
        if len(new_paragraph) > 0:
            sequence_parts.append(new_paragraph)
            
        # Add final [SEP]
        sequence_parts.append(torch.tensor([self.tokenizer.sep_token_id], device=device))
        
        # Concatenate all parts
        combined_sequence = torch.cat(sequence_parts)
        
        # Truncate if too long
        if len(combined_sequence) > self.max_seq_len:
            # Keep [CLS] + Q + [SEP], truncate paragraphs proportionally
            question_with_special = len(question_tokens) + 2  # +2 for [CLS] and [SEP]
            available_space = self.max_seq_len - question_with_special
            
            if available_space > 0:
                # Calculate total paragraph length
                total_paragraph_len = sum(len(p) for p in selected_paragraphs) + len(new_paragraph)
                
                if total_paragraph_len > available_space:
                    # Truncate from the end
                    combined_sequence = combined_sequence[:self.max_seq_len-1]
                    combined_sequence = torch.cat([
                        combined_sequence, 
                        torch.tensor([self.tokenizer.sep_token_id], device=device)
                    ])
        
        return combined_sequence

    def encode_texts(self, texts: List[str], max_length: int = None):
        """
        Encode a list of texts using the tokenizer and encoder
        """
        if max_length is None:
            max_length = self.max_seq_len
        
        # Tokenize all texts
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )
        
        # Move to same device as model
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

    def forward(self, q_codes, c_codes, sf_idx, hop=0):
        """
        Forward pass with multi-hop reasoning using paragraph-based processing
        
        Pipeline:
        1. Extract question tokens
        2. Split each context into paragraphs  
        3. Tokenize each paragraph separately
        4. Apply multi-hop reasoning with proper concatenation
        """
        device = q_codes[0].device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Initialize variables
        loss_function = nn.CrossEntropyLoss()
        focal_loss_function = None
        
        if self.use_focal:
            focal_loss_function = FocalLoss()
        
        # STEP 1: Extract clean question tokens (without special tokens)
        question_ids = q_codes[0]  # Reference question tokens
        context_sequences = c_codes  # Original: [CLS] + Q + C + [SEP] for each context
        
        # Clean question tokens (remove padding and special tokens)
        question_tokens = question_ids[question_ids != self.tokenizer.pad_token_id]
        if len(question_tokens) > 0 and question_tokens[0] == self.tokenizer.cls_token_id:
            question_tokens = question_tokens[1:]  # Remove [CLS]
        if len(question_tokens) > 0 and question_tokens[-1] == self.tokenizer.sep_token_id:
            question_tokens = question_tokens[:-1]  # Remove [SEP]
        
        # STEP 2: Process contexts - extract and split into paragraphs
        context_paragraphs = []  # List of lists: each context -> list of paragraph tokens
        all_paragraph_sequences = []  # Flattened list of [CLS] + Q + P + [SEP] sequences
        context_to_paragraph_mapping = []  # Maps paragraph index to original context index
        
        for ctx_idx, context_sequence in enumerate(context_sequences):
            # Extract context text from the original sequence
            # Format: [CLS] + Q + C + [SEP] -> extract C part
            
            # Find where context starts (after question)
            # Simple approach: find first occurrence of non-question tokens
            context_start_idx = 1 + len(question_tokens)  # Skip [CLS] + Q
            context_end_idx = len(context_sequence) - 1   # Skip final [SEP]
            
            if context_start_idx < context_end_idx:
                context_tokens = context_sequence[context_start_idx:context_end_idx]
                # Convert tokens back to text for paragraph splitting
                context_text = self.tokenizer.decode(context_tokens, skip_special_tokens=True)
            else:
                context_text = ""
            
            # Split context into paragraphs
            paragraphs = self._split_context_to_paragraphs(context_text)
            
            # Tokenize each paragraph
            paragraph_token_sequences = []
            for paragraph_text in paragraphs:
                paragraph_sequence = self._tokenize_paragraph(question_tokens, paragraph_text)
                paragraph_token_sequences.append(paragraph_sequence)
                all_paragraph_sequences.append(paragraph_sequence)
                context_to_paragraph_mapping.append(ctx_idx)
            
            context_paragraphs.append(paragraph_token_sequences)
        
        # STEP 3: Determine training parameters
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
        
        # STEP 4: Multi-hop reasoning
        current_preds = []  # Each beam tracks paragraph indices
        selected_paragraph_tokens = []  # Each beam tracks selected paragraph tokens
        
        for hop_idx in range(hops):
            if hop_idx == 0:
                # FIRST HOP: Process all paragraphs independently
                max_len = max(len(seq) for seq in all_paragraph_sequences)
                num_paragraphs = len(all_paragraph_sequences)
                
                hop1_qp_ids = torch.zeros([num_paragraphs, max_len], device=device, dtype=torch.long)
                hop1_qp_attention_mask = torch.zeros([num_paragraphs, max_len], device=device, dtype=torch.long)
                
                if self.training:
                    hop1_label = torch.zeros([num_paragraphs], dtype=torch.long, device=device)
                
                # Fill tensors with paragraph sequences
                for i, paragraph_seq in enumerate(all_paragraph_sequences):
                    seq_len = len(paragraph_seq)
                    hop1_qp_ids[i, :seq_len] = paragraph_seq
                    hop1_qp_attention_mask[i, :seq_len] = (paragraph_seq != self.tokenizer.pad_token_id).long()
                    
                    if self.training:
                        # Check if this paragraph belongs to a supporting context
                        original_ctx_idx = context_to_paragraph_mapping[i]
                        if original_ctx_idx in sf_idx:
                            hop1_label[i] = 1
                
                # Forward pass for first hop
                hop1_projection = self._hop1_forward(hop1_qp_ids, hop1_qp_attention_mask)
                
                if self.training:
                    total_loss = total_loss + loss_function(hop1_projection, hop1_label)
                
                # Select top beam_size paragraphs
                _, hop1_pred_paragraphs = hop1_projection[:, 1].topk(self.beam_size, dim=-1)
                
                # Initialize beam tracking
                current_preds = [[idx.item()] for idx in hop1_pred_paragraphs]
                selected_paragraph_tokens = []
                
                for pred_idx in hop1_pred_paragraphs:
                    pred_idx = pred_idx.item()
                    # Extract paragraph tokens from the selected sequence
                    selected_seq = all_paragraph_sequences[pred_idx]
                    paragraph_tokens = self._extract_paragraph_tokens(selected_seq, question_tokens)
                    selected_paragraph_tokens.append([paragraph_tokens])
            
            else:
                # SUBSEQUENT HOPS: Combine previous selections with new candidates
                next_sequences = []
                next_labels = []
                next_pred_mapping = []
                
                for beam_idx in range(self.beam_size):
                    beam_selected_paragraphs = selected_paragraph_tokens[beam_idx]
                    beam_used_indices = set(current_preds[beam_idx])
                    
                    # Try each unused paragraph as next candidate
                    for para_idx, paragraph_seq in enumerate(all_paragraph_sequences):
                        if para_idx in beam_used_indices:
                            continue
                        
                        # Extract new paragraph tokens
                        new_paragraph_tokens = self._extract_paragraph_tokens(paragraph_seq, question_tokens)
                        
                        # Create multi-hop sequence
                        multi_hop_seq = self._create_multi_hop_sequence(
                            question_tokens, 
                            beam_selected_paragraphs, 
                            new_paragraph_tokens
                        )
                        
                        next_sequences.append(multi_hop_seq)
                        next_pred_mapping.append(current_preds[beam_idx] + [para_idx])
                        
                        # Label for training
                        if self.training:
                            new_pred_set = set(current_preds[beam_idx] + [para_idx])
                            target_contexts = set()
                            for p_idx in new_pred_set:
                                target_contexts.add(context_to_paragraph_mapping[p_idx])
                            
                            # Check if this combination matches the target supporting facts
                            if target_contexts == set(sf_idx[:hop_idx+1]):
                                next_labels.append(1)
                            else:
                                next_labels.append(0)
                
                if not next_sequences:
                    break
                
                # Prepare tensors for this hop
                max_len = max(len(seq) for seq in next_sequences)
                num_candidates = len(next_sequences)
                
                hop_qp_ids = torch.zeros([num_candidates, max_len], device=device, dtype=torch.long)
                hop_qp_attention_mask = torch.zeros([num_candidates, max_len], device=device, dtype=torch.long)
                
                if self.training:
                    hop_label = torch.tensor(next_labels, dtype=torch.long, device=device)
                
                # Fill tensors
                for i, seq in enumerate(next_sequences):
                    seq_len = len(seq)
                    hop_qp_ids[i, :seq_len] = seq
                    hop_qp_attention_mask[i, :seq_len] = (seq != self.tokenizer.pad_token_id).long()
                
                # Forward pass for subsequent hop
                hop_projection = self._hop_n_forward(hop_qp_ids, hop_qp_attention_mask)
                
                if self.training:
                    if self.use_focal:
                        total_loss = total_loss + focal_loss_function(hop_projection, hop_label)
                    else:
                        total_loss = total_loss + loss_function(hop_projection, hop_label)
                
                # Select top beam_size candidates
                _, hop_pred_indices = hop_projection[:, 1].topk(self.beam_size, dim=-1)
                
                # Update beam tracking
                new_current_preds = []
                new_selected_paragraph_tokens = []
                
                for pred_idx in hop_pred_indices:
                    pred_idx = pred_idx.item()
                    selected_prediction = next_pred_mapping[pred_idx]
                    new_current_preds.append(selected_prediction)
                    
                    # Build paragraph tokens list for this beam
                    beam_paragraph_tokens = []
                    for para_idx in selected_prediction:
                        para_seq = all_paragraph_sequences[para_idx]
                        para_tokens = self._extract_paragraph_tokens(para_seq, question_tokens)
                        beam_paragraph_tokens.append(para_tokens)
                    
                    new_selected_paragraph_tokens.append(beam_paragraph_tokens)
                
                current_preds = new_current_preds
                selected_paragraph_tokens = new_selected_paragraph_tokens
        
        # Convert paragraph predictions back to context predictions
        final_context_preds = []
        for beam_paragraphs in current_preds:
            context_indices = []
            for para_idx in beam_paragraphs:
                ctx_idx = context_to_paragraph_mapping[para_idx]
                if ctx_idx not in context_indices:
                    context_indices.append(ctx_idx)
            final_context_preds.append(context_indices)
        
        # Return results
        return {
            'current_preds': final_context_preds,
            'final_preds': final_context_preds,
            'paragraph_preds': current_preds,  # Also return paragraph-level predictions
            'loss': total_loss
        }
    
    def retrieve_contexts(self, question: str, contexts: List[Dict[str, Any]], max_hops: int = 3):
        """
        High-level interface for retrieval
        """
        self.eval()
        
        # Prepare inputs (simplified for demo)
        # In practice, you'd need to properly tokenize and prepare the inputs
        with torch.no_grad():
            # This is a simplified version - you'd need to implement proper tokenization
            # based on your specific requirements
            pass
        
        # Return top contexts (placeholder)
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
    Factory function to create advanced retriever model
    """
    try:
        config = AutoConfig.from_pretrained(model_name)
        model = Retriever(
            config=config,
            model_name=model_name,
            **kwargs
        )
        logger.info(f"✅ Created advanced retriever with {model_name}")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to create retriever: {e}")
        raise

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
