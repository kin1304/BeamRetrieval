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
from transformers import AutoTokenizer, AutoModel, AutoConfig

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
        self.gradient_checkpointing = gradient_checkpointing
        self.use_focal = use_focal
        self.use_early_stop = use_early_stop
        self.use_label_order = False  # Add this attribute
        
        # Classification layers for different hops
        self.hop_classifier_layer = nn.Linear(config.hidden_size, 2)
        self.hop_n_classifier_layer = nn.Linear(config.hidden_size, 2)
        
        # Enable gradient checkpointing if requested
        if self.gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        logger.info(f"🧠 Retriever initialized with {self.count_parameters():,} parameters")
        logger.info(f"📊 Beam size: {beam_size}, Max seq len: {max_seq_len}")
        logger.info(f"🎯 Using focal loss: {use_focal}, Early stop: {use_early_stop}")
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
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
    
    def forward(self, q_codes, c_codes, sf_idx, hop=0):
        """
        Forward pass with multi-hop reasoning
        """
        device = q_codes[0].device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Initialize variables
        last_prediction = None
        pre_question_ids = None
        loss_function = nn.CrossEntropyLoss()
        focal_loss_function = None
        
        if self.use_focal:
            focal_loss_function = FocalLoss()
        
        question_ids = q_codes[0]  # Reference for question length
        context_ids = c_codes  # These are now: [CLS] + Q + C + [SEP]
        current_preds = []
        
        if self.training:
            sf_idx = sf_idx[0]
            sf = sf_idx
            hops = len(sf)
        else:
            hops = hop if hop > 0 else (len(sf_idx[0]) if sf_idx and len(sf_idx) > 0 else 2)
        
        if len(context_ids) <= hops or hops < 1:
            return {
                'current_preds': [list(range(hops))],
                'loss': total_loss
            }
        
        # Multi-hop reasoning loop
        for idx in range(hops):
            if idx == 0:
                # First hop - context_ids are already formatted as [CLS] + Q + C + [SEP]
                max_len = max(len(c) for c in context_ids)
                hop1_qp_ids = torch.zeros([len(context_ids), max_len], 
                                        device=device, dtype=torch.long)
                hop1_qp_attention_mask = torch.zeros([len(context_ids), max_len], 
                                                   device=device, dtype=torch.long)
                
                if self.training:
                    hop1_label = torch.zeros([len(context_ids)], dtype=torch.long, device=device)
                
                for i in range(len(context_ids)):
                    # Use the pre-formatted sequences directly: [CLS] + Q + C + [SEP]
                    seq_len = len(context_ids[i])
                    hop1_qp_ids[i, :seq_len] = context_ids[i]
                    
                    # Create attention mask - attend to all non-pad tokens
                    hop1_qp_attention_mask[i, :seq_len] = (context_ids[i] != self.tokenizer.pad_token_id).long()
                    
                    if self.training:
                        if self.use_label_order:
                            if len(sf_idx) > 0 and i == sf_idx[0]:
                                hop1_label[i] = 1
                        else:
                            if i in sf_idx:
                                hop1_label[i] = 1
                
                # Store question context info for next hops (simplified)
                next_question_ids = context_ids
                
                # Encode first hop
                hop1_encoder_outputs = self.encoder(
                    input_ids=hop1_qp_ids, 
                    attention_mask=hop1_qp_attention_mask
                )[0][:, 0, :]  # [doc_num, hidden_size]
                
                if self.training and self.gradient_checkpointing:
                    hop1_projection = torch.utils.checkpoint.checkpoint(
                        self.hop_classifier_layer, hop1_encoder_outputs
                    )
                else:
                    hop1_projection = self.hop_classifier_layer(hop1_encoder_outputs)
                
                if self.training:
                    total_loss = total_loss + loss_function(hop1_projection, hop1_label)
                
                _, hop1_pred_documents = hop1_projection[:, 1].topk(self.beam_size, dim=-1)
                last_prediction = hop1_pred_documents
                pre_question_ids = next_question_ids
                current_preds = [[item.item()] for item in hop1_pred_documents]
            
            else:
                # Subsequent hops
                qp_len_total = {}
                max_qp_len = 0
                last_pred_idx = set()
                
                if self.training:
                    # Early stopping check
                    flag = False
                    for i in range(self.beam_size):
                        if self.use_label_order:
                            if idx > 0 and len(sf_idx) > idx-1 and current_preds[i][-1] == sf_idx[idx - 1]:
                                flag = True
                                break
                        else:
                            if set(current_preds[i]) == set(sf_idx[:idx]):
                                flag = True
                                break
                    
                    if not flag and self.use_early_stop:
                        break
                
                # Prepare for beam expansion
                for i in range(self.beam_size):
                    pred_doc = last_prediction[i]
                    last_pred_idx.add(current_preds[i][-1])
                    # For subsequent hops, create new combinations with previous context
                    new_question_context = pre_question_ids[pred_doc] if isinstance(pre_question_ids, list) else context_ids[pred_doc]
                    qp_len = {}
                    
                    for j in range(len(context_ids)):
                        if j in current_preds[i] or j in last_pred_idx:
                            continue
                        # New format: [CLS] + Q + previous_contexts + new_context + [SEP]
                        # Estimate combined length
                        qp_len[j] = min(self.max_seq_len, len(new_question_context) + len(context_ids[j]))
                        max_qp_len = max(max_qp_len, qp_len[j])
                    
                    qp_len_total[i] = qp_len
                
                if len(qp_len_total) < 1:
                    break
                

                vector_num = sum([len(v) for k, v in qp_len_total.items()])
                
                # Setup vectors for current hop
                hop_qp_ids = torch.zeros([vector_num, max_qp_len], 
                                       device=device, dtype=torch.long)
                hop_qp_attention_mask = torch.zeros([vector_num, max_qp_len], 
                                                  device=device, dtype=torch.long)
                
                if self.training:
                    hop_label = torch.zeros([vector_num], dtype=torch.long, device=device)
                
                vec_idx = 0
                pred_mapping = []
                next_question_ids = []
                last_pred_idx = set()
                
                # Process each beam
                for i in range(self.beam_size):
                    pred_doc = last_prediction[i]
                    last_pred_idx.add(current_preds[i][-1])
                    prev_sequence = pre_question_ids[pred_doc]
                    
                    for j in range(len(context_ids)):
                        if j in current_preds[i] or j in last_pred_idx:
                            continue
                        
                        # Create new sequence: combine previous sequence with new context
                        # Format: [CLS] + Q + prev_contexts + new_context + [SEP]
                        
                        # Extract question and previous contexts (remove [SEP] from end)
                        prev_without_sep = prev_sequence[:-1] if prev_sequence[-1] == self.tokenizer.sep_token_id else prev_sequence
                        
                        # Extract new context (remove [CLS] from beginning and [SEP] from end if present)
                        new_context = context_ids[j]
                        if new_context[0] == self.tokenizer.cls_token_id:
                            # Find where the actual context starts (after question part)
                            # For simplicity, take everything after first quarter as context
                            context_start = len(new_context) // 4
                            new_context_part = new_context[context_start:]
                            if new_context_part[-1] == self.tokenizer.sep_token_id:
                                new_context_part = new_context_part[:-1]
                        else:
                            new_context_part = new_context
                        
                        # Combine: prev_sequence + new_context + [SEP]
                        combined_sequence = torch.cat([
                            prev_without_sep,
                            new_context_part,
                            torch.tensor([self.tokenizer.sep_token_id], device=device)
                        ])
                        
                        # Truncate if too long
                        if len(combined_sequence) > max_qp_len:
                            combined_sequence = combined_sequence[:max_qp_len-1]
                            combined_sequence = torch.cat([combined_sequence, torch.tensor([self.tokenizer.sep_token_id], device=device)])
                        
                        # Store for next iteration
                        next_question_ids.append(combined_sequence)
                        
                        # Fill tensors
                        seq_len = len(combined_sequence)
                        hop_qp_ids[vec_idx, :seq_len] = combined_sequence
                        hop_qp_attention_mask[vec_idx, :seq_len] = 1
                        
                        if self.training:
                            if set(current_preds[i] + [j]) == set(sf_idx[:idx+1]):
                                hop_label[vec_idx] = 1
                        
                        pred_mapping.append(current_preds[i] + [j])
                        vec_idx += 1
                
                assert len(pred_mapping) == hop_qp_ids.shape[0]
                
                # Encode current hop
                hop_encoder_outputs = self.encoder(
                    input_ids=hop_qp_ids, 
                    attention_mask=hop_qp_attention_mask
                )[0][:, 0, :]  # [vec_num, hidden_size]
                
                hop_projection_func = self.hop_n_classifier_layer
                
                if self.training and self.gradient_checkpointing:
                    hop_projection = torch.utils.checkpoint.checkpoint(
                        hop_projection_func, hop_encoder_outputs
                    )
                else:
                    hop_projection = hop_projection_func(hop_encoder_outputs)
                
                if self.training:
                    if not self.use_focal:
                        total_loss = total_loss + loss_function(hop_projection, hop_label)
                    else:
                        total_loss = total_loss + focal_loss_function(hop_projection, hop_label)
                
                _, hop_pred_documents = hop_projection[:, 1].topk(self.beam_size, dim=-1)
                last_prediction = hop_pred_documents
                pre_question_ids = next_question_ids
                current_preds = [pred_mapping[hop_pred_documents[i].item()] 
                               for i in range(self.beam_size)]
        
        res = {
            'current_preds': current_preds,
            'loss': total_loss
        }
        return res
    
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
