"""
puBERT: A BERT-based text generation model with parallel token generation and coherence checking.
Version: 0.1
Author: everyoneisgross
Date: January 2025

Key Features:
- Single-direction attention mechanism with layer combination
- Layered token generation using 4D matrix embeddings
- Coherence scoring using sentence transformers
- Streaming text generation capabilities

Architecture:
- Uses BERT for masked token prediction
- Single projection layers for attention
- Simple coherence-based sampling

Dependencies:
- PyTorch
- Transformers (Hugging Face)
- Sentence-Transformers
"""

import warnings
# Must import warnings first and set filters before other imports
warnings.filterwarnings('ignore')  # Suppress all warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Iterator, Set
from transformers import AutoTokenizer, AutoModel, BertForMaskedLM
from sentence_transformers import SentenceTransformer
from torch.nn import CrossEntropyLoss
import string
import sys
import transformers
transformers.logging.set_verbosity_error()  # Only show errors, not warnings

@dataclass
class ModelConfig:
    bert_model_name: str = 'bert-base-cased'
    tokenizer_name: str = 'bert-base-cased'
    sentence_transformer_name: str = 'all-MiniLM-L6-v2'
    attn_implementation: str = 'sdpa'

@dataclass
class GenerationConfig:

    max_length: int
    batch_size: int
    num_candidates: int
    embedding_dim: int
    context_window: int = 5
    base_temperature: float = 1.0
    min_threshold: float = 0.5
    top_k: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ParallelBERTGenerator(nn.Module):
    def __init__(self, config: GenerationConfig, model_config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config
        
        # Use model_config if provided, otherwise use defaults
        model_config = model_config or ModelConfig()
        
        # Initialize BERT model for masked prediction
        self.bert_model = BertForMaskedLM.from_pretrained(
            model_config.bert_model_name,
            attn_implementation=model_config.attn_implementation
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_name)
        
        # Initialize sentence transformer for semantic coherence
        self.sentence_transformer = SentenceTransformer(model_config.sentence_transformer_name)
        
        # Move models to device
        self.bert_model = self.bert_model.to(config.device)
        self.bert_model.eval()
        
        # Projection layers
        self.query_proj = nn.Linear(config.embedding_dim, config.embedding_dim).to(config.device)
        self.key_proj = nn.Linear(config.embedding_dim, config.embedding_dim).to(config.device)
        self.value_proj = nn.Linear(config.embedding_dim, config.embedding_dim).to(config.device)
        self.layer_combine = nn.Linear(2 * config.embedding_dim, config.embedding_dim).to(config.device)

    def encode_sequence(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Encode text sequence using sentence transformer."""
        with torch.no_grad():
            embeddings = self.sentence_transformer.encode(text, convert_to_tensor=True)
            return embeddings.to(self.config.device)

    def get_bert_predictions(self, 
                           input_ids: torch.Tensor, 
                           mask_positions: List[int]) -> torch.Tensor:
        """Get BERT predictions for masked positions."""
        with torch.no_grad():
            outputs = self.bert_model(input_ids)
            predictions = outputs.logits
            
            # Get top-k predictions for each masked position
            masked_predictions = []
            for pos in mask_positions:
                logits = predictions[0, pos]
                top_k_logits, top_k_indices = torch.topk(logits, self.config.top_k)
                masked_predictions.append((top_k_logits, top_k_indices))
            
            return masked_predictions

    def create_4d_matrix(self, initial_text: str, mask_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize the 4D generation matrix with BERT embeddings."""
        tokens = self.tokenizer(initial_text, return_tensors="pt")
        input_ids = tokens['input_ids'].to(self.config.device)
        
        matrix = torch.zeros(
            (len(input_ids[0]) + self.config.max_length,
             self.config.batch_size,
             self.config.num_candidates, 
             self.config.embedding_dim),
            device=self.config.device
        )
        
        with torch.no_grad():
            outputs = self.bert_model.bert(input_ids)
            embeddings = outputs.last_hidden_state
            matrix[:len(input_ids[0]), 0, 0, :] = embeddings[0]
        
        return matrix, input_ids

    def custom_attention(self, 
                        query: torch.Tensor,
                        key_matrix: torch.Tensor,
                        value_matrix: torch.Tensor,
                        prev_layer: Optional[torch.Tensor] = None,
                        attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Custom attention mechanism with previous layer integration."""
        Q = self.query_proj(query)
        K = self.key_proj(key_matrix)
        V = self.value_proj(value_matrix)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        attention_scores = attention_scores / np.sqrt(K.size(-1))
        
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        if prev_layer is not None:
            layer_attention = torch.matmul(attention_scores, prev_layer)
            combined_attention = torch.cat([attention_scores, layer_attention], dim=-1)
            attention_output = self.layer_combine(combined_attention)
        else:
            attention_output = torch.matmul(F.softmax(attention_scores, dim=-1), V)
            
        return attention_output

    def calculate_coherence_scores(self, 
                                 candidates: List[str],
                                 context: str) -> torch.Tensor:
        """Calculate semantic coherence scores using sentence transformer."""
        with torch.no_grad():
            candidate_embeddings = self.encode_sequence(candidates)
            context_embedding = self.encode_sequence(context)
            
            similarity = F.cosine_similarity(
                candidate_embeddings.unsqueeze(1),
                context_embedding.unsqueeze(0),
                dim=-1
            )
            
            return similarity

    def coherence_based_sampling(self, 
                               candidates: torch.Tensor,
                               coherence_scores: torch.Tensor,
                               bert_scores: torch.Tensor) -> torch.Tensor:
        """Sample tokens based on combined BERT and coherence scores."""
        combined_scores = bert_scores * coherence_scores
        
        mean_coherence = torch.mean(coherence_scores)
        temperature = self.config.base_temperature * (1 / mean_coherence)
        temperature = torch.clamp(temperature, min=0.1, max=2.0)
        
        threshold = torch.mean(combined_scores) * self.config.min_threshold
        valid_candidates = combined_scores > threshold
        
        if not torch.any(valid_candidates):
            valid_candidates = torch.ones_like(combined_scores, dtype=torch.bool)
        
        scaled_scores = combined_scores[valid_candidates] / temperature
        probs = F.softmax(scaled_scores, dim=-1)
        
        sample_idx = torch.multinomial(probs, num_samples=1)
        valid_indices = torch.where(valid_candidates)[0]
        selected_idx = valid_indices[sample_idx]
        
        return selected_idx.squeeze()

    def generate_stream(self, initial_text: str, num_tokens: int) -> Iterator[str]:
        """Streaming version of generate that yields tokens as they're generated."""
        tokens = self.tokenizer(initial_text, return_tensors="pt", 
                            add_special_tokens=True)
        input_ids = tokens['input_ids'].to(self.config.device)
        attention_mask = tokens['attention_mask'].to(self.config.device)
        
        matrix, _ = self.create_4d_matrix(initial_text, self.tokenizer.mask_token_id)
        
        seq_len = input_ids.size(1)
        padded_input_ids = torch.full(
            (1, seq_len + num_tokens),
            self.tokenizer.pad_token_id,
            device=self.config.device
        )
        padded_input_ids[0, :seq_len] = input_ids[0]
        
        padded_attention_mask = torch.ones(
            (1, seq_len + num_tokens),
            device=self.config.device
        )
        padded_attention_mask[0, :seq_len] = attention_mask[0]
        
        masked_input_ids = padded_input_ids.clone()
        mask_positions = []
        
        for i in range(num_tokens):
            pos = seq_len + i
            masked_input_ids[0, pos] = self.tokenizer.mask_token_id
            mask_positions.append(pos)
        
        prev_layer = None
        current_sequence = input_ids[0].tolist()
        prev_token = None
        
        for position in mask_positions:
            bert_predictions = self.get_bert_predictions(masked_input_ids, [position])
            top_k_logits, top_k_indices = bert_predictions[0]
            
            candidate_tokens = [self.tokenizer.convert_ids_to_tokens([idx.item()])[0] 
                            for idx in top_k_indices]
            context = self.tokenizer.decode(current_sequence)
            
            coherence_scores = self.calculate_coherence_scores(candidate_tokens, context)
            
            selected_idx = self.coherence_based_sampling(
                top_k_indices,
                coherence_scores,
                F.softmax(top_k_logits, dim=-1)
            )
            
            selected_token_id = top_k_indices[selected_idx]
            masked_input_ids[0, position] = selected_token_id
            current_sequence.append(selected_token_id.item())
            
            with torch.no_grad():
                outputs = self.bert_model.bert(
                    masked_input_ids,
                    attention_mask=padded_attention_mask
                )
                embeddings = outputs.last_hidden_state
                matrix[position, 0, 0, :] = embeddings[0, position]
            
            prev_layer = matrix[max(0, position-self.config.context_window):position, 0, 0, :]
            
            token = self.tokenizer.convert_ids_to_tokens([selected_token_id.item()])[0]
            
            # Improved spacing logic using BERT's tokenization
            if not token.startswith('##'):
                if prev_token and not any(p in prev_token for p in '.,!?;:")}]') and position > seq_len:
                    if not any(p in token for p in '.,!?;:"({['):
                        yield ' '
            
            yield token.replace('##', '')
            prev_token = token.replace('##', '')

    def generate(self, initial_text: str, num_tokens: int) -> str:
        """Non-streaming generation function that returns complete text."""
        return "".join(self.generate_stream(initial_text, num_tokens))

def main():
    config = GenerationConfig(
        max_length=100,
        batch_size=1,
        num_candidates=50,
        embedding_dim=768,
        context_window=5,
        base_temperature=1.0,
        min_threshold=0.5,
        top_k=50
    )
    
    generator = ParallelBERTGenerator(config)
    
    initial_text = "The quick brown fox"
    num_tokens_to_generate = 10
    
    print(f"Input: {initial_text}")
    print("Generated: ", end="", flush=True)
    
    for token in generator.generate_stream(initial_text, num_tokens_to_generate):
        sys.stdout.write(token)
        sys.stdout.flush()
    print()

if __name__ == "__main__":
    main()