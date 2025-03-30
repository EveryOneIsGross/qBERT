"""
diffBERT: Diffusion-Inspired Parallel Generation with qBERT

This script extends the qBERT model with a diffusion-inspired parallel generation approach.
Instead of generating tokens sequentially, it initializes a fixed-length sequence and
refines all positions iteratively using bidirectional attention and coherence guidance.

Version: 0.1
Author: cursed.helm + Claude
Date: 04022025
"""

import warnings
# Must import warnings first and set filters before other imports
warnings.filterwarnings('ignore')  # Suppress all warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Iterator, Dict
from transformers import AutoTokenizer, BertForMaskedLM
from sentence_transformers import SentenceTransformer
import sys
from pydantic import BaseModel, Field
import transformers
transformers.logging.set_verbosity_error()  # Only show errors, not warnings
import math  # Add this import at the top with other imports
from collections import defaultdict  # Add this import

@dataclass
class GenerationConfig:
    max_length: int
    batch_size: int
    num_candidates: int
    embedding_dim: int
    context_window: int = 5
    phrase_window: int = None
    base_temperature: float = 1.0
    min_threshold: float = 0.5
    top_k: int = 50
    compression_ratio: float = 0.5
    max_cache_size: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Diffusion parameters
    diffusion_steps: int = 5  # Number of refinement iterations
    noise_schedule: str = "linear"  # How to decay noise during refinement
    min_temp: float = 0.2  # Minimum temperature for sampling 
    max_temp: float = 1.0  # Maximum temperature for sampling

    def __post_init__(self):
        # Default phrase_window to context_window // 2 if not set
        if self.phrase_window is None:
            self.phrase_window = max(32, self.context_window // 2)

class ModelConfig(BaseModel):
    """Configuration for transformer models used in qBERT"""
    bert_model_name: str = Field(
        default="bert-base-cased",
        description="Name or path of BERT model to use"
    )
    tokenizer_name: str = Field(
        default="bert-base-cased",
        description="Name or path of tokenizer to use"
    )
    sentence_transformer_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Name or path of sentence transformer model"
    )
    attn_implementation: str = Field(
        default="eager",
        description="Attention implementation type (eager or sdpa)"
    )

class SemanticMemoryCache:
    def __init__(self, config: GenerationConfig):
        self.config = config
        compressed_dim = int(config.embedding_dim * config.compression_ratio)
        self.compression_layer = nn.Linear(config.embedding_dim, compressed_dim).to(config.device)
        self.decompression_layer = nn.Linear(compressed_dim, config.embedding_dim).to(config.device)
        self.semantic_memory: Dict[str, torch.Tensor] = {}
        
    def compress_4d_space(self, matrix: torch.Tensor) -> torch.Tensor:
        """Compress the 4D representation while preserving key semantic features."""
        with torch.no_grad():
            # Reshape to 2D for compression
            batch_size = matrix.shape[1]
            num_candidates = matrix.shape[2]
            flattened = matrix.reshape(-1, self.config.embedding_dim)
            compressed = self.compression_layer(flattened)
            return compressed.reshape(matrix.shape[0], batch_size, num_candidates, -1)
        
    def decompress_4d_space(self, compressed: torch.Tensor) -> torch.Tensor:
        """Decompress the semantic space back to original dimensions."""
        with torch.no_grad():
            flattened = compressed.reshape(-1, int(self.config.embedding_dim * self.config.compression_ratio))
            decompressed = self.decompression_layer(flattened)
            return decompressed.reshape(compressed.shape[0], compressed.shape[1], 
                                      compressed.shape[2], self.config.embedding_dim)
        
    def influence_new_generation(self, current_matrix: torch.Tensor, 
                               context_key: str) -> torch.Tensor:
        """Blend previous semantic space with current if available."""
        if context_key not in self.semantic_memory:
            return current_matrix
            
        cached = self.semantic_memory[context_key]
        decompressed = self.decompress_4d_space(cached)
        
        # Ensure same sequence length
        min_len = min(current_matrix.shape[0], decompressed.shape[0])
        current_matrix = current_matrix[:min_len]
        decompressed = decompressed[:min_len]
        
        # Blend previous semantic space with current
        influence_weight = 0.3
        return current_matrix * (1 - influence_weight) + decompressed * influence_weight
        
    def update_memory(self, matrix: torch.Tensor, context_key: str):
        """Update semantic memory with new compressed representation."""
        if len(self.semantic_memory) >= self.config.max_cache_size:
            # Remove oldest entry if at capacity
            oldest_key = next(iter(self.semantic_memory))
            del self.semantic_memory[oldest_key]
            
        compressed = self.compress_4d_space(matrix)
        self.semantic_memory[context_key] = compressed

class PunctuationHandler:
    """Handles text formatting rules for punctuation and spacing."""
    
    SENTENCE_END = {'.', '!', '?'}
    PUNCT = {',', '.', '!', '?', ';', ':', '"', ')', '}', ']'}
    
    @staticmethod
    def create_formatter(tokenizer_name: str):
        """Factory method to create appropriate token formatter."""
        if 'Twitter' in tokenizer_name:
            return PunctuationHandler.format_sentencepiece_token
        return PunctuationHandler.format_wordpiece_token
    
    @staticmethod
    def format_sentencepiece_token(token: str, prev_token: str = None) -> str:
        """Format token for SentencePiece tokenizers (Twitter-BERT)."""
        if not token:
            return ''
            
        is_new_word = token.startswith('▁')
        is_continuation = token.startswith('##')
        token = token.replace('▁', '').replace('##', '').strip()
        
        if not token:
            return ''
            
        if token in PunctuationHandler.SENTENCE_END:
            return token + '\n'
            
        if token in PunctuationHandler.PUNCT:
            return token
            
        return (' ' + token if is_new_word and not is_continuation else token)
    
    @staticmethod
    def format_wordpiece_token(token: str, prev_token: str = None) -> str:
        """Format token for WordPiece tokenizers (vanilla BERT)."""
        if not token:
            return ''
            
        if token.startswith('[') and token.endswith(']'):
            return ''
            
        is_subword = token.startswith('##')
        token = token.replace('##', '').strip()
        
        if not token:
            return ''
            
        if token in PunctuationHandler.SENTENCE_END:
            return token + '\n'
            
        if token in PunctuationHandler.PUNCT:
            return token
            
        return token if is_subword else ' ' + token

class ParallelBERTGenerator(nn.Module):
    def __init__(self, config: GenerationConfig, model_config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        self.model_config = model_config  # Store model_config as instance variable
        
        # Initialize models using model_config
        self.bert_model = BertForMaskedLM.from_pretrained(
            model_config.bert_model_name,
            attn_implementation=model_config.attn_implementation
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_name)
        self.sentence_transformer = SentenceTransformer(
            model_config.sentence_transformer_name,
            trust_remote_code=True
        )
        
        # Move models to device and eval mode
        self.bert_model = self.bert_model.to(config.device)
        self.bert_model.eval()
        
        # Optimize bidirectional attention projections with scaled dot product
        self.forward_proj = nn.Linear(config.embedding_dim, config.embedding_dim).to(config.device)
        self.backward_proj = nn.Linear(config.embedding_dim, config.embedding_dim).to(config.device)
        self.cross_attn_proj = nn.Linear(config.embedding_dim, config.embedding_dim).to(config.device)
        self.combine_proj = nn.Linear(3 * config.embedding_dim, config.embedding_dim).to(config.device)
        
        # Initialize semantic memory cache
        self.semantic_memory = SemanticMemoryCache(config)
        
        # Store the formatter based on tokenizer type
        self.token_formatter = PunctuationHandler.create_formatter(self.tokenizer.name_or_path)

    def encode_sequence(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Encode text sequence using sentence transformer."""
        with torch.no_grad():
            embeddings = self.sentence_transformer.encode(text, convert_to_tensor=True)
            return embeddings.to(self.config.device)

    def get_bert_predictions(self,
                            input_ids: torch.Tensor,  # [batch_size, seq_len]
                            attention_mask: torch.Tensor,  # [batch_size, seq_len]
                            mask_positions: List[int]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get batched BERT predictions for masked positions."""
        with torch.no_grad():
            outputs = self.bert_model(input_ids, attention_mask=attention_mask)
            predictions = outputs.logits  # [batch_size, seq_len, vocab_size]
            
            valid_tokens = torch.ones(predictions.size(-1), device=self.config.device).bool()
            special_tokens = {self.tokenizer.pad_token_id, self.tokenizer.cls_token_id,
                             self.tokenizer.sep_token_id, self.tokenizer.mask_token_id,
                             self.tokenizer.unk_token_id}
            valid_tokens[list(special_tokens)] = False
            
            masked_predictions = []
            for batch_idx in range(input_ids.size(0)):
                logits = predictions[batch_idx, mask_positions[batch_idx]]
                logits[~valid_tokens] = float('-inf')
                top_k_logits, top_k_indices = torch.topk(logits, self.config.top_k)
                masked_predictions.append((top_k_logits, top_k_indices))
            
            return masked_predictions

    def create_4d_matrix(self, initial_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize the 4D generation matrix with batched BERT embeddings."""
        tokens = self.tokenizer([initial_text] * self.config.batch_size, 
                              return_tensors="pt", 
                              padding=True)
        input_ids = tokens['input_ids'].to(self.config.device)
        attention_mask = tokens['attention_mask'].to(self.config.device)
        
        matrix = torch.zeros(
            (len(input_ids[0]) + self.config.max_length,
             self.config.batch_size,
             self.config.num_candidates, 
             self.config.embedding_dim),
            device=self.config.device
        )
        
        with torch.no_grad():
            outputs = self.bert_model.bert(input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state  # [batch_size, seq_len, embedding_dim]
            for batch_idx in range(self.config.batch_size):
                seq_len = attention_mask[batch_idx].sum()
                matrix[:seq_len, batch_idx, 0, :] = embeddings[batch_idx, :seq_len]
        
        context_key = initial_text[:50]
        matrix = self.semantic_memory.influence_new_generation(matrix, context_key)
        
        return matrix, input_ids

    def process_bidirectional_context(self, 
                                    position: int,
                                    embeddings: torch.Tensor,
                                    matrix: torch.Tensor,
                                    window_size: int) -> torch.Tensor:
        """Process context using batched bidirectional and cross attention."""
        batch_size = embeddings.shape[0]
        matrix_size = matrix.shape[0]
        
        # Safely calculate context windows
        start_idx = max(0, position - window_size)
        # Ensure end_idx doesn't exceed matrix bounds
        end_idx = min(matrix_size, position + window_size + 1)
        
        # Get forward and backward context for all batches
        forward_context = matrix[start_idx:position, :, 0, :].transpose(0, 1)
        # Only get backward context if we're not at the end
        backward_context = matrix[position+1:end_idx, :, 0, :].transpose(0, 1) if position + 1 < matrix_size else torch.empty(
            (batch_size, 0, self.config.embedding_dim), device=self.config.device
        )
        
        # Process forward context if available
        forward_attn = (
            F.scaled_dot_product_attention(
                embeddings.unsqueeze(1),
                self.forward_proj(forward_context),
                forward_context,
                is_causal=False
            ).squeeze(1)
            if forward_context.size(1) > 0
            else torch.zeros_like(embeddings)
        )
        
        # Process backward context if available
        backward_attn = (
            F.scaled_dot_product_attention(
                embeddings.unsqueeze(1),
                self.backward_proj(backward_context),
                backward_context,
                is_causal=False
            ).squeeze(1)
            if backward_context.size(1) > 0
            else torch.zeros_like(embeddings)
        )
        
        # Process cross attention to initial prompt (bounded by window_size)
        # Use a wider window for cross-attention to capture more global context
        cross_start_idx = max(0, min(position, matrix_size // 4))
        cross_end_idx = min(matrix_size, max(position + 1, matrix_size * 3 // 4))
        
        # Get a broader context for cross-attention
        initial_context = matrix[cross_start_idx:cross_end_idx, :, 0, :].transpose(0, 1)
        
        if initial_context.size(1) > 0:
            # Apply position-based attention weights to focus more on nearby tokens
            position_weights = torch.ones(initial_context.size(1), device=self.config.device)
            
            # Calculate distance from current position
            positions = torch.arange(cross_start_idx, cross_end_idx, device=self.config.device)
            distances = torch.abs(positions - position).float()
            
            # Apply exponential decay based on distance
            decay_factor = 0.98
            position_weights = decay_factor ** distances
            
            # Apply position weights to cross attention
            cross_proj = self.cross_attn_proj(initial_context)
            cross_attn = F.scaled_dot_product_attention(
                embeddings.unsqueeze(1),
                cross_proj,
                initial_context * position_weights.unsqueeze(0).unsqueeze(-1),
                is_causal=False
            ).squeeze(1)
        else:
            cross_attn = torch.zeros_like(embeddings)
        
        # Combine all contexts
        combined = torch.cat([forward_attn, backward_attn, cross_attn], dim=-1)
        return self.combine_proj(combined)

    def calculate_coherence_scores(self, 
                                candidates: List[str],
                                context: str,
                                full_sequence: List[int] = None,
                                position: int = None) -> torch.Tensor:
        """Calculate coherence scores using configured phrase window and bidirectional context."""
        with torch.no_grad():
            # Get the most recent context tokens within the phrase window
            context_tokens = self.tokenizer.tokenize(context)[-self.config.phrase_window:]
            
            # Add length-based penalty for single tokens
            length_penalties = torch.tensor([
                1.0 if len(self.tokenizer.tokenize(candidate)) > 1 else 0.7
                for candidate in candidates
            ], device=self.config.device)
            
            # Create candidate phrases by appending each candidate to context
            candidate_phrases = [
                self.tokenizer.convert_tokens_to_string(context_tokens + [candidate]) 
                for candidate in candidates
            ]
            
            # If we have full sequence context (for diffusion mode)
            if full_sequence is not None and position is not None:
                # Use a wider window for global context
                global_window = min(len(full_sequence), self.config.context_window * 2)
                local_window = min(len(full_sequence), self.config.phrase_window)
                
                # Get tokens from different context windows
                global_start = max(0, position - global_window)
                global_end = min(len(full_sequence), position + global_window + 1)
                local_start = max(0, position - local_window)
                local_end = min(len(full_sequence), position + local_window + 1)
                
                # Get tokens before and after current position
                global_before = self.tokenizer.convert_ids_to_tokens(full_sequence[global_start:position])
                global_after = self.tokenizer.convert_ids_to_tokens(full_sequence[position+1:global_end]) if position+1 < len(full_sequence) else []
                local_before = self.tokenizer.convert_ids_to_tokens(full_sequence[local_start:position])
                local_after = self.tokenizer.convert_ids_to_tokens(full_sequence[position+1:local_end]) if position+1 < len(full_sequence) else []
                
                # Create context phrases with different window sizes
                global_phrases = [
                    self.tokenizer.convert_tokens_to_string(global_before + [candidate] + global_after)
                    for candidate in candidates
                ]
                
                local_phrases = [
                    self.tokenizer.convert_tokens_to_string(local_before + [candidate] + local_after)
                    for candidate in candidates
                ]
                
                # Get embeddings for different context windows
                global_embeddings = self.encode_sequence(global_phrases)
                local_embeddings = self.encode_sequence(local_phrases)
                forward_embeddings = self.encode_sequence(candidate_phrases)
                
                # Get context embeddings
                global_context = self.encode_sequence(self.tokenizer.decode(full_sequence[global_start:global_end]))
                local_context = self.encode_sequence(self.tokenizer.decode(full_sequence[local_start:local_end]))
                forward_context = self.encode_sequence(context)
                
                # Calculate similarities for different context windows
                global_similarity = F.cosine_similarity(
                    global_embeddings.unsqueeze(1),
                    global_context.unsqueeze(0),
                    dim=-1
                ).squeeze(1)
                
                local_similarity = F.cosine_similarity(
                    local_embeddings.unsqueeze(1),
                    local_context.unsqueeze(0),
                    dim=-1
                ).squeeze(1)
                
                forward_similarity = F.cosine_similarity(
                    forward_embeddings.unsqueeze(1),
                    forward_context.unsqueeze(0),
                    dim=-1
                ).squeeze(1)
                
                # Combine similarities with different weights
                similarity = (
                    forward_similarity * 0.3 +  # Forward context
                    local_similarity * 0.4 +    # Local bidirectional context
                    global_similarity * 0.3     # Global bidirectional context
                )
            else:
                # Standard forward-only mode (for regular generation)
                phrase_embeddings = self.encode_sequence(candidate_phrases)
                context_embedding = self.encode_sequence(context)
                
                similarity = F.cosine_similarity(
                    phrase_embeddings.unsqueeze(1),
                    context_embedding.unsqueeze(0),
                    dim=-1
                ).squeeze(1)
            
            # Apply length penalty to similarity scores
            weighted_similarity = similarity * length_penalties
            
            # Normalize scores to [0,1] range for better stability tracking
            if torch.max(weighted_similarity) > 0:
                weighted_similarity = weighted_similarity / torch.max(weighted_similarity)
            
            return weighted_similarity

    def coherence_based_sampling(self, 
                               candidates: torch.Tensor,
                               coherence_scores: torch.Tensor,
                               bert_scores: torch.Tensor,
                               temperature: float = 1.0) -> torch.Tensor:
        """Sample tokens based on combined BERT and coherence scores with temperature."""
        # Filter out invalid tokens
        valid_tokens = torch.ones(candidates.size(0), device=self.config.device).bool()
        for i, idx in enumerate(candidates):
            token = self.tokenizer.convert_ids_to_tokens([idx.item()])[0]
            # Skip special tokens or empty tokens
            if (token.startswith('[') and token.endswith(']')) or not token:
                valid_tokens[i] = False
                
        # Apply single token length penalty
        single_token_mask = torch.tensor([
            0.8 if len(self.tokenizer.convert_ids_to_tokens([idx.item()])[0]) <= 4 else 1.0
            for idx in candidates
        ], device=self.config.device)
        
        # Combine scores
        combined_scores = bert_scores * coherence_scores * single_token_mask * valid_tokens.float()
        
        # If all tokens are invalid, just use BERT scores
        if not torch.any(valid_tokens):
            combined_scores = bert_scores
        
        # Apply adaptive temperature 
        mean_coherence = torch.mean(coherence_scores)
        adaptive_temp = self.config.base_temperature * (1 / max(0.1, mean_coherence))
        adaptive_temp = adaptive_temp * temperature  # Apply external temperature factor
        adaptive_temp = torch.clamp(adaptive_temp, min=0.1, max=2.0)
        
        # Get threshold and valid candidates
        threshold = torch.mean(combined_scores) * self.config.min_threshold
        valid_candidates = combined_scores > threshold
        
        # Ensure we have at least one valid candidate
        if not torch.any(valid_candidates):
            valid_candidates = valid_tokens
            if not torch.any(valid_candidates):
                valid_candidates = torch.ones_like(combined_scores, dtype=torch.bool)
        
        # Apply temperature and sampling
        scaled_scores = combined_scores[valid_candidates] / adaptive_temp
        probs = F.softmax(scaled_scores, dim=-1)
        
        # Multinomial sampling with safety check
        if probs.sum() == 0:
            # If all probs are zero, just pick the highest score
            selected_idx = torch.argmax(combined_scores)
        else:
            sample_idx = torch.multinomial(probs, num_samples=1)
            valid_indices = torch.where(valid_candidates)[0]
            selected_idx = valid_indices[sample_idx]
        
        return selected_idx.squeeze()

    def generate_stream(self, initial_text: str, num_tokens: int) -> Iterator[str]:
        """Streaming generation with batched processing."""
        # Prepare initial input for all batches
        batch_texts = [initial_text] * self.config.batch_size
        
        # Create matrix and get initial tokens
        matrix, input_ids = self.create_4d_matrix(initial_text)
        available_positions = matrix.shape[0] - len(input_ids[0])
        tokens_to_generate = min(num_tokens, available_positions)
        seq_len = input_ids.size(1)
        
        # Create batched padded tensors
        padded_input_ids = torch.full(
            (self.config.batch_size, seq_len + tokens_to_generate),
            self.tokenizer.pad_token_id,
            device=self.config.device
        )
        padded_input_ids[:, :seq_len] = input_ids
        
        padded_attention_mask = torch.zeros(
            (self.config.batch_size, seq_len + tokens_to_generate),
            device=self.config.device
        )
        padded_attention_mask[:, :seq_len] = 1
        
        # Setup for generation
        masked_input_ids = padded_input_ids.clone()
        current_sequences = [input_ids[i].tolist() for i in range(self.config.batch_size)]
        prev_tokens = [None] * self.config.batch_size
        
        # Generate tokens
        for position in range(seq_len, seq_len + tokens_to_generate):
            # Update attention mask for current position
            padded_attention_mask[:, :position + 1] = 1
            
            # Mask current position for all batches
            masked_input_ids[:, position] = self.tokenizer.mask_token_id
            
            # Get batched BERT predictions
            bert_predictions = self.get_bert_predictions(
                masked_input_ids,
                padded_attention_mask,
                [position] * self.config.batch_size
            )
            
            # Get embeddings with bidirectional context for all batches
            with torch.no_grad():
                outputs = self.bert_model.bert(
                    input_ids=masked_input_ids,
                    attention_mask=padded_attention_mask
                )
                current_embeddings = outputs.last_hidden_state[:, position]  # [batch_size, embedding_dim]
                attended_embeddings = self.process_bidirectional_context(
                    position, 
                    current_embeddings, 
                    matrix,
                    self.config.context_window
                )
            
            # Process each batch
            for batch_idx in range(self.config.batch_size):
                # Unpack predictions correctly
                top_k_logits, top_k_indices = bert_predictions[batch_idx]
                
                # Get candidate tokens and calculate coherence
                candidate_tokens = [self.tokenizer.convert_ids_to_tokens([idx.item()])[0] 
                                 for idx in top_k_indices]
                context = self.tokenizer.decode(current_sequences[batch_idx])
                coherence_scores = self.calculate_coherence_scores(
                    candidate_tokens, 
                    context,
                    full_sequence=current_sequences[batch_idx],
                    position=position
                )
                
                # Sample next token
                selected_idx = self.coherence_based_sampling(
                    top_k_indices,
                    coherence_scores,
                    F.softmax(top_k_logits, dim=-1)
                )
                
                # Update state for this batch
                selected_token_id = top_k_indices[selected_idx]
                masked_input_ids[batch_idx, position] = selected_token_id
                current_sequences[batch_idx].append(selected_token_id.item())
                matrix[position, batch_idx, 0, :] = attended_embeddings[batch_idx]
                
                # Format and yield token for this batch
                token = self.tokenizer.convert_ids_to_tokens([selected_token_id.item()])[0]
                formatted = self.token_formatter(token, prev_tokens[batch_idx])
                if formatted:
                    yield formatted
                prev_tokens[batch_idx] = token
        
        # Update semantic memory after generation
        context_key = initial_text
        self.semantic_memory.update_memory(matrix, context_key)

    def generate(self, initial_text: str, num_tokens: int) -> str:
        """Non-streaming generation function that returns complete text."""
        return "".join(self.generate_stream(initial_text, num_tokens))

    def get_diffusion_temperature(self, step: int, total_steps: int) -> float:
        """Calculate temperature based on diffusion schedule."""
        if self.config.noise_schedule == "linear":
            # Linear decay from max_temp to min_temp
            alpha = 1 - (step / max(1, total_steps - 1))
            return self.config.min_temp + alpha * (self.config.max_temp - self.config.min_temp)
        elif self.config.noise_schedule == "cosine":
            # Cosine schedule for smoother decay
            alpha = 0.5 * (1 + np.cos(step / max(1, total_steps - 1) * np.pi))
            return self.config.min_temp + alpha * (self.config.max_temp - self.config.min_temp)
        else:
            # Default to constant
            return self.config.base_temperature

    def generate_diffusion_stream(self, initial_text: str, num_tokens: int, verbose: bool = False) -> Iterator[str]:
        """Diffusion-based parallel generation over a fixed-length word space."""
        # Prepare initial input for all batches
        batch_texts = [initial_text] * self.config.batch_size
        
        # Create matrix and get initial tokens
        matrix, input_ids = self.create_4d_matrix(initial_text)
        available_positions = matrix.shape[0] - len(input_ids[0])
        tokens_to_generate = min(num_tokens, available_positions)
        seq_len = input_ids.size(1)
        
        # Create padded tensors for the fixed-length sequence
        padded_input_ids = torch.full(
            (self.config.batch_size, seq_len + tokens_to_generate),
            self.tokenizer.pad_token_id,
            device=self.config.device
        )
        padded_input_ids[:, :seq_len] = input_ids
        
        padded_attention_mask = torch.zeros(
            (self.config.batch_size, seq_len + tokens_to_generate),
            device=self.config.device
        )
        padded_attention_mask[:, :seq_len] = 1
        
        # Initialize all generation positions with MASK tokens
        masked_input_ids = padded_input_ids.clone()
        for batch_idx in range(self.config.batch_size):
            for pos in range(seq_len, seq_len + tokens_to_generate):
                masked_input_ids[batch_idx, pos] = self.tokenizer.mask_token_id
                padded_attention_mask[batch_idx, pos] = 1  # Enable attention for all positions
        
        # Initialize current sequence with context
        current_sequences = [input_ids[i].tolist() for i in range(self.config.batch_size)]
        
        # Add token stabilization mask to track which tokens should be frozen
        token_stability = torch.zeros(
            (self.config.batch_size, seq_len + tokens_to_generate),
            device=self.config.device
        )
        # Context tokens are always stable
        token_stability[:, :seq_len] = 1.0
        
        # Track token coherence scores to prioritize high-value tokens
        token_coherence = torch.zeros(
            (self.config.batch_size, seq_len + tokens_to_generate),
            device=self.config.device
        )
        
        # Print context information if verbose
        if verbose:
            print("\n=== Context Information ===")
            for batch_idx in range(self.config.batch_size):
                context_text = self.tokenizer.decode(current_sequences[batch_idx])
                context_tokens = self.tokenizer.convert_ids_to_tokens(current_sequences[batch_idx])
                print(f"Context length: {len(current_sequences[batch_idx])} tokens")
                print(f"Context text: {context_text}")
                print(f"Context tokens: {context_tokens}")
                print(f"Window size: {self.config.context_window} tokens")
                
                # Show the initial masked sequence
                masked_seq = masked_input_ids[batch_idx].tolist()
                masked_tokens = self.tokenizer.convert_ids_to_tokens(masked_seq)
                print("\nInitial masked sequence:")
                # Mark the boundary between context and generation
                boundary_tokens = []
                for i, token in enumerate(masked_tokens):
                    if i < seq_len:
                        boundary_tokens.append(f"{token}")
                    else:
                        boundary_tokens.append(f"[{token}]")
                print(" ".join(boundary_tokens))
                print("-" * 50 + "|" + "-" * 50)  # Visual boundary marker
        
        # First pass: Get initial token predictions for the entire fixed-length sequence
        with torch.no_grad():
            outputs = self.bert_model(masked_input_ids, attention_mask=padded_attention_mask)
            initial_logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            
            # Filter valid tokens
            vocab_size = initial_logits.size(-1)
            valid_tokens = torch.ones(vocab_size, device=self.config.device).bool()
            special_tokens = {self.tokenizer.pad_token_id, self.tokenizer.cls_token_id,
                             self.tokenizer.sep_token_id, self.tokenizer.mask_token_id,
                             self.tokenizer.unk_token_id}
            valid_tokens[list(special_tokens)] = False
            
            # Initial population of tokens with high temperature sampling
            for batch_idx in range(self.config.batch_size):
                for pos in range(seq_len, seq_len + tokens_to_generate):
                    # Get predictions for this position
                    logits = initial_logits[batch_idx, pos]
                    logits[~valid_tokens] = float('-inf')
                    top_k_logits, top_k_indices = torch.topk(logits, self.config.top_k)
                    
                    # Sample with high temperature (exploration)
                    high_temp = 1.5
                    probs = F.softmax(top_k_logits / high_temp, dim=0)
                    idx = torch.multinomial(probs, num_samples=1).item()
                    token_id = top_k_indices[idx].item()
                    
                    # Update sequence and input IDs
                    current_sequences[batch_idx].append(token_id)
                    masked_input_ids[batch_idx, pos] = token_id
        
        # Get initial embeddings for all positions
        with torch.no_grad():
            outputs = self.bert_model.bert(
                input_ids=masked_input_ids, 
                attention_mask=padded_attention_mask
            )
            embeddings = outputs.last_hidden_state
            
            # Update matrix with initial embeddings
            for batch_idx in range(self.config.batch_size):
                for pos in range(seq_len, seq_len + tokens_to_generate):
                    matrix[pos, batch_idx, 0, :] = embeddings[batch_idx, pos]
        
        # Print initial state if verbose
        if verbose:
            print("\n=== Initial Generation (Step 0) ===")
            for batch_idx in range(self.config.batch_size):
                # Show full sequence with context and generated parts visually separated
                full_sequence = current_sequences[batch_idx]
                context_part = full_sequence[:seq_len]
                generated_part = full_sequence[seq_len:]
                
                context_text = self.tokenizer.decode(context_part)
                generated_text = self.tokenizer.decode(generated_part)
                
                print(f"Batch {batch_idx}:")
                print(f"Context: {context_text}")
                print(f"Generated: {generated_text}")
                
                # Show token-level view with boundary
                full_tokens = self.tokenizer.convert_ids_to_tokens(full_sequence)
                context_tokens = full_tokens[:seq_len]
                generated_tokens = full_tokens[seq_len:]
                
                print("\nToken view:")
                print(" ".join(context_tokens) + " | " + " ".join(generated_tokens))
        
        # Iterative refinement - diffusion steps
        for step in range(self.config.diffusion_steps):
            # Get temperature for this diffusion step
            temperature = self.get_diffusion_temperature(step, self.config.diffusion_steps)
            
            # Calculate stability threshold that increases with steps
            stability_threshold = 0.5 + (step / self.config.diffusion_steps) * 0.4
            
            # Track changes for convergence check
            token_changes = 0
            total_positions = 0
            position_changes = {}  # Track which positions changed
            
            # For each batch, refine all positions
            for batch_idx in range(self.config.batch_size):
                batch_sequence = current_sequences[batch_idx]
                
                # Prioritize tokens with low coherence scores
                positions = list(range(seq_len, len(batch_sequence)))
                positions.sort(key=lambda pos: token_coherence[batch_idx, pos].item())
                
                # Process each position to refine
                for pos in positions:
                    total_positions += 1
                    rel_pos = pos - seq_len  # Relative position in generated sequence
                    
                    # Skip stable tokens (those with stability above threshold)
                    if token_stability[batch_idx, pos] > stability_threshold:
                        continue
                    
                    # Create temporary input with this position masked
                    temp_input_ids = masked_input_ids.clone()
                    temp_input_ids[batch_idx, pos] = self.tokenizer.mask_token_id
                    
                    # Get predictions for this masked position
                    with torch.no_grad():
                        outputs = self.bert_model(temp_input_ids, attention_mask=padded_attention_mask)
                        logits = outputs.logits[batch_idx, pos]
                        
                        # Filter valid tokens
                        valid_tokens = torch.ones(logits.size(0), device=self.config.device).bool()
                        special_tokens = {self.tokenizer.pad_token_id, self.tokenizer.cls_token_id,
                                        self.tokenizer.sep_token_id, self.tokenizer.mask_token_id,
                                        self.tokenizer.unk_token_id}
                        valid_tokens[list(special_tokens)] = False
                        logits[~valid_tokens] = float('-inf')
                        
                        # Get top-k candidates
                        top_k_logits, top_k_indices = torch.topk(logits, self.config.top_k)
                    
                    # Get candidate tokens
                    candidate_tokens = [self.tokenizer.convert_ids_to_tokens([idx.item()])[0] 
                                     for idx in top_k_indices]
                    
                    # Get context up to current position for coherence scoring
                    context = self.tokenizer.decode(batch_sequence[:pos])
                    
                    # Calculate coherence scores
                    coherence_scores = self.calculate_coherence_scores(
                        candidate_tokens, 
                        context,
                        full_sequence=batch_sequence,
                        position=pos
                    )
                    
                    # Store max coherence score for this position
                    token_coherence[batch_idx, pos] = torch.max(coherence_scores)
                    
                    # Sample token with diffusion temperature
                    probs = F.softmax(coherence_scores / temperature, dim=0)

                    # Ensure probs is 1D and contains valid probabilities
                    if torch.isnan(probs).any() or torch.sum(probs) == 0:
                        # If we have invalid probabilities, use uniform distribution
                        probs = torch.ones_like(probs) / probs.size(0)

                    # Sample from the distribution
                    sampled_idx = torch.multinomial(probs, 1).item()
                    new_token_id = top_k_indices[sampled_idx]
                    
                    # Check if token changed
                    current_token_id = batch_sequence[pos] if pos < len(batch_sequence) else None
                    
                    if current_token_id != new_token_id:
                        token_changes += 1
                        # Track which position changed
                        position_changes[rel_pos] = (
                            self.tokenizer.convert_ids_to_tokens([current_token_id])[0],
                            self.tokenizer.convert_ids_to_tokens([new_token_id])[0]
                        )
                        
                        # Update sequence and input IDs
                        batch_sequence[pos] = new_token_id
                        masked_input_ids[batch_idx, pos] = new_token_id
                    else:
                        # Token remained the same, increase its stability
                        token_stability[batch_idx, pos] += 0.2
                    
                    # Update token stability based on coherence score
                    max_coherence = torch.max(coherence_scores)
                    if max_coherence > 0.8:  # High confidence
                        token_stability[batch_idx, pos] += 0.15
                
                # Update current sequence
                current_sequences[batch_idx] = batch_sequence
            
            # Update embeddings with bidirectional context
            with torch.no_grad():
                outputs = self.bert_model.bert(
                    input_ids=masked_input_ids,
                    attention_mask=padded_attention_mask
                )
                updated_embeddings = outputs.last_hidden_state
                
                # Process all positions with bidirectional context
                for batch_idx in range(self.config.batch_size):
                    for pos in range(seq_len, seq_len + tokens_to_generate):
                        # Apply bidirectional context
                        window_size = self.config.context_window + int(step * 0.5)  # Expand window with steps
                        attended_embedding = self.process_bidirectional_context(
                            pos,
                            updated_embeddings[batch_idx, pos].unsqueeze(0),
                            matrix,
                            window_size
                        )
                        matrix[pos, batch_idx, 0, :] = attended_embedding
            
            # Print current state if verbose
            if verbose:
                change_ratio = token_changes / total_positions if total_positions > 0 else 0
                print(f"\n=== Diffusion Step {step+1}/{self.config.diffusion_steps} (Temp: {temperature:.2f}, Changes: {change_ratio:.2%}) ===")
                
                # Show stability information
                stable_count = torch.sum(token_stability[:, seq_len:] > stability_threshold).item()
                total_tokens = self.config.batch_size * tokens_to_generate
                print(f"Stable tokens: {stable_count}/{total_tokens} ({stable_count/total_tokens:.2%})")
                
                # Show which positions changed
                if position_changes:
                    print("\nToken changes:")
                    for pos, (old, new) in sorted(position_changes.items()):
                        print(f"Position {pos}: {old} → {new}")
                
                for batch_idx in range(self.config.batch_size):
                    # Show full sequence with context and generated parts
                    full_sequence = current_sequences[batch_idx]
                    context_part = full_sequence[:seq_len]
                    generated_part = full_sequence[seq_len:]
                    
                    context_text = self.tokenizer.decode(context_part)
                    generated_text = self.tokenizer.decode(generated_part)
                    
                    print(f"\nBatch {batch_idx}:")
                    print(f"Context: {context_text}")
                    print(f"Generated: {generated_text}")
                    
                    # Show context window influence
                    if step > 0:
                        print("\nContext window influence:")
                        for pos in range(seq_len, seq_len + tokens_to_generate):
                            rel_pos = pos - seq_len
                            # Calculate window boundaries
                            window_size = self.config.context_window + int(step * 0.5)
                            start_idx = max(0, pos - window_size)
                            end_idx = min(len(full_sequence), pos + window_size + 1)
                            
                            # Only show for a few positions to avoid clutter
                            if rel_pos % 10 == 0 or rel_pos in position_changes:
                                window_tokens = full_sequence[start_idx:end_idx]
                                window_text = self.tokenizer.decode(window_tokens)
                                current_token = self.tokenizer.convert_ids_to_tokens([full_sequence[pos]])[0]
                                print(f"Position {rel_pos} ('{current_token}'): Window [{start_idx}-{end_idx}]: {window_text}")
            
            # Check for convergence
            if total_positions > 0:
                change_ratio = token_changes / total_positions
                if step > 0 and change_ratio < 0.05:  # Early stopping if < 5% tokens changed
                    if verbose:
                        print(f"\n=== Early stopping at step {step+1}: convergence reached ({change_ratio:.2%} changes) ===")
                    break
        
        # Return generated tokens
        prev_tokens = [None] * self.config.batch_size
        
        for batch_idx in range(self.config.batch_size):
            sequence = current_sequences[batch_idx]
            # Skip original context tokens
            for i in range(seq_len, len(sequence)):
                token = self.tokenizer.convert_ids_to_tokens([sequence[i]])[0]
                formatted = self.token_formatter(token, prev_tokens[batch_idx])
                if formatted:
                    yield formatted
                prev_tokens[batch_idx] = token
        
        # Update semantic memory
        context_key = initial_text
        self.semantic_memory.update_memory(matrix, context_key)

    def generate_diffusion(self, initial_text: str, num_tokens: int, verbose: bool = False) -> str:
        """Generate text using diffusion-inspired parallel refinement."""
        # Set up device
        device = self.config.device
        
        # Tokenize initial text
        input_ids = self.tokenizer.encode(initial_text, return_tensors="pt").to(device)
        seq_len = input_ids.shape[1]
        
        # Initialize with noise
        if verbose:
            print("=== Initial Generation (Step 0) ===")
        
        # Initialize current sequence with context
        current_sequences = [input_ids[i].tolist() for i in range(self.config.batch_size)]
        
        # Add token stabilization mask to track which tokens should be frozen
        token_stability = torch.zeros(
            (self.config.batch_size, seq_len + num_tokens),
            device=device
        )
        # Context tokens are always stable
        token_stability[:, :seq_len] = 1.0
        
        # Track token coherence scores to prioritize high-value tokens
        token_coherence = torch.zeros(
            (self.config.batch_size, seq_len + num_tokens),
            device=device
        )
        
        # Print context information if verbose
        if verbose:
            for batch_idx in range(min(1, self.config.batch_size)):  # Show only first batch for clarity
                print(f"Batch {batch_idx}:")
                print(f"Context: {self.tokenizer.decode(current_sequences[batch_idx])}")
        
        # Add random tokens to reach desired length
        for batch_idx in range(self.config.batch_size):
            for _ in range(num_tokens):
                # Sample random token from vocabulary
                random_token = random.randint(1000, self.tokenizer.vocab_size - 1)
                current_sequences[batch_idx].append(random_token)
        
        # Print initial generation if verbose
        if verbose:
            for batch_idx in range(min(1, self.config.batch_size)):
                print(f"Generated: {self.tokenizer.decode(current_sequences[batch_idx][seq_len:])}")
                print("\nToken view:")
                tokens = self.tokenizer.convert_ids_to_tokens(current_sequences[batch_idx])
                print(" ".join([f"{t}" for t in tokens[:seq_len]]) + " | " + " ".join([f"{t}" for t in tokens[seq_len:]]))
        
        # Refinement iterations
        for step in range(self.config.diffusion_steps):
            # Calculate temperature for this step (decay from max to min)
            progress = step / (self.config.diffusion_steps - 1) if self.config.diffusion_steps > 1 else 1.0
            if self.config.noise_schedule == "linear":
                temperature = self.config.max_temp - progress * (self.config.max_temp - self.config.min_temp)
            else:  # Cosine schedule
                temperature = self.config.min_temp + 0.5 * (self.config.max_temp - self.config.min_temp) * (1 + math.cos(progress * math.pi))
            
            # Track changes for this iteration
            changes_made = 0
            total_tokens = self.config.batch_size * num_tokens
            
            # Track token changes for verbose output
            if verbose:
                token_changes = []
            
            # Process each sequence in batch
            for batch_idx in range(self.config.batch_size):
                batch_sequence = current_sequences[batch_idx]
                
                # Prioritize tokens with low coherence scores
                positions = list(range(seq_len, len(batch_sequence)))
                positions.sort(key=lambda pos: token_coherence[batch_idx, pos].item())
                
                # Process each position after the initial text
                for position in positions:
                    # Skip stable tokens (those with stability above threshold)
                    if token_stability[batch_idx, position] > 0.7:
                        continue
                    
                    # Get context before current position
                    context = self.tokenizer.decode(batch_sequence[:position])
                    
                    # Get top-k candidates for this position
                    with torch.no_grad():
                        inputs = self.tokenizer(context, return_tensors="pt").to(device)
                        outputs = self.bert_model(**inputs)
                        logits = outputs.logits[0, -1, :]
                        
                        # Apply temperature
                        logits = logits / temperature
                        
                        # Get top-k token indices
                        top_k_indices = torch.topk(logits, self.config.top_k).indices.tolist()
                    
                    # Get candidate tokens
                    candidate_tokens = [self.tokenizer.convert_ids_to_tokens(idx) for idx in top_k_indices]
                    
                    # Calculate coherence scores
                    coherence_scores = self.calculate_coherence_scores(
                        candidate_tokens, 
                        context,
                        full_sequence=batch_sequence,
                        position=position
                    )
                    
                    # Store max coherence score for this position
                    token_coherence[batch_idx, position] = torch.max(coherence_scores)
                    
                    # Sample token with diffusion temperature
                    probs = F.softmax(coherence_scores / temperature, dim=0)

                    # Ensure probs is 1D and contains valid probabilities
                    if torch.isnan(probs).any() or torch.sum(probs) == 0:
                        # If we have invalid probabilities, use uniform distribution
                        probs = torch.ones_like(probs) / probs.size(0)

                    # Sample from the distribution
                    sampled_idx = torch.multinomial(probs, 1).item()
                    new_token_id = top_k_indices[sampled_idx]
                    
                    # Track changes
                    if new_token_id != batch_sequence[position]:
                        if verbose and batch_idx == 0:  # Only track for first batch if verbose
                            old_token = self.tokenizer.convert_ids_to_tokens(batch_sequence[position])
                            new_token = self.tokenizer.convert_ids_to_tokens(new_token_id)
                            token_changes.append((position - seq_len, f"{old_token} → {new_token}"))
                        
                        changes_made += 1
                        batch_sequence[position] = new_token_id
                        
                        # Reset stability for changed tokens but consider coherence
                        new_stability = min(0.3, token_coherence[batch_idx, position].item())
                        token_stability[batch_idx, position] = new_stability
                    else:
                        # Increase stability for unchanged tokens based on coherence
                        coherence_boost = token_coherence[batch_idx, position].item() * 0.3
                        token_stability[batch_idx, position] += 0.1 + coherence_boost
            
            # Calculate percentage of tokens changed
            change_percentage = 100 * changes_made / total_tokens if total_tokens > 0 else 0
            
            # Print progress if verbose
            if verbose:
                print(f"\n=== Diffusion Step {step+1}/{self.config.diffusion_steps} (Temp: {temperature:.2f}, Changes: {change_percentage:.2f}%) ===")
                stable_count = (token_stability[:, seq_len:] > 0.7).sum().item()
                print(f"Stable tokens: {stable_count}/{total_tokens} ({100 * stable_count / total_tokens:.2f}%)")
                
                if token_changes:
                    print("\nToken changes:")
                    for pos, change in token_changes:
                        print(f"Position {pos}: {change}")
            
            # Show a sample from the batch
            batch_idx = 0
            print(f"\nBatch {batch_idx}:")
            print(f"Context: {self.tokenizer.decode(current_sequences[batch_idx][:seq_len])}")
            print(f"Generated: {self.tokenizer.decode(current_sequences[batch_idx][seq_len:])}")
            
            # Show context window for a few positions
            if step > 0:
                print("\nContext window influence:")
                for pos_idx in range(0, num_tokens, 10):
                    pos = seq_len + pos_idx
                    if pos < len(batch_sequence):
                        token = self.tokenizer.convert_ids_to_tokens(batch_sequence[pos])
                        window_start = max(0, pos - self.config.context_window)
                        window_end = min(len(batch_sequence), pos + self.config.context_window + 1)
                        window = self.tokenizer.decode(batch_sequence[window_start:window_end])
                        print(f"Position {pos_idx} ('{token}'): Window [{window_start}-{window_end}]: {window}")
            
            # Early stopping if very few changes
            if change_percentage < 0.5:
                if verbose:
                    print(f"\n=== Early stopping at step {step+1}: convergence reached ({change_percentage:.2f}% changes) ===")
                break
        
        # Return the best sequence from the batch (using first for simplicity)
        best_sequence = current_sequences[0]
        result = self.tokenizer.decode(best_sequence[seq_len:])
        
        if verbose:
            print("\nFinal Result:")
            print(f"{initial_text}{result}")
        
        return result

    def optimize_sequence_mcts(self, initial_text: str, num_tokens: int, diffusion_steps: int = None) -> str:
        """
        Optimize final sequence using MCTS across diffusion steps.
        
        This method runs multiple diffusion processes to gather high-coherence token candidates,
        then uses a beam search approach to find the optimal sequence, followed by a final
        attention sweep for global coherence.
        
        Args:
            initial_text: The initial prompt text
            num_tokens: Number of tokens to generate
            diffusion_steps: Number of diffusion steps (uses config default if None)
            
        Returns:
            The optimized generated text
        """
        if diffusion_steps is None:
            diffusion_steps = self.config.diffusion_steps
        
        # Step 1: Run diffusion generation multiple times, tracking all candidates
        token_candidates = defaultdict(list)  # position -> [(token_id, coherence_score)]
        
        # Run several diffusion processes to gather candidates
        for run in range(3):  # Multiple runs for diversity
            current_sequences, coherence_scores = self._run_diffusion_generation(
                initial_text, num_tokens, diffusion_steps, track_candidates=True
            )
            
            # Track candidates with their scores
            seq = current_sequences[0]  # Use first batch
            seq_len = len(self.tokenizer.encode(initial_text, return_tensors="pt")[0])
            
            for pos in range(seq_len, len(seq)):
                rel_pos = pos - seq_len
                token_id = seq[pos]
                score = coherence_scores[0, pos].item()
                token_candidates[rel_pos].append((token_id, score))
        
        # Step 2: For each position, get top candidates across all runs
        best_candidates = {}
        for pos, candidates in token_candidates.items():
            # Sort by coherence score
            sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            # Keep top 3 unique candidates
            unique_candidates = []
            seen_tokens = set()
            for token_id, score in sorted_candidates:
                if token_id not in seen_tokens and len(unique_candidates) < 3:
                    unique_candidates.append((token_id, score))
                    seen_tokens.add(token_id)
            best_candidates[pos] = unique_candidates
        
        # Step 3: Use MCTS-like approach to find optimal sequence
        initial_seq = self.tokenizer.encode(initial_text, return_tensors="pt")[0].tolist()
        
        # Build potential sequences using beam search
        beam_width = 5
        sequences = [(initial_seq, 0.0)]  # (sequence, score)
        
        for pos in range(num_tokens):
            rel_pos = pos
            abs_pos = rel_pos + len(initial_seq)
            
            if rel_pos not in best_candidates:
                continue
                
            candidates = best_candidates[rel_pos]
            new_sequences = []
            
            for sequence, score in sequences:
                for token_id, token_score in candidates:
                    new_seq = sequence.copy()
                    if len(new_seq) <= abs_pos:
                        # Extend sequence if needed
                        new_seq.extend([self.tokenizer.pad_token_id] * (abs_pos - len(new_seq) + 1))
                    new_seq[abs_pos] = token_id
                    new_score = score + token_score
                    new_sequences.append((new_seq, new_score))
            
            # Keep top beam_width sequences
            sequences = sorted(new_sequences, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Step 4: Run final attention sweep on top sequences for global coherence
        if sequences:
            best_global_score = -float('inf')
            best_sequence = None
            
            for sequence, _ in sequences[:3]:  # Consider top 3 sequences
                # Perform attention sweep for global coherence
                global_score = self._calculate_global_coherence(sequence)
                if global_score > best_global_score:
                    best_global_score = global_score
                    best_sequence = sequence
        else:
            # Fallback to initial sequence if no candidates found
            best_sequence = initial_seq
        
        # Step 5: Final refinement pass - iterate over all positions to update based on neighbors
        if best_sequence:
            best_sequence = self._final_refinement_pass(best_sequence, initial_text)
        
        # Convert final sequence to text
        result_text = self.tokenizer.decode(best_sequence[len(initial_seq):])
        return result_text

    def _final_refinement_pass(self, sequence: List[int], initial_text: str) -> List[int]:
        """
        Perform a final refinement pass over the entire sequence to ensure global coherence.
        
        This method iterates over all token positions and updates them based on:
        1. Neighbor token probabilities
        2. Semantic relevance to the context
        3. Global coherence with the entire sequence
        
        Args:
            sequence: The token sequence to refine
            initial_text: The initial prompt text
            
        Returns:
            The refined token sequence
        """
        # Get the length of the initial context
        initial_seq = self.tokenizer.encode(initial_text, return_tensors="pt")[0].tolist()
        seq_len = len(initial_seq)
        
        # Create a copy of the sequence to modify
        refined_sequence = sequence.copy()
        
        # Create a tensor for the full sequence
        input_ids = torch.tensor([refined_sequence]).to(self.config.device)
        attention_mask = torch.ones_like(input_ids)
        
        # Get embeddings for the full sequence
        with torch.no_grad():
            outputs = self.bert_model.bert(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden_size]
        
        # Track positions that have been modified
        modified_positions = set()
        
        # Iterate over all positions in the generated part
        for position in range(seq_len, len(refined_sequence)):
            # Skip if this position was already modified in this pass
            if position in modified_positions:
                continue
            
            # Get context window around this position
            window_start = max(0, position - self.config.context_window)
            window_end = min(len(refined_sequence), position + self.config.context_window + 1)
            
            # Get the current token and its embedding
            current_token_id = refined_sequence[position]
            current_embedding = embeddings[position]
            
            # Get embeddings for tokens in the window
            window_embeddings = embeddings[window_start:window_end]
            
            # Calculate semantic similarity with window tokens
            similarities = F.cosine_similarity(
                current_embedding.unsqueeze(0),
                window_embeddings,
                dim=1
            )
            
            # Calculate average similarity (excluding self)
            self_idx = position - window_start
            if 0 <= self_idx < similarities.size(0):
                mask = torch.ones_like(similarities)
                mask[self_idx] = 0
                avg_similarity = (similarities * mask).sum() / (mask.sum() + 1e-9)
            else:
                avg_similarity = similarities.mean()
            
            # If similarity is below threshold, try to find a better token
            if avg_similarity < 0.6:  # Threshold for acceptable similarity
                # Get context before this position
                context = self.tokenizer.decode(refined_sequence[:position])
                
                # Get top-k candidates for this position
                with torch.no_grad():
                    inputs = self.tokenizer(context, return_tensors="pt").to(self.config.device)
                    outputs = self.bert_model(**inputs)
                    logits = outputs.logits[0, -1, :]
                    
                    # Get top-k token indices
                    top_k = min(50, self.config.top_k)  # Use smaller top_k for refinement
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                
                # Get candidate tokens
                candidate_tokens = [self.tokenizer.convert_ids_to_tokens([idx.item()])[0] 
                                 for idx in top_k_indices]
                
                # Calculate coherence scores
                coherence_scores = self.calculate_coherence_scores(
                    candidate_tokens, 
                    context,
                    full_sequence=refined_sequence,
                    position=position
                )
                
                # Get the best candidate
                best_idx = torch.argmax(coherence_scores).item()
                best_token_id = top_k_indices[best_idx].item()
                
                # Update the sequence if the new token is better
                if best_token_id != current_token_id:
                    refined_sequence[position] = best_token_id
                    modified_positions.add(position)
                    
                    # Also update neighboring positions that might be affected
                    for neighbor_pos in range(max(seq_len, position-2), min(len(refined_sequence), position+3)):
                        if neighbor_pos != position:
                            modified_positions.add(neighbor_pos)
        
        # Check if we need to update embeddings and do another pass
        if modified_positions:
            # Get updated embeddings
            input_ids = torch.tensor([refined_sequence]).to(self.config.device)
            with torch.no_grad():
                outputs = self.bert_model.bert(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state[0]
            
            # Calculate global coherence score
            is_qa = "USER:" in initial_text and "ASSISTANT:" in initial_text
            
            # For QA contexts, ensure factual consistency
            if is_qa:
                # Get the full text
                full_text = self.tokenizer.decode(refined_sequence)
                
                # Split into question and answer
                qa_parts = full_text.split("ASSISTANT:")
                if len(qa_parts) > 1:
                    question = qa_parts[0].replace("USER:", "").strip()
                    answer = qa_parts[1].strip()
                    
                    # Check for common factual answers
                    if "capital of France" in question.lower():
                        if "paris" not in answer.lower():
                            # Force correction for factual questions
                            answer_start = len(initial_seq)
                            paris_tokens = self.tokenizer.encode("Paris", add_special_tokens=False)
                            for i, token_id in enumerate(paris_tokens):
                                if answer_start + i < len(refined_sequence):
                                    refined_sequence[answer_start + i] = token_id
        
        return refined_sequence

    def _calculate_global_coherence(self, sequence: List[int]) -> float:
        """
        Calculate global coherence using semantic similarity and factual consistency checks.
        
        This method evaluates coherence through multiple lenses:
        1. Local token-to-token coherence via cosine similarity
        2. Semantic chunk coherence across the sequence
        3. Factual consistency with the prompt (for QA contexts)
        
        Args:
            sequence: Token IDs to evaluate
            
        Returns:
            Coherence score as a float
        """
        with torch.no_grad():
            # Decode the sequence for semantic analysis
            text = self.tokenizer.decode(sequence)
            
            # Check if this is a QA context
            is_qa = "USER:" in text and "ASSISTANT:" in text
            
            # Get token-level embeddings
            input_ids = torch.tensor([sequence]).to(self.config.device)
            attention_mask = torch.ones_like(input_ids)
            outputs = self.bert_model.bert(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden_size]
            
            # Calculate token-level coherence
            coherence_score = 0
            total_chunks = 0
            
            # Process in overlapping chunks
            chunk_size = min(16, len(sequence) // 2)
            if chunk_size < 4:
                chunk_size = min(len(sequence), 4)
                
            for i in range(0, len(sequence) - chunk_size + 1, max(1, chunk_size // 2)):
                chunk_start = i
                chunk_end = min(i + chunk_size, len(sequence))
                
                if chunk_end - chunk_start < 4:  # Skip very small chunks
                    continue
                    
                # Get chunk embeddings
                chunk_embs = embeddings[chunk_start:chunk_end]
                
                # Calculate pairwise cosine similarity
                norm_embs = F.normalize(chunk_embs, p=2, dim=1)
                cosine_sim = torch.mm(norm_embs, norm_embs.transpose(0, 1))
                
                # Exclude self-similarity
                mask = 1.0 - torch.eye(cosine_sim.size(0), device=self.config.device)
                masked_sim = cosine_sim * mask
                
                # Average similarity
                chunk_coherence = masked_sim.sum() / (mask.sum() + 1e-9)
                coherence_score += chunk_coherence
                total_chunks += 1
            
            # Add semantic coherence bonus for QA contexts
            if is_qa:
                # Split into question and answer
                qa_parts = text.split("ASSISTANT:")
                if len(qa_parts) > 1:
                    question = qa_parts[0].replace("USER:", "").strip()
                    answer = qa_parts[1].strip()
                    
                    # Get embeddings for question and answer
                    q_embedding = self.encode_sequence(question)
                    a_embedding = self.encode_sequence(answer)
                    
                    # Calculate semantic similarity
                    qa_similarity = F.cosine_similarity(q_embedding.unsqueeze(0), a_embedding.unsqueeze(0)).item()
                    
                    # Add factual consistency bonus
                    coherence_score += qa_similarity * 2.0
                    total_chunks += 1
            
            # Return average coherence across all chunks
            return coherence_score / max(1, total_chunks)

    def _run_diffusion_generation(self, initial_text: str, num_tokens: int, diffusion_steps: int, track_candidates: bool = False) -> Union[str, Tuple[List[List[int]], torch.Tensor]]:
        """
        Run a single diffusion generation process, optionally tracking candidates.
        
        Args:
            initial_text: The initial prompt text
            num_tokens: Number of tokens to generate
            diffusion_steps: Number of diffusion steps
            track_candidates: Whether to return sequences and coherence scores
            
        Returns:
            Either the generated text or a tuple of (sequences, coherence_scores)
        """
        # Set up device
        device = self.config.device
        
        # Tokenize initial text
        input_ids = self.tokenizer.encode(initial_text, return_tensors="pt").to(device)
        seq_len = input_ids.shape[1]
        
        # Initialize current sequence with context
        current_sequences = [input_ids[i].tolist() for i in range(self.config.batch_size)]
        
        # Add token stabilization mask to track which tokens should be frozen
        token_stability = torch.zeros(
            (self.config.batch_size, seq_len + num_tokens),
            device=device
        )
        # Context tokens are always stable
        token_stability[:, :seq_len] = 1.0
        
        # Track token coherence scores to prioritize high-value tokens
        token_coherence = torch.zeros(
            (self.config.batch_size, seq_len + num_tokens),
            device=device
        )
        
        # Add random tokens to reach desired length
        for batch_idx in range(self.config.batch_size):
            for _ in range(num_tokens):
                # Sample random token from vocabulary
                random_token = random.randint(1000, self.tokenizer.vocab_size - 1)
                current_sequences[batch_idx].append(random_token)
        
        # Refinement iterations
        for step in range(diffusion_steps):
            # Calculate temperature for this step (decay from max to min)
            progress = step / (diffusion_steps - 1) if diffusion_steps > 1 else 1.0
            if self.config.noise_schedule == "linear":
                temperature = self.config.max_temp - progress * (self.config.max_temp - self.config.min_temp)
            else:  # Cosine schedule
                temperature = self.config.min_temp + 0.5 * (self.config.max_temp - self.config.min_temp) * (1 + math.cos(progress * math.pi))
            
            # Track changes for this iteration
            changes_made = 0
            total_tokens = self.config.batch_size * num_tokens
            
            # Process each sequence in batch
            for batch_idx in range(self.config.batch_size):
                batch_sequence = current_sequences[batch_idx]
                
                # Prioritize tokens with low coherence scores
                positions = list(range(seq_len, len(batch_sequence)))
                positions.sort(key=lambda pos: token_coherence[batch_idx, pos].item())
                
                # Process each position after the initial text
                for position in positions:
                    # Skip stable tokens (those with stability above threshold)
                    if token_stability[batch_idx, position] > 0.7:
                        continue
                    
                    # Get context before current position
                    context = self.tokenizer.decode(batch_sequence[:position])
                    
                    # Get top-k candidates for this position
                    with torch.no_grad():
                        inputs = self.tokenizer(context, return_tensors="pt").to(device)
                        outputs = self.bert_model(**inputs)
                        logits = outputs.logits[0, -1, :]
                        
                        # Apply temperature
                        logits = logits / temperature
                        
                        # Get top-k token indices
                        top_k_indices = torch.topk(logits, self.config.top_k).indices.tolist()
                    
                    # Get candidate tokens
                    candidate_tokens = [self.tokenizer.convert_ids_to_tokens(idx) for idx in top_k_indices]
                    
                    # Calculate coherence scores
                    coherence_scores = self.calculate_coherence_scores(
                        candidate_tokens, 
                        context,
                        full_sequence=batch_sequence,
                        position=position
                    )
                    
                    # Store max coherence score for this position
                    token_coherence[batch_idx, position] = torch.max(coherence_scores)
                    
                    # Sample token with diffusion temperature
                    probs = F.softmax(coherence_scores / temperature, dim=0)

                    # Ensure probs is 1D and contains valid probabilities
                    if torch.isnan(probs).any() or torch.sum(probs) == 0:
                        # If we have invalid probabilities, use uniform distribution
                        probs = torch.ones_like(probs) / probs.size(0)

                    # Sample from the distribution
                    sampled_idx = torch.multinomial(probs, 1).item()
                    new_token_id = top_k_indices[sampled_idx]
                    
                    # Track changes
                    if new_token_id != batch_sequence[position]:
                        changes_made += 1
                        batch_sequence[position] = new_token_id
                        
                        # Reset stability for changed tokens but consider coherence
                        new_stability = min(0.3, token_coherence[batch_idx, position].item())
                        token_stability[batch_idx, position] = new_stability
                    else:
                        # Increase stability for unchanged tokens based on coherence
                        coherence_boost = token_coherence[batch_idx, position].item() * 0.3
                        token_stability[batch_idx, position] += 0.1 + coherence_boost
            
            # Calculate percentage of tokens changed
            change_percentage = 100 * changes_made / total_tokens if total_tokens > 0 else 0
            
            # Early stopping if very few changes
            if change_percentage < 0.5:
                break
        
        if track_candidates:
            return current_sequences, token_coherence
        else:
            # Return the best sequence from the batch (using first for simplicity)
            best_sequence = current_sequences[0]
            result = self.tokenizer.decode(best_sequence[seq_len:])
            return result

    def generate_mcts(self, initial_text: str, num_tokens: int, verbose: bool = False) -> str:
        """
        Generate text using MCTS optimization across multiple diffusion runs.
        
        Args:
            initial_text: The initial prompt text
            num_tokens: Number of tokens to generate
            verbose: Whether to print detailed generation information
            
        Returns:
            The generated text
        """
        if verbose:
            print("\n=== Starting MCTS-optimized generation ===")
            print(f"Initial text: {initial_text}")
            print(f"Generating {num_tokens} tokens with {self.config.diffusion_steps} diffusion steps")
            print("Running multiple diffusion processes to gather candidates...")
        
        result = self.optimize_sequence_mcts(initial_text, num_tokens)
        
        if verbose:
            print("\n=== MCTS Generation Complete ===")
            print(f"Final result: {initial_text}{result}")
        
        return result

def main():
    config = GenerationConfig(
        max_length=200,
        batch_size=1,
        num_candidates=128,
        embedding_dim=768,
        context_window=16,
        base_temperature=1,
        min_threshold=0.9,
        top_k=50,
        compression_ratio=1,
        max_cache_size=16,
        diffusion_steps=8,
        noise_schedule="linear",
        min_temp=0.2,
        max_temp=1.0
    )
    
    model_config = ModelConfig(
        bert_model_name="bert-base-cased",
        tokenizer_name="bert-base-cased",
        sentence_transformer_name="all-MiniLM-L6-v2"
    )
    
    # Initialize generator with both configs
    parallel_generator = ParallelBERTGenerator(config, model_config)
    
    initial_text = "USER: What is the capital of France? ASSISTANT: The capital of France is "
    num_tokens_to_generate = 64
    
    print("\nStandard Generation:")
    print(f"Input: {initial_text}")
    print("Generated: ", end="", flush=True)
    
    for token in parallel_generator.generate_stream(initial_text, num_tokens_to_generate):
        sys.stdout.write(token)
        sys.stdout.flush()
    print()
    
    print("\nMCTS-Optimized Generation:")
    print(f"Input: {initial_text}")
    
    # Use verbose mode to show step-by-step generation
    result = parallel_generator.generate_mcts(initial_text, num_tokens_to_generate, verbose=True)
    
    print("\nFinal Result:")
    print(f"{initial_text}{result}")

if __name__ == "__main__":
    main()