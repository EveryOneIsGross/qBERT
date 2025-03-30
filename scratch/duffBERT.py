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
        cross_start_idx = max(0, position - window_size)
        initial_context = matrix[cross_start_idx:position, :, 0, :].transpose(0, 1)
        if initial_context.size(1) > 0:
            cross_proj = self.cross_attn_proj(initial_context)
            cross_attn = F.scaled_dot_product_attention(
                embeddings.unsqueeze(1),
                cross_proj,
                initial_context,
                is_causal=False
            ).squeeze(1)
        else:
            cross_attn = torch.zeros_like(embeddings)
        
        # Combine all contexts
        combined = torch.cat([forward_attn, backward_attn, cross_attn], dim=-1)
        return self.combine_proj(combined)

    def calculate_coherence_scores(self, 
                                candidates: List[str],
                                context: str) -> torch.Tensor:
        """Calculate coherence scores using configured phrase window."""
        with torch.no_grad():
            context_tokens = self.tokenizer.tokenize(context)[-self.config.phrase_window:]
            
            # Add length-based penalty for single tokens
            length_penalties = torch.tensor([
                1.0 if len(self.tokenizer.tokenize(candidate)) > 1 else 0.7
                for candidate in candidates
            ], device=self.config.device)
            
            candidate_phrases = [
                self.tokenizer.convert_tokens_to_string(context_tokens + [candidate]) 
                for candidate in candidates
            ]
            
            phrase_embeddings = self.encode_sequence(candidate_phrases)
            context_embedding = self.encode_sequence(context)
            
            similarity = F.cosine_similarity(
                phrase_embeddings.unsqueeze(1),
                context_embedding.unsqueeze(0),
                dim=-1
            )
            
            # Apply length penalty to similarity scores
            return similarity * length_penalties

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
                coherence_scores = self.calculate_coherence_scores(candidate_tokens, context)
                
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

    def generate_diffusion_stream(self, initial_text: str, num_tokens: int) -> Iterator[str]:
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
        
        # Iterative refinement - diffusion steps
        for step in range(self.config.diffusion_steps):
            # Get temperature for this diffusion step
            temperature = self.get_diffusion_temperature(step, self.config.diffusion_steps)
            
            # Track changes for convergence check
            token_changes = 0
            total_positions = 0
            
            # For each batch, refine all positions
            for batch_idx in range(self.config.batch_size):
                batch_sequence = current_sequences[batch_idx]
                
                # Process each position to refine
                for pos in range(seq_len, seq_len + tokens_to_generate):
                    total_positions += 1
                    
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
                    coherence_scores = self.calculate_coherence_scores(candidate_tokens, context)
                    
                    # Sample token with diffusion temperature
                    selected_idx = self.coherence_based_sampling(
                        top_k_indices,
                        coherence_scores,
                        F.softmax(top_k_logits, dim=0),
                        temperature=temperature
                    )
                    
                    # Get selected token
                    selected_token_id = top_k_indices[selected_idx].item()
                    
                    # Check if token changed
                    current_token_id = batch_sequence[pos] if pos < len(batch_sequence) else None
                    
                    if current_token_id != selected_token_id:
                        token_changes += 1
                        
                        # Update sequence and input IDs
                        batch_sequence[pos] = selected_token_id
                        masked_input_ids[batch_idx, pos] = selected_token_id
                
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
            
            # Check for convergence
            if total_positions > 0:
                change_ratio = token_changes / total_positions
                if step > 0 and change_ratio < 0.05:  # Early stopping if < 5% tokens changed
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

    def generate_diffusion(self, initial_text: str, num_tokens: int) -> str:
        """Non-streaming diffusion generation."""
        return "".join(self.generate_diffusion_stream(initial_text, num_tokens))

def main():
    config = GenerationConfig(
        max_length=100,
        batch_size=1,
        num_candidates=50,
        embedding_dim=768,
        context_window=5,
        base_temperature=0.5,
        min_threshold=0.9,
        top_k=50,
        compression_ratio=0.5,
        max_cache_size=3,
        diffusion_steps=5,
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
    
    initial_text = "The quick brown fox"
    num_tokens_to_generate = 20
    
    print("\nStandard Generation:")
    print(f"Input: {initial_text}")
    print("Generated: ", end="", flush=True)
    
    for token in parallel_generator.generate_stream(initial_text, num_tokens_to_generate):
        sys.stdout.write(token)
        sys.stdout.flush()
    print()
    
    print("\nDiffusion-Style Generation:")
    print(f"Input: {initial_text}")
    print("Generated: ", end="", flush=True)
    
    for token in parallel_generator.generate_diffusion_stream(initial_text, num_tokens_to_generate):
        sys.stdout.write(token)
        sys.stdout.flush()
    print()

if __name__ == "__main__":
    main()