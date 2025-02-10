"""
qBERT: Bidirectional Attention Language BERT - Coherent Text Generation with Bidirectional Context

Version: 0.5
Author: cursed.helm
Date: 04022025

qBERT is an experimental neural text generation model designed to explore techniques for generating more coherent and thematically consistent text, moving beyond traditional unidirectional language models.

**Core Explorations and Innovations:**

*   **Bidirectional Contextual Attention:**  Implements a novel approach to text generation by processing context bidirectionally.  Unlike standard models that primarily consider preceding tokens, qBERT utilizes dual-direction projection layers to attend to both forward (preceding) and backward (succeeding, within a window) context, aiming for a more holistic understanding when generating each token.

*   **Semantic Coherence Guidance:**  Integrates semantic coherence scoring to guide the generation process. Candidate tokens are evaluated not only based on statistical likelihood but also on their semantic similarity to the surrounding context (using sentence transformers). This approach aims to produce text that is not just fluent but also thematically relevant and consistent.

*   **Adaptive Temperature Sampling:**  Employs adaptive temperature sampling, dynamically adjusting the sampling temperature based on the calculated coherence scores. This allows for more creative exploration when coherence is high and more focused generation when thematic consistency needs to be maintained.

*   **Semantic Memory Cache:**  Introduces a semantic memory cache to enhance long-term thematic coherence and potentially enable more consistent "voice" or persona in generated text.  Compressed semantic representations of generated content are cached and can influence subsequent generation, promoting thematic continuity across longer sequences.

*   **Experimental Batch Processing with 4D Matrix:** Explores the use of a 4D matrix structure to manage embeddings and potentially enable parallel processing for improved generation efficiency.  While the structural foundation for batch processing is implemented, full parallel speedup is an ongoing area of development.

**Key Features:**

*   **Bidirectional Contextual Attention Mechanism:** Custom-built layers process forward and backward context, going beyond unidirectional generation.
*   **Semantic Coherence Scoring:**  Leverages sentence transformers to assess and guide generation towards semantic relevance.
*   **Adaptive Temperature Sampling:**  Dynamically adjusts sampling randomness based on coherence.
*   **Semantic Memory Caching:**  Caches semantic information to promote thematic consistency over extended generations.
*   **Cross-Attention:**  Uses cross-attention to guide generation towards original position bound by context_window.
*   **Streaming Generation:** Supports efficient streaming text generation with attention masking.

**Architecture Overview:**

*   **Base Model:** BERT (bert-base-uncased) - Provides the foundation for language understanding and masked language modeling.
*   **Semantic Model:** Sentence Transformer (all-MiniLM-L6-v2) - Used for semantic embedding and coherence scoring.
*   **Custom Components:**
    *   **Bidirectional Attention Layers:** `forward_proj`, `backward_proj`, `combine_proj` - Implement the novel bidirectional context processing.
    *   **Semantic Coherence and Sampling Logic:** `calculate_coherence_scores`, `coherence_based_sampling` - Guide generation based on semantic relevance.
    *   **Semantic Memory Cache:** `SemanticMemoryCache` class - Manages and utilizes cached semantic information.
    *   **4D Matrix:** `matrix` - Experimental structure for managing embeddings and exploring batch processing.

**Dependencies:**

*   PyTorch
*   Transformers (Hugging Face)
*   Sentence-Transformers

**Note:**

qBERT is a buzzy wee guy. It aims to explore and evaluate novel techniques for coherent text generation outside of the enc/dec paradigm.  The architecture is under attention-driven development, and results should be considered preliminary and exploratory.  Performance and coherence characteristics are subjects of occasional investigation and refinement.


**Auxiliary Notes:**

I am using this to guide berts output, but the same framework I posit would work using logits from llm models, by batching the logits and using the same framework to guide the output. 

"""

import warnings
# Must import warnings first and set filters before other imports
warnings.filterwarnings('ignore')  # Suppress all warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Iterator, Dict
from transformers import AutoTokenizer, BertForMaskedLM
from sentence_transformers import SentenceTransformer
import sys
from pydantic import BaseModel, Field

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
    compression_ratio: float = 0.5
    max_cache_size: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

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
        self.sentence_transformer = SentenceTransformer(model_config.sentence_transformer_name)
        
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
                                context: str,
                                phrase_window: int = 3) -> torch.Tensor:
        with torch.no_grad():
            context_tokens = self.tokenizer.tokenize(context)[-phrase_window:]
            
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
                               bert_scores: torch.Tensor) -> torch.Tensor:
        """Sample tokens based on combined BERT and coherence scores."""
        # Increase threshold for single-token candidates
        single_token_mask = torch.tensor([
            0.8 if len(self.tokenizer.convert_ids_to_tokens([idx.item()])[0]) <= 4 else 1.0
            for idx in candidates
        ], device=self.config.device)
        
        combined_scores = bert_scores * coherence_scores * single_token_mask
        
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
        max_cache_size=3
    )
    
    model_config = ModelConfig(
        bert_model_name="bert-base-cased",
        tokenizer_name="bert-base-cased",
        sentence_transformer_name="all-MiniLM-L6-v2"
    )
    
    # Initialize generator with both configs
    parallel_generator = ParallelBERTGenerator(config, model_config)
    
    initial_text = "The quick brown fox"
    num_tokens_to_generate = 10
    
    print("\nParallel BERT Generator:")
    print(f"Input: {initial_text}")
    print("Generated: ", end="", flush=True)
    
    for token in parallel_generator.generate_stream(initial_text, num_tokens_to_generate):
        sys.stdout.write(token)
        sys.stdout.flush()
    print()

if __name__ == "__main__":
    main()