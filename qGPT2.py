"""
qGPT2: Semantically Guided GPT-2 Text Generation with Coherence Enforcement

Version: 0.1
Date: 05072025

qGPT2 adapts the semantic coherence mechanisms from qBERT but uses GPT-2 as the 
underlying language model. It guides traditional autoregressive generation with
semantic similarity scoring and thematic memory to produce more coherent text.

**Core Mechanisms:**

*   **Semantic Coherence Guidance:** Candidate tokens from GPT-2 are evaluated based on 
    their semantic similarity to the current context using sentence transformers.
    
*   **Adaptive Temperature Sampling:** Dynamically adjusts sampling temperature based on
    the calculated coherence scores, allowing for more creative exploration when 
    coherence is high and more focused generation when thematic consistency needs to
    be maintained.
    
*   **Sequence Memory Cache:** Stores generated sequences and their semantic embeddings,
    allowing the model to boost tokens that appeared in semantically similar contexts
    previously.

**Dependencies:**
*   PyTorch
*   Transformers (Hugging Face)
*   Sentence-Transformers
"""

import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Iterator, Dict, Any
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import transformers
from pydantic import BaseModel, Field
transformers.logging.set_verbosity_error()  # Only show errors, not warnings

@dataclass
class GenerationConfig:
    """Configuration for the qGPT2 generator."""
    max_length: int = 100
    batch_size: int = 1
    num_candidates: int = 64
    context_window: int = 256
    phrase_window: int = 32
    base_temperature: float = 1.0
    min_threshold: float = 0.5
    top_k: int = 32
    top_p: float = 0.95
    sequence_cache_size: int = 5
    max_cache_size: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Toggle mechanisms
    use_adaptive_temperature: bool = True
    use_coherence_scoring: bool = True
    use_sequence_memory: bool = True

class ModelConfig(BaseModel):
    """Configuration for transformer models used in qGPT2"""
    gpt2_model_name: str = Field(
        default="gpt2",
        description="Name or path of GPT-2 model to use"
    )
    tokenizer_name: str = Field(
        default="gpt2",
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

class SequenceMemoryCache:
    """Caches and retrieves complete text sequences for better coherence and recollection."""
    
    def __init__(self, config: GenerationConfig, sentence_transformer: SentenceTransformer):
        self.config = config
        self.sentence_transformer = sentence_transformer
        self.sequence_memory: Dict[str, List[Dict[str, Union[str, torch.Tensor]]]] = {}
        self.similarity_threshold = 0.7
        
    def encode_sequence(self, text: str) -> torch.Tensor:
        """Encode text sequence using sentence transformer."""
        with torch.no_grad():
            embeddings = self.sentence_transformer.encode(text, convert_to_tensor=True)
            return embeddings.to(self.config.device)
    
    def add_sequence(self, context_key: str, generated_text: str):
        """Add a generated sequence to the memory cache."""
        if not self.config.use_sequence_memory:
            return
            
        # Initialize memory for this context if it doesn't exist
        if context_key not in self.sequence_memory:
            self.sequence_memory[context_key] = []
            
        # Encode the sequence
        embedding = self.encode_sequence(generated_text)
        
        # Add to memory
        self.sequence_memory[context_key].append({
            "text": generated_text,
            "embedding": embedding
        })
        
        # Limit memory size per context
        if len(self.sequence_memory[context_key]) > self.config.sequence_cache_size:
            self.sequence_memory[context_key].pop(0)  # Remove oldest
            
        # Limit total contexts
        if len(self.sequence_memory) > self.config.max_cache_size:
            # Remove oldest context
            oldest_key = next(iter(self.sequence_memory))
            del self.sequence_memory[oldest_key]
    
    def retrieve_similar_sequences(self, query: str, context_key: str = None) -> List[str]:
        """Retrieve sequences similar to the query."""
        if not self.config.use_sequence_memory:
            return []
            
        query_embedding = self.encode_sequence(query)
        similar_sequences = []
        
        # Helper to calculate similarity and filter
        def process_sequences(sequences):
            results = []
            for seq in sequences:
                similarity = F.cosine_similarity(
                    query_embedding.unsqueeze(0),
                    seq["embedding"].unsqueeze(0),
                    dim=1
                ).item()
                
                if similarity > self.similarity_threshold:
                    results.append({"text": seq["text"], "similarity": similarity})
            return results
            
        # If context_key provided, only search in that context
        if context_key and context_key in self.sequence_memory:
            similar_sequences = process_sequences(self.sequence_memory[context_key])
        else:
            # Search across all contexts
            for context_sequences in self.sequence_memory.values():
                similar_sequences.extend(process_sequences(context_sequences))
                
        # Sort by similarity (highest first)
        similar_sequences.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return just the text
        return [seq["text"] for seq in similar_sequences]

class SemanticGPT2Generator(nn.Module):
    """GPT-2 generator with semantic coherence guidance, adaptive temperature, and sequence memory."""
    def __init__(self, config: GenerationConfig, model_config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        self.model_config = model_config
        
        # Initialize models
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(
            model_config.gpt2_model_name,
            attn_implementation=model_config.attn_implementation
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_config.tokenizer_name)
        # Ensure the tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.sentence_transformer = SentenceTransformer(
            model_config.sentence_transformer_name,
            trust_remote_code=True
        )
        
        # Move models to device and eval mode
        self.gpt2_model = self.gpt2_model.to(config.device)
        self.gpt2_model.eval()
        
        # Initialize sequence memory cache
        self.sequence_memory = SequenceMemoryCache(config, self.sentence_transformer)
    
    def encode_sequence(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Encode text sequence using sentence transformer."""
        with torch.no_grad():
            embeddings = self.sentence_transformer.encode(text, convert_to_tensor=True)
            return embeddings.to(self.config.device)
    
    def get_gpt2_predictions(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get GPT-2 predictions for the next token."""
        with torch.no_grad():
            outputs = self.gpt2_model(input_ids)
            logits = outputs.logits[:, -1, :]  # Get logits for the last position only
            
            # Apply top-k filtering
            top_k_logits, top_k_indices = torch.topk(
                logits, 
                min(self.config.top_k, logits.size(-1))
            )
            
            # Optional: Apply top-p filtering (nucleus sampling)
            if hasattr(self.config, 'top_p') and self.config.top_p < 1.0:
                probs = F.softmax(top_k_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > self.config.top_p
                # Keep tokens that are above threshold but make sure at least one token remains
                if sorted_indices_to_remove.any():
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                # Scatter back the indices
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, 
                    index=sorted_indices, 
                    src=sorted_indices_to_remove
                )
                
                # Only keep tokens that pass the filter
                filtered_logits = top_k_logits.masked_fill(indices_to_remove, -float('inf'))
                filtered_indices = top_k_indices.clone()
                
                # Get new top-k after filtering
                valid_count = (~indices_to_remove).sum().item()
                if valid_count > 0:
                    filtered_top_k = min(valid_count, self.config.top_k)
                    top_k_logits, idx = torch.topk(filtered_logits, k=filtered_top_k)
                    top_k_indices = filtered_indices.gather(-1, idx)
            
            return top_k_logits, top_k_indices
    
    def calculate_coherence_scores(self, 
                                  candidates: List[str],
                                  context: str) -> torch.Tensor:
        """Calculate coherence scores using configured phrase window."""
        if not candidates:
            return torch.tensor([], device=self.config.device)
            
        with torch.no_grad():
            # Limit context to the phrase window to avoid excessive computation
            if len(context) > 0:
                # Use tokenizer to get the last N tokens from context
                context_tokens = self.tokenizer.tokenize(context)[-self.config.phrase_window:]
                context_text = self.tokenizer.convert_tokens_to_string(context_tokens)
            else:
                context_text = ""
                context_tokens = []
            
            # Add length-based penalty for single tokens
            length_penalties = torch.tensor([
                1.0 if len(self.tokenizer.tokenize(candidate)) > 1 else 0.8
                for candidate in candidates
            ], device=self.config.device)
            
            # Create candidate phrases by combining context with each candidate
            candidate_phrases = []
            for candidate in candidates:
                if context_tokens:
                    # For GPT-2, we need to ensure proper spacing
                    if not context_text.endswith(' ') and not candidate.startswith(' '):
                        phrase = context_text + ' ' + candidate
                    else:
                        phrase = context_text + candidate
                else:
                    phrase = candidate
                candidate_phrases.append(phrase)
            
            # Handle empty context case
            if not context_text:
                # If no context, just return length penalties as scores
                return length_penalties
            
            # Check sequence memory for similar sequences
            if self.config.use_sequence_memory:
                context_key = context_text[:50]  # Use prefix as key
                similar_sequences = self.sequence_memory.retrieve_similar_sequences(context_text, context_key)
                
                # If we found similar sequences, boost candidates that appear in them
                if similar_sequences:
                    sequence_boost = torch.ones(len(candidates), device=self.config.device)
                    for i, candidate in enumerate(candidates):
                        for sequence in similar_sequences:
                            if candidate in sequence:
                                sequence_boost[i] *= 1.2  # Boost by 20%
                                break
                    
                    # Apply boosting to length penalties
                    length_penalties *= sequence_boost
            
            # Calculate embeddings and similarity
            try:
                phrase_embeddings = self.encode_sequence(candidate_phrases)
                context_embedding = self.encode_sequence(context_text)
                
                similarity = F.cosine_similarity(
                    phrase_embeddings,
                    context_embedding.unsqueeze(0).expand(len(candidates), -1),
                    dim=-1
                )
                
                # Apply length penalty to similarity scores
                return similarity * length_penalties
            except Exception as e:
                # Fallback to uniform scores if embedding fails
                print(f"Warning: Coherence scoring failed with error: {e}")
                return torch.ones(len(candidates), device=self.config.device)
    
    def coherence_based_sampling(self, 
                               candidates: torch.Tensor,
                               coherence_scores: torch.Tensor,
                               gpt2_scores: torch.Tensor) -> torch.Tensor:
        """Sample tokens based on combined GPT-2 and coherence scores."""
        # Handle empty candidates case
        if len(candidates) == 0:
            return torch.tensor(0, device=self.config.device)
            
        # Ensure all tensors have the same length
        min_len = min(len(candidates), len(coherence_scores), len(gpt2_scores))
        candidates = candidates[:min_len]
        coherence_scores = coherence_scores[:min_len]
        gpt2_scores = gpt2_scores[:min_len]
        
        # Normalize scores to prevent numerical issues
        gpt2_scores = F.softmax(gpt2_scores, dim=-1)
        coherence_scores = torch.clamp(coherence_scores, min=0.1, max=1.0)
        
        combined_scores = gpt2_scores * coherence_scores
        
        # Apply adaptive temperature if enabled, otherwise use base temperature
        if self.config.use_adaptive_temperature:
            mean_coherence = torch.mean(coherence_scores)
            temperature = self.config.base_temperature * (1 / mean_coherence)
            temperature = torch.clamp(temperature, min=0.1, max=2.0)
        else:
            temperature = self.config.base_temperature
        
        # Apply threshold filtering
        threshold = torch.mean(combined_scores) * self.config.min_threshold
        valid_candidates = combined_scores > threshold
        
        # If no candidates pass the threshold, use all candidates
        if not torch.any(valid_candidates):
            valid_candidates = torch.ones_like(combined_scores, dtype=torch.bool)
        
        # Apply temperature and sample
        scaled_scores = combined_scores[valid_candidates] / temperature
        
        # Handle numerical stability
        max_score = torch.max(scaled_scores)
        exp_scores = torch.exp(scaled_scores - max_score)
        probs = exp_scores / torch.sum(exp_scores)
        
        # Ensure probs sum to 1
        probs = probs / torch.sum(probs)
        
        # Sample from the distribution
        try:
            sample_idx = torch.multinomial(probs, num_samples=1)
            valid_indices = torch.where(valid_candidates)[0]
            selected_idx = valid_indices[sample_idx]
        except RuntimeError:
            # Fallback to argmax if sampling fails
            selected_idx = torch.argmax(combined_scores).unsqueeze(0)
        
        return selected_idx.squeeze()
    
    def generate_stream(self, initial_text: str, num_tokens: int) -> Iterator[str]:
        """Stream tokens one by one with semantic coherence guidance."""
        # Tokenize initial text
        input_ids = self.tokenizer.encode(initial_text, return_tensors='pt').to(self.config.device)
        current_text = initial_text
        
        # Retrieve similar sequences if enabled
        context_key = initial_text[:50]  # Use prefix as key
        similar_sequences = []
        if self.config.use_sequence_memory:
            similar_sequences = self.sequence_memory.retrieve_similar_sequences(initial_text, context_key)
        
        # Generate tokens
        for _ in range(num_tokens):
            # Get GPT-2 predictions
            top_k_logits, top_k_indices = self.get_gpt2_predictions(input_ids)
            
            # Convert token IDs to text for coherence scoring
            candidate_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_k_indices[0]]
            
            # Calculate coherence scores if enabled, otherwise use uniform scores
            if self.config.use_coherence_scoring:
                coherence_scores = self.calculate_coherence_scores(candidate_tokens, current_text)
            else:
                coherence_scores = torch.ones(len(candidate_tokens), device=self.config.device)
            
            # Sample next token
            selected_idx = self.coherence_based_sampling(
                top_k_indices[0],
                coherence_scores,
                top_k_logits[0]
            )
            
            # Get the selected token ID and convert to text
            selected_token_id = top_k_indices[0][selected_idx].unsqueeze(0).unsqueeze(0)
            selected_token_text = self.tokenizer.decode(selected_token_id[0])
            
            # Update state
            input_ids = torch.cat([input_ids, selected_token_id], dim=1)
            current_text += selected_token_text
            
            # Yield the token
            yield selected_token_text
            
            # Check for EOS
            if selected_token_id.item() == self.tokenizer.eos_token_id:
                break
        
        # Update sequence memory with complete generated text
        if self.config.use_sequence_memory:
            self.sequence_memory.add_sequence(context_key, current_text)
    
    def generate(self, initial_text: str, num_tokens: int, stop_sequences: List[str] = None) -> str:
        """Non-streaming generation function that returns complete text.
        
        Args:
            initial_text: The prompt text to start generation from
            num_tokens: Maximum number of tokens to generate
            stop_sequences: Optional list of strings that signal where to stop generation
        """
        tokens = []
        generated_text = ""
        
        # Default stop sequences if none provided
        if stop_sequences is None:
            stop_sequences = []
        
        # Always include EOS token
        if self.tokenizer.eos_token not in stop_sequences:
            stop_sequences.append(self.tokenizer.eos_token)
        
        # Generate token stream
        for token in self.generate_stream(initial_text, num_tokens):
            tokens.append(token)
            generated_text += token
            
            # Check if any stop sequence appears in the recently generated text
            # We look at the most recent chunk to avoid checking the entire text each time
            check_text = "".join(tokens[-10:])  # Look at the last 10 tokens
            
            for stop_seq in stop_sequences:
                if stop_seq in check_text:
                    # Find where in the entire text the stop sequence appears
                    stop_idx = generated_text.find(stop_seq)
                    if stop_idx != -1:
                        # Return everything before the stop sequence
                        return generated_text[:stop_idx]
        
        return generated_text

def generate_text(model: SemanticGPT2Generator, 
                  prompt: str, 
                  max_length: int = 100, 
                  verbose: bool = False,
                  stream: bool = False,
                  show_probabilities: bool = False) -> str:
    """Generate text using the model with optional streaming display and probability visualization."""
    print(f"\nPrompt: {prompt}")
    
    # If we want to show probabilities, we need to modify the generation process
    if show_probabilities:
        print("\nGenerated with probabilities:")
        generated_text = prompt
        input_ids = model.tokenizer.encode(prompt, return_tensors='pt').to(model.config.device)
        
        for _ in range(max_length):
            # Get GPT-2 predictions
            top_k_logits, top_k_indices = model.get_gpt2_predictions(input_ids)
            
            # Convert token IDs to text
            candidate_tokens = [model.tokenizer.decode([idx.item()]) for idx in top_k_indices[0]]
            
            # Calculate coherence scores
            coherence_scores = model.calculate_coherence_scores(candidate_tokens, generated_text) if model.config.use_coherence_scoring else torch.ones(len(candidate_tokens), device=model.config.device)
            
            # Get GPT-2 probabilities
            gpt2_probs = F.softmax(top_k_logits[0], dim=-1)
            
            # Calculate combined scores
            combined_scores = gpt2_probs * torch.clamp(coherence_scores, min=0.1, max=1.0)
            combined_probs = combined_scores / combined_scores.sum()
            
            # Sample next token
            selected_idx = model.coherence_based_sampling(
                top_k_indices[0],
                coherence_scores,
                top_k_logits[0]
            )
            
            # Get selected token
            selected_token_id = top_k_indices[0][selected_idx].unsqueeze(0).unsqueeze(0)
            selected_token_text = model.tokenizer.decode(selected_token_id[0])
            
            # Update for next iteration
            input_ids = torch.cat([input_ids, selected_token_id], dim=1)
            generated_text += selected_token_text
            
            # Print top 5 candidates with their scores
            print(f"\nStep {_+1}:")
            print(f"Token chosen: '{selected_token_text}' → {generated_text}")
            
            # Create a sorted list of (token, gpt2_prob, coherence, combined) tuples
            token_data = []
            for i, token in enumerate(candidate_tokens[:10]):  # Limit to top 10 for display
                token_data.append((
                    token, 
                    gpt2_probs[i].item(), 
                    coherence_scores[i].item() if i < len(coherence_scores) else 0.0,
                    combined_probs[i].item() if i < len(combined_probs) else 0.0
                ))
            
            # Sort by combined probability
            token_data.sort(key=lambda x: x[3], reverse=True)
            
            # Print in a compact format
            print(f"{'Token':<8} | {'GPT2%':<7} | {'Coh.':<7} | {'Combined%':<7}")
            print("-" * 36)
            for token, gpt2_p, coh, comb in token_data[:5]:  # Show top 5
                # Highlight the selected token
                is_selected = token == selected_token_text
                token_fmt = f"'{token}'" if len(token) <= 6 else f"'{token[:5]}…'"
                prefix = "→ " if is_selected else "  "
                print(f"{prefix}{token_fmt:<8} | {gpt2_p*100:>6.1f}% | {coh:>6.3f} | {comb*100:>6.1f}%")
            
            # Check for EOS
            if selected_token_id.item() == model.tokenizer.eos_token_id:
                break
        
        # Update sequence memory
        if model.config.use_sequence_memory:
            context_key = prompt[:50]
            model.sequence_memory.add_sequence(context_key, generated_text)
            
        return generated_text
    
    elif stream:
        print("\nGenerated: ", end="", flush=True)
        generated_text = prompt
        
        for token in model.generate_stream(prompt, max_length):
            print(token, end="", flush=True)
            generated_text += token
            
        print("\n")
        return generated_text
    else:
        result = model.generate(prompt, max_length)
        print(f"\nGenerated: {result}\n")
        return result

if __name__ == "__main__":
    # Configuration for the model
    gen_config = GenerationConfig(
        max_length=100,
        batch_size=1,
        num_candidates=32,
        context_window=5,
        phrase_window=32,
        base_temperature=1.0,
        min_threshold=0.5,
        top_k=20,  # Reduced for cleaner probability display
        top_p=0.9,
        use_adaptive_temperature=True,
        use_coherence_scoring=True,
        use_sequence_memory=True
    )
    
    model_config = ModelConfig(
        gpt2_model_name="gpt2",  # Can be gpt2, gpt2-medium, gpt2-large, etc.
        tokenizer_name="gpt2",
        sentence_transformer_name="all-MiniLM-L6-v2"
    )
    
    # Create the model
    model = SemanticGPT2Generator(gen_config, model_config)
    
    # Example prompts to demonstrate the model's capabilities
    prompts = [
        "The future of artificial intelligence is",
        "Climate change will impact our society by",
    ]
    
    # Generate text with probability display for each prompt
    for prompt in prompts:
        generate_text(model, prompt, max_length=5, show_probabilities=True)
        
    # Demonstrate the difference with coherence scoring disabled
    print("\n--- Comparison with coherence scoring disabled ---")
    model.config.use_coherence_scoring = False
    generate_text(model, "The future of artificial intelligence is", max_length=5, show_probabilities=True)
    
    # Re-enable coherence scoring and disable adaptive temperature
    print("\n--- Comparison with adaptive temperature disabled ---")
    model.config.use_coherence_scoring = True
    model.config.use_adaptive_temperature = False
    generate_text(model, "The future of artificial intelligence is", max_length=5, show_probabilities=True)
    
    # Additional short demonstrations of regular generation
    print("\n--- Standard generation examples ---")
    standard_prompts = [
        "The key to effective communication is",
        "Scientists have recently discovered that"
    ]
    
    for prompt in standard_prompts:
        generate_text(model, prompt, max_length=30, stream=True) 