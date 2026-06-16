#!/usr/bin/env python3
"""
qGPT2 Chatbot Demo

A simple interactive chatbot that demonstrates the capabilities of qGPT2 vs vanilla GPT-2,
allowing users to configure generation parameters and see the effects of semantic guidance.
"""

import argparse
import sys
import torch
from qGPT2 import (
    SemanticGPT2Generator, 
    GenerationConfig, 
    ModelConfig,
)
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ChatbotDemo:
    def __init__(self, 
                 model_name="gpt2", 
                 max_length=50, 
                 top_k=20, 
                 top_p=0.9, 
                 temperature=1.0,
                 num_candidates=32,
                 context_window=5,
                 phrase_window=32,
                 min_threshold=0.5,
                 sequence_cache_size=5,
                 max_history_tokens=512,
                 use_coherence=True,
                 use_adaptive_temp=True,
                 use_sequence_memory=True,
                 model_mode="both"):
        
        print(f"Initializing models (this may take a moment)...")
        
        # Set model mode (vanilla, qgpt2, or both)
        self.model_mode = model_mode.lower()
        valid_modes = ["vanilla", "qgpt2", "both"]
        if self.model_mode not in valid_modes:
            print(f"Invalid model mode: {model_mode}. Using 'both' as default.")
            self.model_mode = "both"
            
        print(f"Model mode: {self.model_mode}")
        
        # Store max_history_tokens
        self.max_history_tokens = max_history_tokens
        
        # Initialize vanilla GPT-2 if needed
        if self.model_mode in ["vanilla", "both"]:
            self.vanilla_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            if self.vanilla_tokenizer.pad_token is None:
                self.vanilla_tokenizer.pad_token = self.vanilla_tokenizer.eos_token
            
            self.vanilla_model = GPT2LMHeadModel.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.vanilla_model = self.vanilla_model.to(self.device)
            self.vanilla_model.eval()
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.vanilla_model = None
            self.vanilla_tokenizer = None
        
        # Initialize qGPT2 if needed
        if self.model_mode in ["qgpt2", "both"]:
            self.gen_config = GenerationConfig(
                max_length=max_length,
                batch_size=1,
                num_candidates=num_candidates,
                context_window=context_window,
                phrase_window=phrase_window,
                base_temperature=temperature,
                min_threshold=min_threshold,
                top_k=top_k,
                top_p=top_p,
                sequence_cache_size=sequence_cache_size,
                use_adaptive_temperature=use_adaptive_temp,
                use_coherence_scoring=use_coherence,
                use_sequence_memory=use_sequence_memory,
                device=self.device
            )
            
            self.model_config = ModelConfig(
                gpt2_model_name=model_name,
                tokenizer_name=model_name,
                sentence_transformer_name="all-MiniLM-L6-v2"
            )
            
            self.qgpt2_model = SemanticGPT2Generator(self.gen_config, self.model_config)
        else:
            self.gen_config = None
            self.model_config = None
            self.qgpt2_model = None
        
        # Separate conversation histories for each model
        self.vanilla_history = ""
        self.qgpt2_history = ""
        
        print(f"Models loaded successfully. Using device: {self.device}")
        
    def generate_vanilla_response(self, user_input, max_length=50):
        """Generate a response using vanilla GPT-2 with its own history."""
        # Check if vanilla model is enabled
        if self.model_mode not in ["vanilla", "both"]:
            return "Vanilla GPT-2 model is disabled."
            
        # Build prompt with vanilla history
        prompt = self.vanilla_history + f"\nUser: {user_input}\nAI:"
        
        input_ids = self.vanilla_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Define stop sequences for proper turn-taking
        stop_tokens = ["\nUser:", "\n\nUser:", "\n\n", "<|endoftext|>"]
        stop_token_ids = [self.vanilla_tokenizer.encode(token, add_special_tokens=False, return_tensors="pt").to(self.device) for token in stop_tokens]
        
        # Generate response
        output = self.vanilla_model.generate(
            input_ids,
            max_length=input_ids.shape[1] + max_length,
            do_sample=True,
            top_k=self.gen_config.top_k if self.gen_config else 50,
            top_p=self.gen_config.top_p if self.gen_config else 0.9,
            temperature=self.gen_config.base_temperature if self.gen_config else 1.0,
            pad_token_id=self.vanilla_tokenizer.pad_token_id,
            num_return_sequences=1,
        )
        
        # Decode and extract only the newly generated part
        full_response = self.vanilla_tokenizer.decode(output[0], skip_special_tokens=True)
        new_response = full_response[len(prompt):]
        
        # Trim response at any stop sequence
        for stop_token in stop_tokens:
            if stop_token in new_response:
                new_response = new_response.split(stop_token)[0]
        
        # Update vanilla history
        self.update_vanilla_history(user_input, new_response.strip())
        
        return new_response.strip()
    
    def generate_qgpt2_response(self, user_input, max_length=50):
        """Generate a response using qGPT2 with its own history."""
        # Check if qGPT2 model is enabled
        if self.model_mode not in ["qgpt2", "both"]:
            return "qGPT2 model is disabled."
            
        # Build prompt with qGPT2 history
        prompt = self.qgpt2_history + f"\nUser: {user_input}\nAI:"
        
        # Set up stop sequences for proper turn-taking
        stop_sequences = ["\nUser:", "\n\nUser:", "\n\n", "<|endoftext|>"]
        
        # Use the stop_sequences parameter directly in the generate method
        response = self.qgpt2_model.generate(
            prompt, 
            max_length,
            stop_sequences=stop_sequences
        ).strip()
        
        # Update qGPT2 history with the trimmed response
        self.update_qgpt2_history(user_input, response)
        
        return response
    
    def update_vanilla_history(self, user_input, response):
        """Update the vanilla GPT-2 conversation history."""
        if self.model_mode not in ["vanilla", "both"]:
            return
            
        new_content = f"\nUser: {user_input}\nAI: {response}"
        
        # Add new content to history
        self.vanilla_history += new_content
        
        # Trim history if it gets too long
        if self.vanilla_tokenizer and len(self.vanilla_tokenizer.encode(self.vanilla_history)) > self.max_history_tokens:
            tokens = self.vanilla_tokenizer.encode(self.vanilla_history)
            tokens = tokens[-self.max_history_tokens:]
            self.vanilla_history = self.vanilla_tokenizer.decode(tokens)
    
    def update_qgpt2_history(self, user_input, response):
        """Update the qGPT2 conversation history."""
        if self.model_mode not in ["qgpt2", "both"]:
            return
            
        new_content = f"\nUser: {user_input}\nAI: {response}"
        
        # Add new content to history
        self.qgpt2_history += new_content
        
        # Trim history if it gets too long
        if self.qgpt2_model and len(self.qgpt2_model.tokenizer.encode(self.qgpt2_history)) > self.max_history_tokens:
            tokens = self.qgpt2_model.tokenizer.encode(self.qgpt2_history)
            tokens = tokens[-self.max_history_tokens:]
            self.qgpt2_history = self.qgpt2_model.tokenizer.decode(tokens)
    
    def update_config(self, **kwargs):
        """Update configuration parameters and reinitialize model if needed."""
        updated_params = []
        for key, value in kwargs.items():
            if hasattr(self.gen_config, key):
                # Store old value for logging
                old_value = getattr(self.gen_config, key)
                
                # Update the config value
                setattr(self.gen_config, key, value)
                updated_params.append((key, old_value, value))
                
        # Only reinitialize if we actually updated any parameters
        if updated_params and self.qgpt2_model and self.model_mode in ["qgpt2", "both"]:
            # Preserve history
            old_history = self.qgpt2_history
            
            # Reinitialize model with new config
            self.qgpt2_model = SemanticGPT2Generator(self.gen_config, self.model_config)
            
            # Restore history
            self.qgpt2_history = old_history
            
            # Print updates
            for param, old_val, new_val in updated_params:
                print(f"Updated {param} from {old_val} to {new_val}")
    
    def set_model_mode(self, mode):
        """Change the current model mode."""
        valid_modes = ["vanilla", "qgpt2", "both"]
        if mode.lower() in valid_modes:
            old_mode = self.model_mode
            self.model_mode = mode.lower()
            
            # Initialize vanilla models if switching to vanilla/both and they don't exist
            if self.model_mode in ["vanilla", "both"] and (self.vanilla_model is None or self.vanilla_tokenizer is None):
                print("Initializing vanilla GPT-2 models...")
                try:
                    # Get model name from existing config or use default
                    model_name = self.model_config.gpt2_model_name if self.model_config else "gpt2"
                    
                    self.vanilla_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                    if self.vanilla_tokenizer.pad_token is None:
                        self.vanilla_tokenizer.pad_token = self.vanilla_tokenizer.eos_token
                    
                    self.vanilla_model = GPT2LMHeadModel.from_pretrained(model_name)
                    self.vanilla_model = self.vanilla_model.to(self.device)
                    self.vanilla_model.eval()
                    print("Vanilla GPT-2 models initialized successfully.")
                except Exception as e:
                    print(f"Error initializing vanilla models: {e}")
                    self.model_mode = old_mode  # Revert mode change
                    return False
            
            # Initialize qGPT2 models if switching to qgpt2/both and they don't exist
            if self.model_mode in ["qgpt2", "both"] and (self.qgpt2_model is None or self.gen_config is None):
                print("Initializing qGPT2 models...")
                try:
                    # Create default configs if they don't exist
                    if self.gen_config is None:
                        self.gen_config = GenerationConfig(
                            max_length=128,
                            batch_size=1,
                            num_candidates=32,
                            context_window=512,
                            phrase_window=32,
                            base_temperature=0.2,
                            min_threshold=0.5,
                            top_k=256,
                            top_p=0.9,
                            sequence_cache_size=5,
                            use_adaptive_temperature=True,
                            use_coherence_scoring=True,
                            use_sequence_memory=True,
                            device=self.device
                        )
                    
                    if self.model_config is None:
                        model_name = "gpt2-medium"  # Default model
                        self.model_config = ModelConfig(
                            gpt2_model_name=model_name,
                            tokenizer_name=model_name,
                            sentence_transformer_name="all-MiniLM-L6-v2"
                        )
                    
                    self.qgpt2_model = SemanticGPT2Generator(self.gen_config, self.model_config)
                    print("qGPT2 models initialized successfully.")
                except Exception as e:
                    print(f"Error initializing qGPT2 models: {e}")
                    self.model_mode = old_mode  # Revert mode change
                    return False
            
            print(f"Model mode set to: {self.model_mode}")
            return True
        else:
            print(f"Invalid model mode: {mode}. Valid options are: {', '.join(valid_modes)}")
            return False
    
    def print_current_config(self):
        """Print current configuration settings."""
        print("\n--- Current Configuration ---")
        print(f"Model Mode (/mode): {self.model_mode}")
        
        # Always show model name first as it's a string value
        print(f"model: \"{self.model_config.gpt2_model_name if hasattr(self, 'model_config') and self.model_config else 'gpt2'}\"")
        
        if self.gen_config:
            print(f"max_length: {self.gen_config.max_length}")
            print(f"top_k: {self.gen_config.top_k}")
            print(f"top_p: {self.gen_config.top_p}")
            print(f"temperature: {self.gen_config.base_temperature}")
            print(f"num_candidates: {self.gen_config.num_candidates}")
            print(f"context_window: {self.gen_config.context_window}")
            print(f"phrase_window: {self.gen_config.phrase_window}")
            print(f"min_threshold: {self.gen_config.min_threshold}")
            print(f"sequence_cache_size: {self.gen_config.sequence_cache_size}")
            print(f"max_history_tokens: {self.max_history_tokens}")
            print(f"use_coherence_scoring: {self.gen_config.use_coherence_scoring}")
            print(f"use_adaptive_temperature: {self.gen_config.use_adaptive_temperature}")
            print(f"use_sequence_memory: {self.gen_config.use_sequence_memory}")
            
            # Display sentence transformer model name if available
            if hasattr(self.model_config, 'sentence_transformer_name'):
                print(f"sentence_transformer_name: \"{self.model_config.sentence_transformer_name}\"")
        else:
            print(f"max_length: N/A")
            print(f"top_k: N/A")
            print(f"top_p: N/A")
            print(f"temperature: N/A")
            print(f"num_candidates: N/A")
            print(f"context_window: N/A")
            print(f"phrase_window: N/A")
            print(f"min_threshold: N/A")
            print(f"sequence_cache_size: N/A")
            print(f"max_history_tokens: N/A")
            print(f"use_coherence_scoring: N/A")
            print(f"use_adaptive_temperature: N/A")
            print(f"use_sequence_memory: N/A")
        
        print("\nTo change settings use: /set <parameter> <value>")
        print("Example: /set model gpt2-medium")
        print("Example: /set top_k 50")
        print("Example: /set num_candidates 64")
        print("----------------------------\n")
    
    def run_chatbot(self):
        """Run the chatbot in an interactive loop."""
        print("=" * 50)
        print("Welcome to the qGPT2 vs. GPT-2 Chatbot Demo!")
        print("Type your message to get responses from enabled models.")
        print("Special commands:")
        print("  /config - Show current configuration")
        print("  /set <param> <value> - Update a configuration parameter")
        print("  /mode <vanilla|qgpt2|both> - Toggle which models to use")
        print("  /clear - Clear conversation histories")
        print("  /quit or /exit - Exit the demo")
        print("=" * 50)
        
        self.print_current_config()
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['/quit', '/exit']:
                    print("Goodbye!")
                    break
                    
                elif user_input.lower() == '/config':
                    self.print_current_config()
                    continue
                    
                elif user_input.lower() == '/clear':
                    self.vanilla_history = ""
                    self.qgpt2_history = ""
                    print("Conversation histories cleared.")
                    continue
                
                elif user_input.lower().startswith('/mode '):
                    mode = user_input.split(maxsplit=1)[1].strip()
                    self.set_model_mode(mode)
                    continue
                    
                elif user_input.lower().startswith('/set '):
                    if not self.gen_config:
                        print("Cannot update config: qGPT2 model is not loaded.")
                        continue
                        
                    parts = user_input.split()
                    if len(parts) >= 3:
                        # Normalize parameter name by converting hyphens to underscores
                        param = parts[1].replace('-', '_')
                        
                        # Get the value part (handle quoted strings)
                        value_part = ' '.join(parts[2:])
                        value_part = value_part.strip('"\'')  # Remove quotes if present
                        
                        try:
                            # Special handling for model parameter
                            if param == 'model':
                                # Changing the model requires reinitialization
                                model_name = value_part
                                print(f"Reinitializing with model: {model_name}")

                                # Save current mode
                                old_mode = self.model_mode

                                # Reinitialize the chatbot with the new model, preserving ALL current config
                                self.__init__(
                                    model_name=model_name,
                                    max_length=self.gen_config.max_length if self.gen_config else 50,
                                    top_k=self.gen_config.top_k if self.gen_config else 50,
                                    top_p=self.gen_config.top_p if self.gen_config else 0.9,
                                    temperature=self.gen_config.base_temperature if self.gen_config else 1.0,
                                    num_candidates=self.gen_config.num_candidates if self.gen_config else 32,
                                    context_window=self.gen_config.context_window if self.gen_config else 5,
                                    phrase_window=self.gen_config.phrase_window if self.gen_config else 32,
                                    min_threshold=self.gen_config.min_threshold if self.gen_config else 0.5,
                                    sequence_cache_size=self.gen_config.sequence_cache_size if self.gen_config else 5,
                                    max_history_tokens=self.max_history_tokens,
                                    use_coherence=self.gen_config.use_coherence_scoring if self.gen_config else True,
                                    use_adaptive_temp=self.gen_config.use_adaptive_temperature if self.gen_config else True,
                                    use_sequence_memory=self.gen_config.use_sequence_memory if self.gen_config else True,
                                    model_mode=old_mode
                                )
                                print(f"Model changed to: {model_name}")
                                continue
                            
                            # Parse value with appropriate type for other parameters
                            if value_part.lower() in ['true', 'false']:
                                value = value_part.lower() == 'true'
                            elif '.' in value_part:
                                value = float(value_part)
                            else:
                                try:
                                    value = int(value_part)
                                except ValueError:
                                    # If not a number, keep as string
                                    value = value_part
                                
                            self.update_config(**{param: value})
                        except (ValueError, AttributeError) as e:
                            print(f"Error updating config: {e}")
                    else:
                        print("Usage: /set <parameter> <value>")
                    continue
                
                # Generate responses
                print("\nGenerating responses...")
                
                # Only generate vanilla response if enabled
                if self.model_mode in ["vanilla", "both"]:
                    vanilla_response = self.generate_vanilla_response(
                        user_input, 
                        self.gen_config.max_length if self.gen_config else 50
                    )
                    print(f"\nVanilla GPT-2: {vanilla_response}")
                
                # Only generate qGPT2 response if enabled
                if self.model_mode in ["qgpt2", "both"]:
                    qgpt2_response = self.generate_qgpt2_response(
                        user_input, 
                        self.gen_config.max_length if self.gen_config else 50
                    )
                    print(f"\nqGPT2: {qgpt2_response}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
                
            except Exception as e:
                print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="qGPT2 vs GPT-2 Chatbot Demo")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name (gpt2, gpt2-medium, etc.)")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum response length")
    parser.add_argument("--top-k", type=int, default=256, help="Top-k sampling parameter")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling temperature")
    parser.add_argument("--num-candidates", type=int, default=512, help="Number of candidate tokens to consider")
    parser.add_argument("--context-window", type=int, default=512, help="Context window size")
    parser.add_argument("--phrase-window", type=int, default=128, help="Phrase window for coherence scoring")
    parser.add_argument("--min-threshold", type=float, default=0.8, help="Minimum threshold for coherence scores")
    parser.add_argument("--sequence-cache-size", type=int, default=32, help="Size of sequence memory cache")
    parser.add_argument("--max-history-tokens", type=int, default=1024, help="Maximum history tokens to maintain")
    parser.add_argument("--no-coherence", action="store_true", help="Disable coherence scoring")
    parser.add_argument("--no-adaptive-temp", action="store_true", help="Disable adaptive temperature")
    parser.add_argument("--no-sequence-memory", action="store_true", help="Disable sequence memory")
    parser.add_argument("--mode", type=str, choices=["vanilla", "qgpt2", "both"], default="qgpt2",
                       help="Model mode: vanilla (GPT-2 only), qgpt2 (qGPT2 only), or both")
    
    args = parser.parse_args()
    
    chatbot = ChatbotDemo(
        model_name=args.model,
        max_length=args.max_length,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        num_candidates=args.num_candidates,
        context_window=args.context_window,
        phrase_window=args.phrase_window,
        min_threshold=args.min_threshold,
        sequence_cache_size=args.sequence_cache_size,
        max_history_tokens=args.max_history_tokens,
        use_coherence=not args.no_coherence,
        use_adaptive_temp=not args.no_adaptive_temp,
        use_sequence_memory=not args.no_sequence_memory,
        model_mode=args.mode
    )
    
    chatbot.run_chatbot()

if __name__ == "__main__":
    main() 