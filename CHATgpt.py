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
                num_candidates=32,
                context_window=5,
                phrase_window=32,
                base_temperature=temperature,
                min_threshold=0.5,
                top_k=top_k,
                top_p=top_p,
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
        self.max_history_tokens = 512  # Cap history to avoid exceeding context limits
        
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
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.gen_config, key):
                setattr(self.gen_config, key, value)
                print(f"Updated {key} to {value}")
    
    def set_model_mode(self, mode):
        """Change the current model mode."""
        valid_modes = ["vanilla", "qgpt2", "both"]
        if mode.lower() in valid_modes:
            self.model_mode = mode.lower()
            print(f"Model mode set to: {self.model_mode}")
            return True
        else:
            print(f"Invalid model mode: {mode}. Valid options are: {', '.join(valid_modes)}")
            return False
    
    def print_current_config(self):
        """Print current configuration settings."""
        print("\n--- Current Configuration ---")
        print(f"Model Mode: {self.model_mode}")
        print(f"Max Length: {self.gen_config.max_length if self.gen_config else 'N/A'}")
        print(f"Top-k: {self.gen_config.top_k if self.gen_config else 'N/A'}")
        print(f"Top-p: {self.gen_config.top_p if self.gen_config else 'N/A'}")
        print(f"Temperature: {self.gen_config.base_temperature if self.gen_config else 'N/A'}")
        
        if self.gen_config:
            print(f"Coherence Scoring: {self.gen_config.use_coherence_scoring}")
            print(f"Adaptive Temperature: {self.gen_config.use_adaptive_temperature}")
            print(f"Sequence Memory: {self.gen_config.use_sequence_memory}")
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
                        param = parts[1]
                        try:
                            # Parse value with appropriate type
                            if parts[2].lower() in ['true', 'false']:
                                value = parts[2].lower() == 'true'
                            elif '.' in parts[2]:
                                value = float(parts[2])
                            else:
                                value = int(parts[2])
                                
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
    parser.add_argument("--max-length", type=int, default=50, help="Maximum response length")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling parameter")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--no-coherence", action="store_true", help="Disable coherence scoring")
    parser.add_argument("--no-adaptive-temp", action="store_true", help="Disable adaptive temperature")
    parser.add_argument("--no-sequence-memory", action="store_true", help="Disable sequence memory")
    parser.add_argument("--mode", type=str, choices=["vanilla", "qgpt2", "both"], default="both",
                       help="Model mode: vanilla (GPT-2 only), qgpt2 (qGPT2 only), or both")
    
    args = parser.parse_args()
    
    chatbot = ChatbotDemo(
        model_name=args.model,
        max_length=args.max_length,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        use_coherence=not args.no_coherence,
        use_adaptive_temp=not args.no_adaptive_temp,
        use_sequence_memory=not args.no_sequence_memory,
        model_mode=args.mode
    )
    
    chatbot.run_chatbot()

if __name__ == "__main__":
    main() 