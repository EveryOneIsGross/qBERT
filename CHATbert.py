import sys
import torch
from typing import Union, Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import json
from pydantic import BaseModel, Field, validator
from puBERT import ParallelBERTGenerator as puBERTGenerator, GenerationConfig as puBERTConfig
from qBERT import ParallelBERTGenerator as qBERTGenerator, GenerationConfig as qBERTConfig
import yaml
from colorama import init, Fore, Back, Style
import time
from transformers import AutoConfig
init(autoreset=True)

class ConversationEntry(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    user_input: str
    model_output: str
    model_type: str
    config: Dict[str, Any]
    tokens_generated: int
    stream_mode: bool
    generation_time: float

class SystemUpdate(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    update_type: str
    previous_value: Any
    new_value: Any
    model_type: str

class Logger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.conversation_log = self.log_dir / "conversation_history.jsonl"
        self.system_log = self.log_dir / "system_updates.jsonl"
        self.context_log = self.log_dir / "context.jsonl"
    
    def log_conversation(self, entry: ConversationEntry):
        with open(self.conversation_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.dict()) + "\n")
    
    def log_system_update(self, update: SystemUpdate):
        with open(self.system_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(update.dict()) + "\n")
            
    def log_context(self, context_data: Dict[str, Any]):
        with open(self.context_log, "a", encoding="utf-8") as f:
            entry = {
                "timestamp": datetime.now().isoformat(),
                **context_data
            }
            f.write(json.dumps(entry) + "\n")

class ModelConfig(BaseModel):
    """BERT model configuration settings"""
    bert_model_name: str = Field(default="bert-base-uncased")
    tokenizer_name: str = Field(default="bert-base-uncased")
    sentence_transformer_name: str = Field(default="all-MiniLM-L6-v2")
    attn_implementation: str = Field(default="eager")
    
    @validator('attn_implementation')
    def validate_attn_impl(cls, v):
        if v not in ['eager', 'sdpa', 'flash_attention_2']:
            raise ValueError("attn_implementation must be eager, sdpa, or flash_attention_2")
        return v

def create_generator(model_type: str = "qbert", 
                    config: Optional[Union[puBERTConfig, qBERTConfig]] = None,
                    model_config: Optional[ModelConfig] = None):
    """Initialize model with config and model settings"""
    # Load config from YAML if exists
    config_path = Path("config/model_config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f)
            
        if model_type.lower() == "pubert":
            config = config or puBERTConfig(**yaml_config.get("pubert", {}))
        else:
            config = config or qBERTConfig(**yaml_config.get("qbert", {}))
    
    # Use provided model_config or create default
    model_config = model_config or ModelConfig()
    
    # Create generator with model config
    if model_type.lower() == "pubert":
        if config is None:
            config = puBERTConfig(
                max_length=1000,
                batch_size=4,
                num_candidates=128,
                embedding_dim=768,
                context_window=32,
                base_temperature=0.5,
                min_threshold=0.8,
                top_k=50,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        return puBERTGenerator(config, model_config=model_config)
    elif model_type.lower() == "qbert":
        if config is None:
            config = qBERTConfig(
                max_length=512,
                batch_size=8,
                num_candidates=128,
                embedding_dim=768,
                context_window=256,
                base_temperature=0.7,
                min_threshold=0.5,
                top_k=32,
                compression_ratio=0.2,
                max_cache_size=16,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        return qBERTGenerator(config, model_config=model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def update_config(generator: Union[puBERTGenerator, qBERTGenerator], param: str, value: str):
    """Update configuration parameter with type checking"""
    current_config = generator.config
    config_dict = vars(current_config)
    
    try:
        # Type conversion based on current parameter type
        current_type = type(config_dict[param])
        if current_type == bool:
            value = value.lower() == 'true'
        else:
            value = current_type(value)
        
        # Create new config with updated parameter
        new_config_dict = vars(current_config).copy()
        new_config_dict[param] = value
        
        # Determine model type and create appropriate config
        if isinstance(generator, puBERTGenerator):
            new_config = puBERTConfig(**new_config_dict)
            return create_generator("pubert", new_config)
        else:
            new_config = qBERTConfig(**new_config_dict)
            return create_generator("qbert", new_config)
        
    except ValueError:
        raise ValueError(f"Invalid value for {param}. Expected type: {current_type.__name__}")
    except KeyError:
        raise KeyError(f"Unknown parameter: {param}")

def main():
    """Initialize model and run CLI interface"""
    num_tokens = 20
    stream_mode = True
    model_type = "qbert"
    
    # Initialize logger and model config
    logger = Logger()
    model_config = ModelConfig()
    
    # ASCII art banner
    banner = f"""
    {Fore.CYAN}╔═══════════════════════════════════════╗
    ║            qBERT Interface            ║
    ╚═══════════════════════════════════════╝{Style.RESET_ALL}
    """
    print(banner)
    
    # Initialize model with config
    print(f"{Fore.YELLOW}Initializing model...{Style.RESET_ALL}")
    generator = create_generator(model_type, model_config=model_config)
    print(f"{Fore.GREEN}Model ready!{Style.RESET_ALL}\n")
    
    # Enhanced help text
    help_text = f"""
    {Fore.CYAN}Commands:{Style.RESET_ALL}
    /help   - Show this help message
    /quit   - Exit the program
    /config - Show current configuration
    /clear  - Clear the screen
    /device - Switch between CPU and GPU (usage: /device cpu|cuda)
    /tokens - Set number of tokens to generate (usage: /tokens number)
    /stream - Toggle streaming mode (currently: {stream_mode})
    /model  - Switch model type (usage: /model pubert|qbert)
    
    {Fore.CYAN}Model Configuration:{Style.RESET_ALL}
    /bert_model <name>     - Change BERT model
    /sentence_model <name> - Change sentence transformer
    /attn_impl <type>     - Change attention (eager/sdpa/flash_attention_2)
    
    {Fore.CYAN}Parameters:{Style.RESET_ALL}
    - max_length      (int)   : Maximum sequence length
    - batch_size      (int)   : Batch size for processing
    - num_candidates  (int)   : Number of candidate tokens
    - embedding_dim   (int)   : Embedding dimension
    - context_window  (int)   : Context window size
    - base_temperature(float) : Temperature for sampling
    - min_threshold   (float) : Minimum threshold for candidates
    - top_k           (int)   : Top-k tokens to consider
    
    {Fore.YELLOW}Note: Update parameters with /<parameter> <value>{Style.RESET_ALL}
    """
    
    # Main interaction loop
    while True:
        try:
            user_input = input(f"\n{Fore.GREEN}> {Style.RESET_ALL}")
            
            if user_input.startswith('/'):
                parts = user_input[1:].split(maxsplit=1)
                command = parts[0].lower()
                
                # Model configuration commands
                if command == 'bert_model':
                    try:
                        if len(parts) != 2:
                            print(f"{Fore.RED}Usage: /bert_model <model_name>{Style.RESET_ALL}")
                            continue
                        model_name = parts[1].strip('"')
                        old_model = model_config.bert_model_name
                        
                        # Preserve existing config settings
                        current_config = {k: v for k, v in vars(generator.config).items() 
                                         if k in ['max_length', 'batch_size', 'num_candidates', 
                                                'embedding_dim', 'context_window', 'base_temperature',
                                                'min_threshold', 'top_k', 'device']}
                        
                        # Create new model config while preserving other settings
                        new_config = ModelConfig(
                            bert_model_name=model_name,
                            tokenizer_name=model_name,
                            sentence_transformer_name=model_config.sentence_transformer_name,
                            attn_implementation=model_config.attn_implementation
                        )
                        
                        # Create new generator with preserved settings
                        if model_type == 'pubert':
                            config = puBERTConfig(**current_config)
                        else:
                            config = qBERTConfig(**current_config)
                            
                        generator = create_generator(model_type, config=config, model_config=new_config)
                        model_config = new_config
                        
                        logger.log_system_update(SystemUpdate(
                            update_type="bert_model",
                            previous_value=old_model,
                            new_value=model_name,
                            model_type=model_type
                        ))
                        print(f"{Fore.GREEN}Switched BERT model to {model_name} with preserved settings{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"{Fore.RED}Error switching BERT model: {str(e)}{Style.RESET_ALL}")
                    continue
                
                elif command == 'sentence_model':
                    try:
                        if len(parts) != 2:
                            print(f"{Fore.RED}Usage: /sentence_model <model_name>{Style.RESET_ALL}")
                            continue
                        model_name = parts[1]
                        old_model = model_config.sentence_transformer_name
                        
                        # Preserve existing config settings
                        current_config = {k: v for k, v in vars(generator.config).items() 
                                         if k in ['max_length', 'batch_size', 'num_candidates', 
                                                'embedding_dim', 'context_window', 'base_temperature',
                                                'min_threshold', 'top_k', 'device']}
                        
                        new_config = ModelConfig(
                            bert_model_name=model_config.bert_model_name,
                            tokenizer_name=model_config.tokenizer_name,
                            sentence_transformer_name=model_name,
                            attn_implementation=model_config.attn_implementation
                        )
                        
                        # Create new generator with preserved settings
                        if model_type == 'pubert':
                            config = puBERTConfig(**current_config)
                        else:
                            config = qBERTConfig(**current_config)
                            
                        generator = create_generator(model_type, config=config, model_config=new_config)
                        model_config = new_config
                        
                        logger.log_system_update(SystemUpdate(
                            update_type="sentence_model",
                            previous_value=old_model,
                            new_value=model_name,
                            model_type=model_type
                        ))
                        print(f"{Fore.GREEN}Switched sentence transformer to {model_name} with preserved settings{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"{Fore.RED}Error switching sentence transformer: {str(e)}{Style.RESET_ALL}")
                    continue
                
                elif command == 'attn_impl':
                    try:
                        if len(parts) != 2:
                            print(f"{Fore.RED}Usage: /attn_impl <eager|sdpa|flash_attention_2>{Style.RESET_ALL}")
                            continue
                        impl_type = parts[1]
                        old_impl = model_config.attn_implementation
                        
                        # Preserve existing config settings
                        current_config = {k: v for k, v in vars(generator.config).items() 
                                         if k in ['max_length', 'batch_size', 'num_candidates', 
                                                'embedding_dim', 'context_window', 'base_temperature',
                                                'min_threshold', 'top_k', 'device']}
                        
                        new_config = ModelConfig(
                            bert_model_name=model_config.bert_model_name,
                            tokenizer_name=model_config.tokenizer_name,
                            sentence_transformer_name=model_config.sentence_transformer_name,
                            attn_implementation=impl_type
                        )
                        
                        # Create new generator with preserved settings
                        if model_type == 'pubert':
                            config = puBERTConfig(**current_config)
                        else:
                            config = qBERTConfig(**current_config)
                            
                        generator = create_generator(model_type, config=config, model_config=new_config)
                        model_config = new_config
                        
                        logger.log_system_update(SystemUpdate(
                            update_type="attn_implementation",
                            previous_value=old_impl,
                            new_value=impl_type,
                            model_type=model_type
                        ))
                        print(f"{Fore.GREEN}Switched attention implementation to {impl_type} with preserved settings{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"{Fore.RED}Error switching attention implementation: {str(e)}{Style.RESET_ALL}")
                    continue

                # Direct parameter update with improved feedback
                if hasattr(generator.config, command):
                    try:
                        if len(parts) != 2:
                            print(f"{Fore.RED}Please provide a value for {command}{Style.RESET_ALL}")
                            continue
                            
                        # Preserve model config settings
                        new_config = ModelConfig(
                            bert_model_name=model_config.bert_model_name,
                            tokenizer_name=model_config.tokenizer_name,
                            sentence_transformer_name=model_config.sentence_transformer_name,
                            attn_implementation=model_config.attn_implementation
                        )
                        
                        old_config = vars(generator.config).copy()
                        generator = update_config(generator, command, parts[1])
                        generator = create_generator(model_type, config=generator.config, model_config=new_config)
                        
                        logger.log_system_update(SystemUpdate(
                            update_type="config",
                            previous_value=old_config[command],
                            new_value=parts[1],
                            model_type=model_type
                        ))
                        print(f"{Fore.GREEN}Updated {command} to {parts[1]} with preserved model settings{Style.RESET_ALL}")
                        continue
                    except (ValueError, KeyError) as e:
                        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
                        continue

            if user_input.lower() == '/quit':
                print("Goodbye!")
                break
                
            elif user_input.lower() == '/help':
                print(help_text.format(stream_mode))
                continue
                
            elif user_input.lower() == '/config':
                print("\nCurrent Configuration:")
                print(f"Model Type: {model_type}")
                for key, value in vars(generator.config).items():
                    print(f"{key}: {value}")
                continue
                
            elif user_input.lower() == '/clear':
                print('\033[2J\033[H')  # Clear screen
                print(banner)
                continue
                
            elif user_input.lower().startswith('/device '):
                try:
                    _, device = user_input.split()
                    if device not in ['cpu', 'cuda']:
                        raise ValueError("Device must be 'cpu' or 'cuda'")
                    if device == 'cuda' and not torch.cuda.is_available():
                        print("CUDA is not available on this system")
                        continue
                    old_device = generator.config.device
                    generator = update_config(generator, 'device', device)
                    logger.log_system_update(SystemUpdate(
                        update_type="device",
                        previous_value=old_device,
                        new_value=device,
                        model_type=model_type
                    ))
                    print(f"Switched to {device.upper()}")
                except Exception as e:
                    print(f"Error switching device: {str(e)}")
                continue
                
            elif user_input.lower().startswith('/tokens '):
                try:
                    _, value = user_input.split()
                    old_tokens = num_tokens
                    num_tokens = int(value)
                    logger.log_system_update(SystemUpdate(
                        update_type="tokens",
                        previous_value=old_tokens,
                        new_value=num_tokens,
                        model_type=model_type
                    ))
                    print(f"Will generate {num_tokens} tokens")
                except ValueError:
                    print("Please provide a valid number")
                continue
                
            elif user_input.lower() == '/stream':
                old_stream = stream_mode
                stream_mode = not stream_mode
                logger.log_system_update(SystemUpdate(
                    update_type="stream_mode",
                    previous_value=old_stream,
                    new_value=stream_mode,
                    model_type=model_type
                ))
                print(f"Streaming mode: {'ON' if stream_mode else 'OFF'}")
                continue
                
            elif user_input.lower().startswith('/model '):
                try:
                    _, new_model_type = user_input.split()
                    if new_model_type.lower() not in ['pubert', 'qbert']:
                        raise ValueError("Model type must be 'puBERT' or 'qBERT'")
                    old_model = model_type
                    model_type = new_model_type.lower()  # Store lowercase version
                    generator = create_generator(model_type)
                    logger.log_system_update(SystemUpdate(
                        update_type="model_type",
                        previous_value=old_model,
                        new_value=model_type,
                        model_type=model_type
                    ))
                    print(f"Switched to {model_type}")
                except Exception as e:
                    print(f"Error switching model: {str(e)}")
                continue
                
            elif not user_input:
                continue
            
            # Generation with improved error handling and feedback
            try:
                model_output = ""
                formatted_input = f"@user: {user_input}\n@bot: "
                start_time = time.time()
                
                if stream_mode:
                    print(f"{Fore.CYAN}Generating response...{Style.RESET_ALL}")
                    for token in generator.generate_stream(formatted_input, num_tokens=num_tokens):
                        sys.stdout.write(token)
                        sys.stdout.flush()
                        model_output += token
                else:
                    model_output = generator.generate(formatted_input, num_tokens=num_tokens)
                    print(model_output)
                
                generation_time = time.time() - start_time
                #print(f"\n{Fore.YELLOW}Generated in {generation_time:.2f}s{Style.RESET_ALL}")
                
                # Log conversation with timing
                logger.log_conversation(ConversationEntry(
                    user_input=user_input,
                    model_output=model_output,
                    model_type=model_type,
                    config=vars(generator.config),
                    tokens_generated=num_tokens,
                    stream_mode=stream_mode,
                    generation_time=generation_time
                ))
                
            except Exception as e:
                print(f"\n{Fore.RED}Error during generation: {str(e)}{Style.RESET_ALL}")
                
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Caught interrupt signal{Style.RESET_ALL}")
            confirm = input(f"\n{Fore.YELLOW}Do you want to quit? (y/n): {Style.RESET_ALL}")
            if confirm.lower() == 'y':
                print(f"{Fore.GREEN}Goodbye!{Style.RESET_ALL}")
                break
            
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1) 