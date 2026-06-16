import sys
import shutil
import torch
from typing import Union, Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import json
from pydantic import BaseModel, Field, validator
from qBERT import ParallelBERTGenerator as qBERTGenerator, GenerationConfig as qBERTConfig
import yaml
from colorama import init, Fore, Back, Style
import time

import warnings
warnings.filterwarnings("ignore")

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

init(autoreset=True)

ROLE_USER = "[USER]"
ROLE_ASSISTANT = "[ASSISTANT]"
MODEL_CONFIG_PATH = Path("config/model_config.yaml")


def load_model_config(path: Path = MODEL_CONFIG_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _config_section(config_data: Dict[str, Any], section: str) -> Dict[str, Any]:
    values = (config_data or {}).get(section, {})
    return values if isinstance(values, dict) else {}


def _generation_config_kwargs(config_data: Dict[str, Any]) -> Dict[str, Any]:
    section = _config_section(config_data, "qbert")
    valid_fields = set(getattr(qBERTConfig, "__dataclass_fields__", {}).keys())
    kwargs: Dict[str, Any] = {}
    for key, value in section.items():
        if key not in valid_fields or value is None:
            continue
        if key == "device" and str(value).strip().lower() in {"", "auto"}:
            continue
        kwargs[key] = value
    return kwargs


def create_qbert_config(config_data: Optional[Dict[str, Any]] = None) -> qBERTConfig:
    return qBERTConfig(**_generation_config_kwargs(config_data or load_model_config()))


def _visual_line_count(text: str, width: int) -> int:
    width = max(20, width)
    lines = text.splitlines() or [""]
    return sum(max(1, (len(line) + width - 1) // width) for line in lines)


def redraw_denoise_preview(previous_lines: int, text: str, status: str) -> int:
    if previous_lines > 0:
        sys.stdout.write(f"\033[{previous_lines}A\033[J")
    else:
        sys.stdout.write("\n")

    title = "Denoise preview (not saved to chat history)"
    plain = f"{title}\n{status}\n{text}"
    rendered = (
        f"{Fore.CYAN}{title}{Style.RESET_ALL}\n"
        f"{Fore.MAGENTA}{status}{Style.RESET_ALL}\n"
        f"{text}"
    )
    sys.stdout.write(rendered + "\n")
    sys.stdout.flush()
    cols = shutil.get_terminal_size((80, 24)).columns
    return _visual_line_count(plain, cols)


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
                **context_data,
            }
            f.write(json.dumps(entry) + "\n")


class ModelConfig(BaseModel):
    bert_model_name: str = Field(default="bert-base-cased")
    tokenizer_name: str = Field(default="bert-base-cased")
    sentence_transformer_name: str = Field(default="all-MiniLM-L6-v2")
    attn_implementation: str = Field(default="eager")

    @validator("attn_implementation")
    def validate_attn_impl(cls, v):
        if v not in ["eager", "sdpa", "flash_attention_2"]:
            raise ValueError("attn_implementation must be eager, sdpa, or flash_attention_2")
        return v


def _model_config_kwargs(config_data: Dict[str, Any]) -> Dict[str, Any]:
    defaults = _config_section(config_data, "model_defaults")
    aliases = {
        "bert_model_name": ("bert_model_name", "bert_model"),
        "tokenizer_name": ("tokenizer_name", "tokenizer"),
        "sentence_transformer_name": ("sentence_transformer_name", "sentence_transformer"),
        "attn_implementation": ("attn_implementation", "attention_implementation"),
    }
    kwargs: Dict[str, Any] = {}
    for target, names in aliases.items():
        for name in names:
            value = defaults.get(name)
            if value is not None and value != "":
                kwargs[target] = value
                break
    return kwargs


def create_model_config(config_data: Optional[Dict[str, Any]] = None) -> ModelConfig:
    return ModelConfig(**_model_config_kwargs(config_data or load_model_config()))


def create_generator(
    model_type: str = "qbert",
    config: Optional[qBERTConfig] = None,
    model_config: Optional[ModelConfig] = None,
):
    yaml_config = load_model_config()
    config = config or create_qbert_config(yaml_config)
    model_config = model_config or create_model_config(yaml_config)
    return qBERTGenerator(config, model_config=model_config)


def update_config(generator: qBERTGenerator, param: str, value: str):
    current_config = generator.config
    config_dict = vars(current_config)

    try:
        current_type = type(config_dict[param])
        if current_type == bool:
            value = value.lower() == "true"
        else:
            value = current_type(value)

        new_config_dict = vars(current_config).copy()
        new_config_dict[param] = value
        new_config = qBERTConfig(**new_config_dict)
        return create_generator("qbert", new_config)
    except ValueError:
        raise ValueError(f"Invalid value for {param}. Expected type: {current_type.__name__}")
    except KeyError:
        raise KeyError(f"Unknown parameter: {param}")


def main():
    num_tokens = 256
    stream_mode = True
    model_type = "qbert"

    logger = Logger()
    model_config = create_model_config()

    banner = f"""
    {Fore.CYAN}╔═══════════════════════════════════════╗
    ║            qBERT Interface            ║
    ╚═══════════════════════════════════════╝{Style.RESET_ALL}
    """
    print(banner)

    print(f"{Fore.YELLOW}Initializing model...{Style.RESET_ALL}")
    generator = create_generator(model_type, model_config=model_config)
    print(f"{Fore.GREEN}Model ready!{Style.RESET_ALL}\n")

    help_text = f"""
    {Fore.CYAN}Commands:{Style.RESET_ALL}
    /help   - Show this help message
    /quit   - Exit the program
    /config - Show current configuration
    /clear  - Clear the screen
    /device - Switch between CPU and GPU (usage: /device cpu|cuda)
    /tokens - Set number of tokens to generate (usage: /tokens number)
    /stream - Toggle streaming mode (currently: {stream_mode})
    /denoise - Toggle post-generation denoise pass
    /denoise_passes <number> - Set denoise passes
    /denoise_pattern <pattern> - Set noise pattern: sequential|random|entropy|span|checkerboard
    /denoise_batch_size <n> - Set BERT batch size for denoise (0 = all at once)

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
    - denoise_top_k   (int)   : Top-k tokens for denoise replacements
    - denoise_accept_margin(float): Required score improvement for edits

    {Fore.YELLOW}Note: Update parameters with /<parameter> <value>{Style.RESET_ALL}
    """

    while True:
        try:
            user_input = input(f"\n{Fore.GREEN}> {Style.RESET_ALL}")

            if user_input.startswith("/"):
                parts = user_input[1:].split(maxsplit=1)
                command = parts[0].lower()

                if command == "bert_model":
                    try:
                        if len(parts) != 2:
                            print(f"{Fore.RED}Usage: /bert_model <model_name>{Style.RESET_ALL}")
                            continue
                        model_name = parts[1].strip('"')
                        old_model = model_config.bert_model_name

                        current_config = vars(generator.config).copy()

                        new_config = ModelConfig(
                            bert_model_name=model_name,
                            tokenizer_name=model_name,
                            sentence_transformer_name=model_config.sentence_transformer_name,
                            attn_implementation=model_config.attn_implementation,
                        )

                        config = qBERTConfig(**current_config)
                        generator = create_generator(model_type, config=config, model_config=new_config)
                        model_config = new_config

                        logger.log_system_update(
                            SystemUpdate(
                                update_type="bert_model",
                                previous_value=old_model,
                                new_value=model_name,
                                model_type=model_type,
                            )
                        )
                        print(
                            f"{Fore.GREEN}Switched BERT model to {model_name} with preserved settings{Style.RESET_ALL}"
                        )
                    except Exception as e:
                        print(f"{Fore.RED}Error switching BERT model: {str(e)}{Style.RESET_ALL}")
                    continue

                elif command == "sentence_model":
                    try:
                        if len(parts) != 2:
                            print(f"{Fore.RED}Usage: /sentence_model <model_name>{Style.RESET_ALL}")
                            continue
                        model_name = parts[1]
                        old_model = model_config.sentence_transformer_name

                        current_config = vars(generator.config).copy()

                        new_config = ModelConfig(
                            bert_model_name=model_config.bert_model_name,
                            tokenizer_name=model_config.tokenizer_name,
                            sentence_transformer_name=model_name,
                            attn_implementation=model_config.attn_implementation,
                        )

                        config = qBERTConfig(**current_config)
                        generator = create_generator(model_type, config=config, model_config=new_config)
                        model_config = new_config

                        logger.log_system_update(
                            SystemUpdate(
                                update_type="sentence_model",
                                previous_value=old_model,
                                new_value=model_name,
                                model_type=model_type,
                            )
                        )
                        print(
                            f"{Fore.GREEN}Switched sentence transformer to {model_name} with preserved settings{Style.RESET_ALL}"
                        )
                    except Exception as e:
                        print(f"{Fore.RED}Error switching sentence transformer: {str(e)}{Style.RESET_ALL}")
                    continue

                elif command == "attn_impl":
                    try:
                        if len(parts) != 2:
                            print(
                                f"{Fore.RED}Usage: /attn_impl <eager|sdpa|flash_attention_2>{Style.RESET_ALL}"
                            )
                            continue
                        impl_type = parts[1]
                        old_impl = model_config.attn_implementation

                        current_config = vars(generator.config).copy()

                        new_config = ModelConfig(
                            bert_model_name=model_config.bert_model_name,
                            tokenizer_name=model_config.tokenizer_name,
                            sentence_transformer_name=model_config.sentence_transformer_name,
                            attn_implementation=impl_type,
                        )

                        config = qBERTConfig(**current_config)
                        generator = create_generator(model_type, config=config, model_config=new_config)
                        model_config = new_config

                        logger.log_system_update(
                            SystemUpdate(
                                update_type="attn_implementation",
                                previous_value=old_impl,
                                new_value=impl_type,
                                model_type=model_type,
                            )
                        )
                        print(
                            f"{Fore.GREEN}Switched attention implementation to {impl_type} with preserved settings{Style.RESET_ALL}"
                        )
                    except Exception as e:
                        print(f"{Fore.RED}Error switching attention implementation: {str(e)}{Style.RESET_ALL}")
                    continue

                if hasattr(generator.config, command):
                    try:
                        if len(parts) != 2:
                            print(f"{Fore.RED}Please provide a value for {command}{Style.RESET_ALL}")
                            continue

                        new_config = ModelConfig(
                            bert_model_name=model_config.bert_model_name,
                            tokenizer_name=model_config.tokenizer_name,
                            sentence_transformer_name=model_config.sentence_transformer_name,
                            attn_implementation=model_config.attn_implementation,
                        )

                        old_config = vars(generator.config).copy()
                        generator = update_config(generator, command, parts[1])
                        generator = create_generator(model_type, config=generator.config, model_config=new_config)

                        logger.log_system_update(
                            SystemUpdate(
                                update_type="config",
                                previous_value=old_config[command],
                                new_value=parts[1],
                                model_type=model_type,
                            )
                        )
                        print(
                            f"{Fore.GREEN}Updated {command} to {parts[1]} with preserved model settings{Style.RESET_ALL}"
                        )
                        continue
                    except (ValueError, KeyError) as e:
                        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
                        continue

            if user_input.lower() == "/quit":
                print("Goodbye!")
                break

            elif user_input.lower() == "/help":
                print(help_text.format(stream_mode))
                continue

            elif user_input.lower() == "/config":
                print("\nCurrent Configuration:")
                print(f"Model Type: {model_type}")
                for key, value in vars(generator.config).items():
                    print(f"{key}: {value}")
                continue

            elif user_input.lower() == "/clear":
                print("\033[2J\033[H")
                print(banner)
                continue

            elif user_input.lower().startswith("/device "):
                try:
                    _, device = user_input.split()
                    if device not in ["cpu", "cuda"]:
                        raise ValueError("Device must be 'cpu' or 'cuda'")
                    if device == "cuda" and not torch.cuda.is_available():
                        print("CUDA is not available on this system")
                        continue
                    old_device = generator.config.device
                    generator = update_config(generator, "device", device)
                    logger.log_system_update(
                        SystemUpdate(
                            update_type="device",
                            previous_value=old_device,
                            new_value=device,
                            model_type=model_type,
                        )
                    )
                    print(f"Switched to {device.upper()}")
                except Exception as e:
                    print(f"Error switching device: {str(e)}")
                continue

            elif user_input.lower().startswith("/tokens "):
                try:
                    _, value = user_input.split()
                    old_tokens = num_tokens
                    num_tokens = int(value)
                    logger.log_system_update(
                        SystemUpdate(
                            update_type="tokens",
                            previous_value=old_tokens,
                            new_value=num_tokens,
                            model_type=model_type,
                        )
                    )
                    print(f"Will generate {num_tokens} tokens")
                except ValueError:
                    print("Please provide a valid number")
                continue

            elif user_input.lower() == "/stream":
                old_stream = stream_mode
                stream_mode = not stream_mode
                logger.log_system_update(
                    SystemUpdate(
                        update_type="stream_mode",
                        previous_value=old_stream,
                        new_value=stream_mode,
                        model_type=model_type,
                    )
                )
                print(f"Streaming mode: {'ON' if stream_mode else 'OFF'}")
                continue

            elif user_input.lower() == "/denoise":
                old_denoise = generator.config.use_denoise
                generator.config.use_denoise = not old_denoise
                logger.log_system_update(
                    SystemUpdate(
                        update_type="use_denoise",
                        previous_value=old_denoise,
                        new_value=generator.config.use_denoise,
                        model_type=model_type,
                    )
                )
                print(f"Denoise: {'ON' if generator.config.use_denoise else 'OFF'}")
                continue

            elif user_input.lower().startswith("/denoise_passes "):
                try:
                    _, value = user_input.split()
                    old_passes = generator.config.denoise_passes
                    generator.config.denoise_passes = max(0, int(value))
                    logger.log_system_update(
                        SystemUpdate(
                            update_type="denoise_passes",
                            previous_value=old_passes,
                            new_value=generator.config.denoise_passes,
                            model_type=model_type,
                        )
                    )
                    print(f"Denoise passes: {generator.config.denoise_passes}")
                except ValueError:
                    print("Please provide a valid number")
                continue

            elif user_input.lower().startswith("/denoise_pattern "):
                valid_patterns = {"sequential", "random", "entropy", "span", "checkerboard"}
                try:
                    _, value = user_input.split()
                    value = value.lower()
                    if value not in valid_patterns:
                        print(f"Unknown pattern. Choose from: {', '.join(sorted(valid_patterns))}")
                    else:
                        old_pattern = generator.config.denoise_pattern
                        generator.config.denoise_pattern = value
                        logger.log_system_update(
                            SystemUpdate(
                                update_type="denoise_pattern",
                                previous_value=old_pattern,
                                new_value=value,
                                model_type=model_type,
                            )
                        )
                        print(f"Denoise pattern: {value}")
                except ValueError:
                    print("Usage: /denoise_pattern <sequential|random|entropy|span|checkerboard>")
                continue

            elif user_input.lower().startswith("/denoise_batch_size "):
                try:
                    _, value = user_input.split()
                    old_bs = generator.config.denoise_batch_size
                    generator.config.denoise_batch_size = max(0, int(value))
                    logger.log_system_update(
                        SystemUpdate(
                            update_type="denoise_batch_size",
                            previous_value=old_bs,
                            new_value=generator.config.denoise_batch_size,
                            model_type=model_type,
                        )
                    )
                    print(f"Denoise batch size: {generator.config.denoise_batch_size} (0 = all at once)")
                except ValueError:
                    print("Please provide a valid integer")
                continue

            elif not user_input:
                continue

            try:
                model_output = ""
                formatted_input = f"{ROLE_USER} {user_input}\n{ROLE_ASSISTANT} "
                start_time = time.time()

                if stream_mode:
                    print(f"{Fore.CYAN}Generating response...{Style.RESET_ALL}")
                    denoise_started = False
                    denoise_edits = 0
                    denoise_preview = ""
                    denoise_preview_lines = 0

                    for event in generator.generate_with_denoise_stream(formatted_input, num_tokens=num_tokens):
                        phase = event.get("phase")
                        if phase == "draft":
                            sys.stdout.write(event["token"])
                            sys.stdout.flush()
                            model_output = event["text"]
                        elif phase == "denoise_start":
                            denoise_started = True
                            denoise_preview_lines = redraw_denoise_preview(
                                denoise_preview_lines,
                                model_output,
                                "starting from streamed draft",
                            )
                        elif phase == "denoise":
                            denoise_edits += 1
                            denoise_preview = event["text"]
                            denoise_preview_lines = redraw_denoise_preview(
                                denoise_preview_lines,
                                denoise_preview,
                                (
                                    f"pass {event['pass']} edit {denoise_edits}: "
                                    f"{event.get('old_token', '')!r} -> {event.get('new_token', '')!r}"
                                ),
                            )
                        elif phase == "denoise_complete":
                            denoise_preview = event["text"]
                            if denoise_started:
                                if denoise_edits or event.get("edits", 0):
                                    denoise_preview_lines = redraw_denoise_preview(
                                        denoise_preview_lines,
                                        denoise_preview,
                                        f"complete: {event.get('edits', denoise_edits)} edits accepted",
                                    )
                                else:
                                    denoise_preview_lines = redraw_denoise_preview(
                                        denoise_preview_lines,
                                        model_output,
                                        "complete: no edits accepted",
                                    )
                else:
                    if generator.config.use_denoise:
                        denoise_preview = ""
                        denoise_edits = 0
                        for event in generator.generate_with_denoise_stream(formatted_input, num_tokens=num_tokens):
                            phase = event.get("phase")
                            if phase == "draft":
                                model_output = event["text"]
                            elif phase == "denoise":
                                denoise_edits += 1
                                denoise_preview = event["text"]
                            elif phase == "denoise_complete":
                                denoise_preview = event["text"]
                        print(model_output)
                        if denoise_edits:
                            print(f"{Fore.CYAN}Denoise preview (not saved to history):{Style.RESET_ALL}")
                            print(denoise_preview)
                    else:
                        model_output = generator.generate(formatted_input, num_tokens=num_tokens)
                        print(model_output)

                generation_time = time.time() - start_time

                logger.log_conversation(
                    ConversationEntry(
                        user_input=user_input,
                        model_output=model_output,
                        model_type=model_type,
                        config=vars(generator.config),
                        tokens_generated=num_tokens,
                        stream_mode=stream_mode,
                        generation_time=generation_time,
                    )
                )

            except Exception as e:
                print(f"\n{Fore.RED}Error during generation: {str(e)}{Style.RESET_ALL}")

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Caught interrupt signal{Style.RESET_ALL}")
            confirm = input(f"\n{Fore.YELLOW}Do you want to quit? (y/n): {Style.RESET_ALL}")
            if confirm.lower() == "y":
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
