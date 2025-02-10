<div align="center">
<img src="assets/qbert.png" alt="qBERT Logo" width="800">
</div>

# qBERT Framework

*chat with your base bert model*

qBERT is an experimental neural text generation framework that explores coherent text generation through bidirectional context processing and semantic guidance. qBERT leverages BERT's bidirectional understanding while adding embedding mechanisms for controlled generation. 

## Core Innovations

- **Bidirectional Contextual Attention**: Processes both forward and backward context through dual-direction projection layers

- **Semantic Coherence Scoring**: Guides token selection using sentence transformer similarity metrics
- **Adaptive Temperature Control**: Dynamically adjusts sampling temperature based on coherence scores
- **Semantic Memory Cache**: Maintains thematic consistency through compressed semantic representations
- **Cross-Attention Guidance**: Binds generation to original context through position-aware attention
- **4D Matrix Processing**: Experimental batch processing structure for improved generation efficiency

## Framework Components

### Core Models
- **qBERT**: Main bidirectional generation model with semantic guidance
- **puBERT**: Parallel/Unidirectional variant for comparison studies (the pubescent version of qBERT)

### Interfaces
- **CHATbert**: Interactive chat interface with base BERT model control
- **autoCHATbert**: Autonomous chat system with:
  - Semantic reflection capabilities
  - Vector-based memory search
  - Multi-model conversation orchestration

## Quick Start

## Configuration

### Model Configuration
- Located in `config/model_config.yaml`
- Defines model architectures and parameters
- Supports multiple BERT and sentence transformer models

### Prompts Configuration
- Located in `config/prompts.yaml`
- Contains system and reflection prompts for autonomous chat
- Customizable conversation templates
- For defining the autonomous chat system, use the `autoCHATbert.py` script

## Dependencies

- PyTorch
- Transformers (Hugging Face)
- Sentence-Transformers
- Pydantic
- Ollama (for autonomous chat)
- Colorama (CLI interface)
- tiktoken (token counting)

## Architecture Details

### Base Models
- BERT (bert-base-uncased/cased) (Supports other models, fuck around and find out which is the most coherent)
- Sentence Transformer (all-MiniLM-L6-v2)

### Custom Components
- Bidirectional Attention Layers (`forward_proj`, `backward_proj`, `combine_proj`)
- Semantic Coherence Scoring System
- Semantic Memory Cache with Compression
- 4D Matrix for Batch Processing
- Adaptive Temperature Sampling

## Hyperparameters

```
> /config

Current Configuration:
Model Type: qbert
max_length: 512
batch_size: 8
num_candidates: 128
embedding_dim: 768
context_window: 256
base_temperature: 0.7
min_threshold: 0.5
top_k: 32
compression_ratio: 0.2
max_cache_size: 16
device: cuda

> /help

    Commands:
    /help   - Show this help message
    /quit   - Exit the program
    /config - Show current configuration
    /clear  - Clear the screen
    /device - Switch between CPU and GPU (usage: /device cpu|cuda)
    /tokens - Set number of tokens to generate (usage: /tokens number)
    /stream - Toggle streaming mode (currently: True)
    /model  - Switch model type (usage: /model pubert|qbert)

    Model Configuration:
    /bert_model <name>     - Change BERT model
    /sentence_model <name> - Change sentence transformer
    /attn_impl <type>     - Change attention (eager/sdpa/flash_attention_2)

    Parameters:
    - max_length      (int)   : Maximum sequence length
    - batch_size      (int)   : Batch size for processing
    - num_candidates  (int)   : Number of candidate tokens
    - embedding_dim   (int)   : Embedding dimension
    - context_window  (int)   : Context window size
    - base_temperature(float) : Temperature for sampling
    - min_threshold   (float) : Minimum threshold for candidates
    - top_k           (int)   : Top-k tokens to consider

    Note: Update parameters with /<parameter> <value>


> /bert_model Twitter/TwHIN-BERT-base

>
```

## CLI Features

```bash
python CHATbert.py  # Interactive chat interface
python autoCHATbert.py  # Autonomous chat system
```

### Commands
- `/model <pubert|qbert>` - Switch models
- `/config` - View current configuration
- `/device <cpu|cuda>` - Switch compute device
- `/stream` - Toggle token streaming
- `/help` - Show all commands
- `/tokens <number>` - Set number of tokens to generate (if generations are too long tensors will malform)

## Logging

- Conversation history: `logs/conversation_history.jsonl`
- System updates: `logs/system_updates.jsonl`
- Context tracking: `logs/context.jsonl`

## Development Status

qBERT is an experimental framework exploring novel approaches to coherent text generation. The architecture is under active development, and results should be considered preliminary.

## License

MIT

