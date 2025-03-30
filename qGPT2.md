# qGPT2: Semantically Guided GPT-2 Text Generation

qGPT2 is an experimental text generation system that enhances GPT-2 with semantic coherence guidance mechanisms. It adapts the core semantic guidance principles from the qBERT model but applies them to GPT-2's autoregressive generation.

## Key Features

- **Semantic Coherence Scoring**: Uses sentence transformers to evaluate how well candidate tokens maintain thematic consistency with the context.
- **Adaptive Temperature Sampling**: Dynamically adjusts sampling temperature based on measured coherence, allowing for creative exploration when coherence is high and more focused generation when thematic consistency needs reinforcement.
- **Sequence Memory Cache**: Stores and retrieves previously generated text sequences, boosting the scores of tokens that appeared in semantically similar contexts before.

## How It Works

1. **Base Generation**: Uses GPT-2 to predict the probability distribution for the next token.
2. **Semantic Filtering**: Generates potential next phrases for the top-k candidates and scores them based on semantic similarity to the context.
3. **Coherence-Based Sampling**: Combines GPT-2's probabilities with the semantic coherence scores and samples the next token.
4. **Memory Influence**: Boosts candidates that appear in previously generated, semantically similar text.

## Requirements

- PyTorch
- Transformers (Hugging Face)
- Sentence-Transformers

## Usage

```python
from qGPT2 import SemanticGPT2Generator, GenerationConfig, ModelConfig

# Create configurations
gen_config = GenerationConfig(
    max_length=100,
    top_k=50,
    top_p=0.9,
    use_adaptive_temperature=True,
    use_coherence_scoring=True,
    use_sequence_memory=True
)

model_config = ModelConfig(
    gpt2_model_name="gpt2",  # Can be gpt2, gpt2-medium, gpt2-large, etc.
    sentence_transformer_name="all-MiniLM-L6-v2"
)

# Initialize the model
model = SemanticGPT2Generator(gen_config, model_config)

# Generate text
prompt = "The future of artificial intelligence is"
generated_text = model.generate(prompt, num_tokens=50)
print(generated_text)

# Or stream the generation token by token
for token in model.generate_stream(prompt, num_tokens=50):
    print(token, end="", flush=True)
```

## Adjustable Parameters

- **max_length**: Maximum number of tokens to generate
- **top_k**: Number of highest probability tokens to consider
- **top_p**: Cumulative probability threshold for nucleus sampling (0.0-1.0)
- **base_temperature**: Base sampling temperature (higher = more random)
- **context_window**: Size of the context window for semantic coherence
- **phrase_window**: Number of tokens to consider for phrase construction in coherence scoring

## Mechanism Toggles

You can enable/disable specific mechanisms:

```python
# Disable adaptive temperature
model.config.use_adaptive_temperature = False

# Disable coherence scoring
model.config.use_coherence_scoring = False

# Disable sequence memory
model.config.use_sequence_memory = False
```

## Example

Running the demo script in `qGPT2.py`:

```bash
python qGPT2.py
```

This will generate text for several example prompts, demonstrating the effects of the semantic guidance mechanisms.

## Differences from Standard GPT-2

- **More Coherent Output**: Tends to produce more thematically consistent text compared to standard GPT-2 sampling.
- **Improved Local Context**: Makes better word choices by considering semantic similarity to the context.
- **Memory Across Generations**: Can learn from and be influenced by its previous generations on similar topics.

## License

🤷💅