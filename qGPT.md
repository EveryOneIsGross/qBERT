# qGPT2: Semantically Guided Autoregressive Generation

**qGPT2** is an inference-time steering framework that upgrades standard GPT-2 models. It acts as a "Semantic Co-Pilot" for the language model, forcing it to check the meaning of its predictions against the conversation context before committing to them.

Unlike fine-tuning, which requires training data and compute, qGPT2 modifies the generation process on the fly. It combines the grammatical fluency of **GPT-2** with the semantic understanding of **SentenceTransformers** to produce text that stays on topic, reduces hallucinations, and maintains long-range coherence.

## ⚡ Key Features

*   **Inference-Time Steering:** No training required. The model steers probability distributions (logits) dynamically during generation.
*   **The Co-Pilot Mechanism:**
    *   **GPT-2** proposes a list of grammatically likely next words.
    *   **SentenceTransformer** ranks them based on how well they fit the current topic.
*   **Adaptive Temperature:** The system monitors a "Confusion Metric" in real-time:
    *   *High Semantic Agreement:* Temperature lowers (Model becomes confident/deterministic).
    *   *Low Semantic Agreement:* Temperature rises (Model becomes creative to break out of loops).
*   **Sequence Memory:** A lightweight Retrieval-Augmented Generation (RAG) system that remembers specific phrases from earlier in the chat and boosts related tokens, solving GPT-2's short context amnesia.
*   **A/B Testing Console:** Includes a CLI that runs "Vanilla GPT-2" and "qGPT2" side-by-side for direct comparison.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/qGPT2.git
    cd qGPT2
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch transformers sentence-transformers pydantic
    ```

## 🎮 Interactive Demo

The quickest way to test the semantic guidance is via the included chat interface:

```bash
python CHATgpt.py
```

### CLI Commands
*   `/mode both` - Generate answers from Standard GPT-2 and qGPT2 simultaneously to compare quality.
*   `/mode qgpt2` - Run only the guided model.
*   `/set temperature 0.7` - Adjust sampling parameters on the fly.
*   `/set model gpt2-medium` - Hot-swap the base generator (supports `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`).
*   `/config` - View current settings.

## 💻 Python API Usage

You can use `SemanticGPT2Generator` in your own scripts:

```python
from qGPT2 import SemanticGPT2Generator, GenerationConfig, ModelConfig

# 1. Configure
gen_config = GenerationConfig(
    max_length=100,
    base_temperature=1.0,
    use_coherence_scoring=True,
    device="cuda"
)

model_config = ModelConfig(
    gpt2_model_name="gpt2-medium",
    sentence_transformer_name="all-MiniLM-L6-v2"
)

# 2. Initialize
model = SemanticGPT2Generator(gen_config, model_config)

# 3. Generate
response = model.generate(
    initial_text="The future of AI is",
    num_tokens=50
)
print(response)
```

## 🔍 Technical Architecture

Standard GPT-2 generation relies on $P(w_t | w_{<t})$. qGPT2 alters this formula by injecting an external semantic score:

1.  **Proposal:** GPT-2 calculates logits for the next token, filtering for the Top-$k$ candidates.
2.  **Hypothesis Formation:** The system temporarily appends each candidate to the current context.
    *   *Candidate A:* "The sky is [blue]"
    *   *Candidate B:* "The sky is [falling]"
3.  **Semantic Scoring:** The **SentenceTransformer** encodes these hypotheses and compares them to the context embedding via Cosine Similarity.
4.  **Fusion:**
    $$ Logits_{final} = Logits_{GPT} + (\alpha \times Score_{Semantic}) $$
5.  **Outcome:** If the model is discussing weather, "blue" receives a semantic boost. If the model is discussing panic, "falling" receives the boost.

## 📊 Comparison

| Feature | Vanilla GPT-2 | qGPT2 (Guided) |
| :--- | :--- | :--- |
| **Topic Stability** | Drifts easily (dream-like) | Stays focused on prompt |
| **Context Memory** | Limited to window size | Extended via Sequence Cache |
| **Repetition** | Prone to loops | Mitigated by adaptive temp |
| **Speed** | Very High | Moderate (due to semantic checks) |