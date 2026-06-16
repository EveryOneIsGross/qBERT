# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

qBERT is an experimental framework for forcing BERT (a masked language model) to generate text. BERT was designed for bidirectional understanding, not generation — this project hacks it into a generative model by treating responses as a canvas of `[MASK]` tokens filled iteratively. Each token is scored by both BERT (grammatical confidence) and a SentenceTransformer (semantic coherence), then fused and sampled.

qGPT2 applies the same semantic co-pilot idea to GPT-2: standard autoregressive generation steered at inference time by SentenceTransformer coherence scores — no fine-tuning needed.

## Running

```bash
# Interactive BERT chat
python CHATbert.py

# Autonomous multi-model chat (requires Ollama running)
python autoCHATbert.py

# GPT-2 semantic chat (A/B comparison mode available)
python CHATgpt.py

# Fine-tune BERT on dialogue data
python trainer.py \
  --dataset_name "ae-1/sharegpt_chat" \
  --format sharegpt \
  --output_dir "./qbert-finetuned" \
  --model_name_or_path "bert-base-cased" \
  --num_train_epochs 3

# Fine-tune SentenceTransformer on the same dialogue data
python trainer_st.py --config config/trainer.yaml
```

## Install dependencies

```bash
pip install torch transformers sentence-transformers pydantic pyyaml colorama datasets ollama tiktoken scikit-learn
```

## Architecture

### Core files

| File | Purpose |
|------|---------|
| `qBERT.py` | Main generation engine — `ParallelBERTGenerator` + `GenerationConfig` + `ModelConfig` |
| `qGPT2.py` | GPT-2 variant — `SemanticGPT2Generator` with same coherence mechanics |
| `CHATbert.py` | Interactive CLI wrapping `qBERT.py`; logs to `logs/` |
| `autoCHATbert.py` | Autonomous chat orchestrating qBERT + Ollama with a `TeacherAgent` that tunes hyperparams dynamically |
| `CHATgpt.py` | Interactive CLI wrapping `qGPT2.py` |
| `trainer.py` | Fine-tunes BERT on dialogue datasets (sharegpt/qa/text formats) via HuggingFace `Trainer`; registers structure tokens as atomic specials and uses structure-aware MLM |
| `trainer_st.py` | Fine-tunes a SentenceTransformer on (user, assistant) pairs from the same dataset using `MultipleNegativesRankingLoss`; output replaces the base coherence model in qBERT |

### Generation pipeline (qBERT)

1. Input formatted as `[CLS] [USER] <text> [ASSISTANT] [MASK]×N [SEP]`
2. For each `[MASK]` position: BERT forward pass → top-K candidates
3. Each candidate scored by SentenceTransformer cosine similarity (semantic coherence)
4. Scores fused: either learned gate or heuristic `softmax(bert_logits) * coherence`
5. Nucleus sampling with adaptive temperature, repetition penalty, entity boosting
6. Optional Gibbs refinement: re-mask and resample already-placed tokens given full context

Key `GenerationConfig` parameters: `top_k`, `base_temperature`, `p_nucleus`, `tau_coherence`, `repetition_penalty`, `context_window`, `gibbs_every_M`, `gibbs_span_L`, `use_learned_gate`.

### Memory systems

- **Semantic cache** (`LRU`): compressed embeddings of past conversation turns (linear projection at `compression_ratio`)
- **Sequence cache**: stores generated phrases + embeddings; boosts tokens from semantically similar prior contexts

### autoCHATbert specifics

- `TeacherAgent`: an Ollama LLM that observes generation quality and mutates `GenerationConfig` between turns
- Vector memory stored as pickle files in `data/`; FAISS-style cosine retrieval via sklearn
- Logs structured JSONL to `logs/` (conversation, system updates, responses)

## Chat commands

**CHATbert.py:**
- `/stream` — toggle streaming
- `/tokens <n>` — response length
- `/bert_model <name>` — hot-swap BERT model
- `/sentence_model <name>` — hot-swap semantic model
- `/config` — show current params
- `/<param> <value>` — set any `GenerationConfig` field

**CHATgpt.py:**
- `/mode both|qgpt2` — A/B comparison or guided-only
- `/set <param> <value>` — e.g. `/set temperature 0.7`, `/set model gpt2-medium`
- `/config` — show current params

## Trainer — special token handling

`trainer.py` registers `[USER]`, `[ASSISTANT]`, `[SYSTEM]`, `[EOT]` via `tokenizer.add_special_tokens()` before any tokenization, making them atomic (WordPiece never decomposes them). `model.resize_token_embeddings()` adds fresh embedding rows for each. The pipeline:

1. **`build_token_ids_and_labels()`** — splits conversation text on role tokens using a regex, encodes each segment, assigns `labels = token_ids` for assistant content and `labels = -100` everywhere else (role tokens, user/system turns, padding). Loss is only computed on assistant completions.
2. **`StructureAwareMLMCollator`** — replaces `DataCollatorForLanguageModeling`. Builds the MLM mask probability matrix then zeroes it out for: all special token IDs, padding, and (when `--assistant_only`, the default) any position with `labels == -100`. Only assistant completion tokens are eligible for `[MASK]` replacement.

When loading a fine-tuned model, always load the tokenizer from the same output directory — not from the original HuggingFace checkpoint — or the registered specials and resized embeddings won't match:

```python
# correct
tokenizer = AutoTokenizer.from_pretrained("./qbert-finetuned")
model = BertForMaskedLM.from_pretrained("./qbert-finetuned")
```

Key trainer flags: `--mlm_probability` (default 0.15), `--assistant_only` (default true), `--eot_token` (pass `""` to disable).

## Notes

- `archive/` contains old iterations; treat as read-only history
- `autoCHATbert.py` has a known bug noted in its docstring: config state tracking between turns may be unreliable for the TeacherAgent
- Models default to CUDA if available, CPU otherwise
- No test suite exists; validation is manual via the chat interfaces
