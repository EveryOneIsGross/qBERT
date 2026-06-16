# qBERT

![](/assets/qbert.png)

qBERT is an experiment in making masked-language models generate conversational
text. It uses BERT's masked-token predictions as the grammatical engine, then
adds semantic scoring, memory, adaptive sampling, and optional denoising to turn
that cloze model into a streaming response generator.

BERT was not built as an autoregressive writer. The current qBERT sampler works
by advancing a generation frontier one token at a time: it masks the current
position, asks a masked LM for likely replacements, reranks those candidates
with a sentence-transformer coherence critic, samples a token, commits it, and
moves on.

## Repository Map

```text
qBERT.py             Core qBERT generation framework
CHATbert.py          Manual terminal chat interface for qBERT
autoCHATbert.py      Autonomous qBERT <-> Ollama conversation loop
trainer.py           Structure-aware BERT MLM fine-tuning
trainer_st.py        SentenceTransformer dialogue-pair fine-tuning
validate_trainer.py  Checkpoint validation helpers
audit_dataset.py     Dataset structure and length audit tool
config/              Runtime, trainer, and prompt YAML files
assets/              Static assets
```

Generated checkpoints, tokenizer files, TensorBoard runs, logs, caches, and
chat history are expected to live under directories such as `qbert-hermes*/`,
`qbert-st-hermes-v4/`, `checkpoints/`, `logs/`, `cache/`, and `data/`.

## Framework Flow

`qBERT.py` exposes three main config/model layers:

- `GenerationConfig`: sampling, coherence, memory, denoise, and device settings.
- `ModelConfig`: BERT model, tokenizer, SentenceTransformer, and attention backend.
- `ParallelBERTGenerator`: the actual generator that owns BERT, semantic scoring,
  memory caches, token priors, sampling, denoise, and backedit logic.

High-level flow:

```text
prompt text
  -> tokenize and initialize a BERT-sized generation sequence
  -> for each generated position:
       mask the frontier position
       run the BERT MLM head
       remove special tokens and apply token priors
       keep top-k candidates
       score candidates with SentenceTransformer coherence
       apply sequence-memory, left-to-right, entity, style, and repetition biases
       update coherence homeostasis
       fuse BERT logits with coherence scores
       apply adaptive temperature and nucleus sampling
       commit the sampled token and stream formatted text
  -> update semantic and sequence memory
  -> optionally run denoise passes over generated tokens
```

The public methods are:

- `generate_stream(initial_text, num_tokens)`: yields formatted draft token text.
- `generate_with_denoise_stream(initial_text, num_tokens)`: yields structured
  events for draft, denoise start, denoise edits, and denoise completion.
- `generate(initial_text, num_tokens)`: returns the final string.

## Core Pieces

### BERT MLM Engine

The generator loads `AutoModelForMaskedLM` and uses BERT's MLM logits at the
current position. Special tokens (`PAD`, `CLS`, `SEP`, `MASK`, `UNK`) are filtered
out before sampling. A simple token prior downweights digit-only and
punctuation-only tokens.

### Semantic Critic

For each candidate token, qBERT builds a short hypothetical continuation and
encodes it with a SentenceTransformer. Candidate embeddings are compared with
the recent context tail, optionally blended with a prompt anchor embedding so the
response does not drift entirely into its own generated text.

Important controls:

- `tau_coherence`: semantic score sharpness.
- `coherence_anchor_alpha`: how strongly the user prompt anchors coherence.
- `coherence_anchor_taper`: starts anchor pressure stronger, then eases it.
- `use_coherence_homeostasis`: dynamically tightens or relaxes semantic pressure
  based on a running coherence EMA.

### Sampling Stack

Candidate scores are shaped by:

- BERT logits.
- Semantic coherence scores.
- Optional learned gate (`use_learned_gate`).
- Sequence-memory boosts.
- Left-to-right tokenization/punctuation bias.
- Entity reuse bias.
- Early-token style penalties for punctuation-heavy starts.
- Sliding-window repetition penalty.
- Adaptive temperature from entropy.
- Nucleus sampling via `p_nucleus`.

### Memory Systems

qBERT keeps two in-process memories:

- `SemanticMemoryCache`: compresses the 4D hidden-state memory matrix and can
  blend prior state back into a future generation with `memory_alpha` and
  `memory_beta`.
- `SequenceMemoryCache`: stores recent generated sequences per prompt key and
  boosts candidates that resemble similar prior continuations.

These memories are runtime-only inside the generator instance; they are not
saved as checkpoint files by `qBERT.py`.

### Denoise and Backedit

The draft generator can be followed by a denoise pass:

- `use_denoise`: enables post-draft denoising.
- `denoise_passes`: number of revision passes.
- `denoise_pattern`: `sequential`, `random`, `entropy`, `span`, or `checkerboard`.
- `denoise_batch_size`: number of positions masked per BERT batch (`0` means all).
- `denoise_accept_margin`: required fused-score improvement before replacing a
  draft token.
- `stream_denoise`: whether denoise edits are emitted as stream events.

If denoise is off, `generate()` can still use the older Gibbs-style `_backedit`
path when both `gibbs_every_M` and `gibbs_span_L` are greater than zero.

The current `qBERT.py` streaming path still commits left-to-right from the active
frontier.

## Installation

Create a virtual environment, then install the pinned requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The checked-in `requirements.txt` targets Python 3.12 and CUDA 12.1 PyTorch
wheels. If you need CPU-only PyTorch or a different CUDA version, install the
matching PyTorch build first, then install the remaining packages.

## Manual Chat

Start the qBERT terminal interface:

```bash
python CHATbert.py
```

`CHATbert.py` wraps each prompt as:

```text
[USER] <your input>
[ASSISTANT]
```

Then it calls `generate_with_denoise_stream()`.

Useful commands:

```text
/help                         show commands
/quit                         exit
/config                       print GenerationConfig
/clear                        clear the terminal
/device cpu|cuda              rebuild on CPU or CUDA
/tokens <n>                   set generated token count
/stream                       toggle token streaming
/denoise                      toggle post-generation denoise
/denoise_passes <n>           set denoise passes
/denoise_pattern <pattern>    sequential|random|entropy|span|checkerboard
/denoise_batch_size <n>       denoise batch size, 0 = all positions
/bert_model <name_or_path>    swap BERT/tokenizer
/sentence_model <name_or_path> swap SentenceTransformer
/attn_impl <type>             eager|sdpa|flash_attention_2
/<config_field> <value>       update any GenerationConfig field
```

Conversation and system updates are appended to `logs/conversation_history.jsonl`
and `logs/system_updates.jsonl`.

## Autonomous Chat

Start the autonomous qBERT/Ollama loop:

```bash
python autoCHATbert.py
```

The loop is:

```text
seed input
  -> qBERT generates a response
  -> an Ollama teacher retrieves related attempts from persistent BM25 memory
  -> the teacher evaluates the qBERT output as a sparse JSON patch
  -> confident teacher feedback mutates qBERT GenerationConfig in memory
  -> Ollama generates a reflection and stores it in hybrid memory
  -> Ollama responds to qBERT using recent history and relevant reflections
  -> the teacher records the post-change reflection/outcome
  -> qBERT receives that response as the next turn
  -> repeat
```

Runtime pieces:

- `OllamaChat`: owns the Ollama client, prompt templates, message history, and
  hybrid reflection memory.
- `TeacherAgent`: asks an Ollama model for sparse tuning suggestions. Current
  tunable fields include `base_temperature`, `top_k`, `context_window`,
  `tau_coherence`, `p_nucleus`, `repetition_penalty`, `repetition_window`,
  `coherence_anchor_alpha`, and the denoise controls.
- `HybridMemoryDB`: combines vector retrieval from `VectorIndex` with persistent
  BM25 lexical search from `PersistentBM25Index`.
- `VectorIndex`: stores reflection embeddings in `data/vector_index.pkl`.
- `PersistentBM25Index`: stores reflection, teacher-attempt, and post-change
  records in `data/hybrid_memory_bm25_index.json`.

`autoCHATbert.py` expects a local Ollama server and the configured models. Current
defaults are loaded from `config/model_config.yaml`: `gemma4:12b` for chat,
`nomic-embed-text` for embeddings, and `http://127.0.0.1:11434` for the Ollama
host when no host is configured. The teacher follows the chat model when
`teacher.use_ollama_model` is true. Prompt templates live in `config/prompts.yaml`.

During the loop:

- `Ctrl+C` toggles pause/resume.
- `Ctrl+D` exits.
- At the seed prompt, `/help`, `/models`, and `/quit` are available before
  starting the loop.
- While paused, commands include `/help`, `/models`, `/ollama <model>`,
  `/ollama_host <host>`, `/device <cpu|cuda>`, `/tokens <n>`, `/stream`,
  `/denoise`, `/denoise_passes <n>`, `/bert_model <name>`,
  `/sentence_model <name>`, and `/<config_field> <value>`.

Teacher updates are applied to the live generator object but are not written back
to YAML automatically. The teacher itself stays alive for the process and also
indexes attempts/reflections into persistent BM25 memory.

## Configuration

Main config files:

- `config/model_config.yaml`: qBERT/model defaults used by `CHATbert.py` and
  `autoCHATbert.py`, plus autonomous chat, Ollama, teacher, and memory defaults.
- `config/prompts.yaml`: Ollama system, reflection, response, and teacher prompts.
- `config/trainer.yaml`: shared dataset/training defaults for `trainer.py` and
  `trainer_st.py`.

`config/model_config.yaml` is split into these runtime sections:

- `model_defaults`: BERT checkpoint, tokenizer, SentenceTransformer, and
  attention implementation.
- `qbert`: `GenerationConfig` fields. `device: null` means use qBERT's automatic
  CUDA/CPU default, and `phrase_window: null` derives from `context_window`.
- `autochat` and `autochat_session`: loop timing, context limits, token count,
  streaming, and model type.
- `ollama`: chat model, embedding model, host, history length, and history path.
- `teacher`: teacher model policy, confidence threshold, short in-process history
  length, and related-attempt retrieval count.
- `memory`: vector/BM25 file paths, embedding dimension, backfill limit, and
  hybrid retrieval weights.

Key `GenerationConfig` fields:

| Field | Default | Purpose |
| --- | ---: | --- |
| `max_length` | 512 | Maximum requested generation length before BERT cap |
| `top_k` | 32 | Candidate tokens kept from MLM logits |
| `base_temperature` | 0.7 | Sampling temperature |
| `p_nucleus` | 0.9 | Top-p sampling threshold |
| `context_window` | 256 | Window used by bidirectional memory attention |
| `phrase_window` | `context_window / 2` | Context tail for semantic scoring |
| `tau_coherence` | 1.7 | Coherence score shaping |
| `repetition_penalty` | 0.8 | Downweight repeated recent tokens |
| `repetition_window` | 64 | Repetition lookback |
| `coherence_anchor_alpha` | 0.5 | Prompt-anchor strength |
| `use_coherence_homeostasis` | true | Dynamic coherence pressure |
| `use_semantic_memory` | true | Enable compressed hidden-state memory |
| `use_sequence_memory` | true | Enable recent-sequence retrieval boosts |
| `use_denoise` | false | Enable post-draft denoise |
| `denoise_pattern` | sequential | Denoise position order |
| `denoise_batch_size` | 32 | Denoise BERT batch size |
| `gibbs_every_M` | 0 | Backedit interval, disabled at 0 |
| `gibbs_span_L` | 0 | Backedit span length |

`min_threshold` remains present in `GenerationConfig` for compatibility, but the
current sampler and teacher prompt do not use it as an active tuning knob.

BERT has a 512-token sequence limit. The generator caps the active prompt and
generation canvas to fit that window, leaving room for special tokens.

## Training Workflow

Audit a dataset before training:

```powershell
python audit_dataset.py `
  --dataset_name json `
  --data_files "hf://datasets/NousResearch/Hermes-3-Dataset/hermes-3-dataset.jsonl" `
  --format sharegpt `
  --model_name_or_path bert-base-cased
```

Fine-tune the BERT masked LM:

```bash
python trainer.py --config config/trainer.yaml
```

The trainer supports `sharegpt`, `qa`, and `text` formats. It inserts structure
tokens such as `[SYSTEM]`, `[USER]`, `[ASSISTANT]`, and `[EOT]`, then applies a
structure-aware MLM collator with:

- separate assistant/context masking rates,
- optional assistant-only masking,
- boundary masking around `[ASSISTANT]` and `[EOT]`,
- optional `[EOT]` prediction as a real target.

Fine-tune the SentenceTransformer coherence model:

```bash
python trainer_st.py --config config/trainer.yaml
```

Validate a fine-tuned checkpoint:

```bash
python validate_trainer.py --model_dir ./qbert-hermes-v4
```

Then load trained artifacts in `CHATbert.py`:

```text
/bert_model ./qbert-hermes-v4
/sentence_model ./qbert-st-hermes-v4
```

## Programmatic Use

```python
from qBERT import GenerationConfig, ModelConfig, ParallelBERTGenerator

config = GenerationConfig(max_length=128, top_k=32, use_denoise=False)
model_config = ModelConfig(
    bert_model_name="bert-base-cased",
    tokenizer_name="bert-base-cased",
    sentence_transformer_name="all-MiniLM-L6-v2",
)

generator = ParallelBERTGenerator(config, model_config=model_config)
text = generator.generate("[USER] hello\n[ASSISTANT] ", num_tokens=64)
print(text)
```

For live streaming:

```python
for event in generator.generate_with_denoise_stream(prompt, num_tokens=64):
    if event["phase"] == "draft":
        print(event["token"], end="", flush=True)
```

## Current Limitations

- The generator is experimental and intentionally nonstandard; output quality
  depends heavily on the BERT checkpoint, SentenceTransformer, and sampling
  settings.
- qBERT's own semantic and sequence caches are a work-in-process. `autoCHATbert.py`
  adds persistent reflection/teacher memory on top, but it does not serialize the
  generator's internal caches yet.
- Teacher updates change the live generator only; the tuned values are not
  written back to `config/model_config.yaml`.
- Model swaps rebuild the generator and can be slow.
- Denoise previews in the chat interfaces are displayed separately from the
  draft response stored in conversation logs.
