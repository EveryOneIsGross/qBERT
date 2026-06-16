import argparse
import re
import sys
import torch
import yaml
from dataclasses import dataclass
from pathlib import Path
from datasets import load_dataset
from transformers import (
    BertForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


def _load_yaml_defaults(path: str) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    # Flatten one optional nesting level (e.g. a top-level 'trainer:' key)
    if len(data) == 1 and isinstance(next(iter(data.values())), dict):
        data = next(iter(data.values()))
    return {k: v for k, v in data.items() if v is not None}


def parse_args():
    # Pre-parse for --config only, then fold YAML values in as defaults
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None,
                     help="Path to YAML config file (CLI args override file values)")
    known, _ = pre.parse_known_args()

    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--dataset_config_name", type=str, default=None)
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--eval_split", type=str, default=None)
    p.add_argument("--format", type=str, default="sharegpt", choices=["sharegpt", "qa", "text"])
    p.add_argument("--conversation_column", type=str, default="conversations")
    p.add_argument("--role_field", type=str, default="from")
    p.add_argument("--content_field", type=str, default="value")
    p.add_argument("--question_column", type=str, default="question")
    p.add_argument("--answer_column", type=str, default="answer")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--system_role", type=str, default="system")
    p.add_argument("--user_role", type=str, default="human")
    p.add_argument("--assistant_role", type=str, default="gpt")
    p.add_argument("--system_token", type=str, default="[SYSTEM]")
    p.add_argument("--user_token", type=str, default="[USER]")
    p.add_argument("--assistant_token", type=str, default="[ASSISTANT]")
    p.add_argument("--eot_token", type=str, default="[EOT]")
    p.add_argument("--default_system_prompt", type=str, default="")
    p.add_argument("--model_name_or_path", type=str, default="bert-base-cased")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--per_device_eval_batch_size", type=int, default=16)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--data_files", type=str, default=None,
                   help="Direct file path/URL when the dataset repo contains non-data files")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--eval_strategy", type=str, default="steps")
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--mlm_probability", type=float, default=0.15,
                   help="Masking probability for assistant completion tokens.")
    p.add_argument("--context_mlm_probability", type=float, default=0.10,
                   help="Masking probability for user/system content tokens. Lower than "
                        "--mlm_probability to prioritise assistant gradient signal while "
                        "still tuning BERT's representations of the full conversation context.")
    p.add_argument("--assistant_only", action="store_true", default=False,
                   help="Restrict masking to assistant turns only (disables context MLM).")
    # Boundary steering
    p.add_argument("--boundary_mlm_bonus", type=float, default=0.25,
                   help="Extra masking probability added within --boundary_window positions "
                        "of [ASSISTANT] and [EOT] tokens. Creates stronger gradient signal "
                        "at turn-opening and turn-closing positions.")
    p.add_argument("--boundary_window", type=int, default=5,
                   help="Number of token positions on each side of a boundary token "
                        "to apply the masking bonus.")
    p.add_argument("--eot_as_target", action="store_true", default=True,
                   help="Give [EOT] a real label (not -100) when it closes an assistant turn, "
                        "so the model learns to predict end-of-turn rather than just observe it.")
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config file (CLI args override file values)")

    # Apply YAML values as defaults before final parse
    if known.config:
        config_path = Path(known.config)
        if not config_path.exists():
            print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        yaml_defaults = _load_yaml_defaults(str(config_path))
        # Boolean flags need special treatment — store_true args can't use set_defaults
        # with a string from YAML, so convert them explicitly
        bool_flags = {"trust_remote_code", "assistant_only", "eot_as_target"}
        for flag in bool_flags:
            if flag in yaml_defaults:
                yaml_defaults[flag] = str(yaml_defaults[flag]).lower() in ("true", "1", "yes")
        p.set_defaults(**yaml_defaults)
        print(f"Loaded config: {config_path}")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Text builders — same as before, structure tokens inserted as plain strings
# ---------------------------------------------------------------------------

def build_sharegpt_string(example, args):
    convs = example[args.conversation_column]
    parts = []
    sys_tok = args.system_token.strip()
    usr_tok = args.user_token.strip()
    asst_tok = args.assistant_token.strip()
    eot_tok = args.eot_token.strip()
    for turn in convs:
        role = str(turn[args.role_field])
        text = str(turn[args.content_field]).strip()
        if not text:
            continue
        if role == args.system_role and sys_tok:
            parts.append(f"{sys_tok} {text} {eot_tok}")
        elif role == args.user_role:
            parts.append(f"{usr_tok} {text} {eot_tok}")
        elif role == args.assistant_role:
            parts.append(f"{asst_tok} {text} {eot_tok}")
        else:
            parts.append(text)
    return "\n".join(parts)

def build_qa_string(example, args):
    q = str(example[args.question_column]).strip()
    a = str(example[args.answer_column]).strip()
    sys_prompt = args.default_system_prompt.strip()
    eot_tok = args.eot_token.strip()
    chunks = []
    if sys_prompt and args.system_token.strip():
        chunks.append(f"{args.system_token.strip()} {sys_prompt} {eot_tok}")
    chunks.append(f"{args.user_token.strip()} {q} {eot_tok}")
    chunks.append(f"{args.assistant_token.strip()} {a} {eot_tok}")
    return "\n".join(chunks)

def build_text_string(example, args):
    return str(example[args.text_column]).strip()


# ---------------------------------------------------------------------------
# Structure-aware tokenization: build labels with -100 on non-assistant tokens
# ---------------------------------------------------------------------------

def build_token_ids_and_labels(text, tokenizer, max_len, args):
    """
    Splits conversation text on role tokens, encodes each segment, and assigns
    labels = token_ids for assistant turns (compute loss) or -100 everywhere
    else (no loss).

    Frame tokens ([USER], [ASSISTANT], [SYSTEM]) are always -100 — they are
    structural landmarks, not content, and the model should never generate them.

    [EOT] is treated differently: when eot_as_target=True and we are inside an
    assistant turn, [EOT] gets a real label so the model learns to predict the
    end-of-turn boundary. This is the training signal for "I am done responding."
    In all other contexts [EOT] remains -100.
    """
    frame_tokens = [args.assistant_token, args.user_token, args.system_token]
    role_tokens = frame_tokens + ([args.eot_token] if args.eot_token else [])
    pattern = "(" + "|".join(re.escape(t) for t in role_tokens if t) + ")"

    parts = re.split(pattern, text)

    input_ids = []
    labels = []
    is_assistant = []   # True for assistant content positions, False elsewhere
    current_role = None

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part in frame_tokens:
            # Frame tokens: always present, never predicted
            current_role = part
            role_ids = tokenizer.encode(part, add_special_tokens=False)
            input_ids.extend(role_ids)
            labels.extend([-100] * len(role_ids))
            is_assistant.extend([False] * len(role_ids))

        elif args.eot_token and part == args.eot_token:
            # EOT: predict it when closing an assistant turn, mask otherwise
            eot_ids = tokenizer.encode(part, add_special_tokens=False)
            input_ids.extend(eot_ids)
            in_asst = current_role == args.assistant_token
            if in_asst and getattr(args, "eot_as_target", True):
                labels.extend(eot_ids)
                is_assistant.extend([True] * len(eot_ids))
            else:
                labels.extend([-100] * len(eot_ids))
                is_assistant.extend([False] * len(eot_ids))
            current_role = None  # turn has closed

        else:
            content_ids = tokenizer.encode(part, add_special_tokens=False)
            input_ids.extend(content_ids)
            in_asst = current_role == args.assistant_token
            labels.extend(content_ids)      # real labels for all content
            is_assistant.extend([in_asst] * len(content_ids))

    # Truncate
    input_ids  = input_ids[:max_len]
    labels     = labels[:max_len]
    is_assistant = is_assistant[:max_len]
    attention_mask = [1] * len(input_ids)

    # Pad
    pad_len = max_len - len(input_ids)
    input_ids    += [tokenizer.pad_token_id] * pad_len
    labels       += [-100] * pad_len
    is_assistant += [False] * pad_len
    attention_mask += [0] * pad_len

    return {
        "input_ids":    input_ids,
        "labels":       labels,
        "is_assistant": is_assistant,
        "attention_mask": attention_mask,
    }


# ---------------------------------------------------------------------------
# Collator: structure-aware MLM that never masks specials or non-assistant tokens
# ---------------------------------------------------------------------------

@dataclass
class StructureAwareMLMCollator:
    """
    Structure-aware MLM collator with boundary steering.

    Standard behaviour:
    - Never masks frame tokens ([USER], [ASSISTANT], [SYSTEM]) or padding
    - When assistant_only=True, restricts masking to assistant completion regions

    Boundary steering:
    - Adds boundary_mlm_bonus to the masking probability of tokens within
      boundary_window positions of boundary_ids ([ASSISTANT], [EOT]).
      This concentrates gradient signal at turn-opening and turn-closing
      positions — the highest-stakes slots for discourse framing.
    - The bonus is applied after all exclusions, so it only lifts positions
      that are already eligible for masking (never resurrects zeroed specials).
    """
    tokenizer: object
    mlm_probability: float = 0.15          # assistant content masking rate
    context_mlm_probability: float = 0.10  # user/system content masking rate
    special_ids: set = None
    assistant_only: bool = False
    boundary_ids: set = None    # token IDs of [ASSISTANT] and [EOT]
    boundary_mlm_bonus: float = 0.25
    boundary_window: int = 5

    def __call__(self, features):
        input_ids    = torch.tensor([f["input_ids"]    for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
        label_mask   = torch.tensor([f["labels"]       for f in features], dtype=torch.long)
        is_assistant = torch.tensor([f["is_assistant"] for f in features], dtype=torch.bool)

        labels = input_ids.clone()

        # Build probability matrix: assistant positions get mlm_probability,
        # user/system content positions get context_mlm_probability.
        # Frame tokens, padding, and non-content positions get 0.
        probability_matrix = torch.where(
            is_assistant,
            torch.full(input_ids.shape, self.mlm_probability),
            torch.full(input_ids.shape, self.context_mlm_probability),
        )

        # Never mask frame/special tokens
        if self.special_ids:
            for sid in self.special_ids:
                probability_matrix.masked_fill_(input_ids == sid, 0.0)

        # Never mask padding
        probability_matrix.masked_fill_(input_ids == self.tokenizer.pad_token_id, 0.0)

        # Hard assistant-only mode: zero out context positions entirely
        if self.assistant_only:
            probability_matrix.masked_fill_(~is_assistant, 0.0)

        # Zero out positions with no real label (frame tokens already handled above,
        # but this catches anything else left at -100)
        probability_matrix.masked_fill_(label_mask == -100, 0.0)

        # Boundary bonus: boost masking probability near [ASSISTANT] and [EOT].
        # Applied after exclusions so the bonus only affects already-eligible positions.
        if self.boundary_ids and self.boundary_mlm_bonus > 0:
            bonus = torch.zeros_like(probability_matrix)
            W = self.boundary_window
            for bid in self.boundary_ids:
                # find every position in the batch that holds this boundary token
                positions = (input_ids == bid).nonzero(as_tuple=False)  # (N, 2)
                for b, p in positions:
                    lo = max(0, p.item() - W)
                    hi = min(input_ids.size(1), p.item() + W + 1)
                    bonus[b, lo:hi] += self.boundary_mlm_bonus
            # Only apply bonus where masking is already permitted
            eligible = probability_matrix > 0
            probability_matrix = (probability_matrix + bonus).clamp(0.0, 1.0)
            probability_matrix[~eligible] = 0.0  # don't resurrect zeroed positions

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        input_ids[masked_indices] = self.tokenizer.mask_token_id

        # is_assistant is consumed above — don't forward it to the model
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    missing = [f"--{k}" for k, v in [("dataset_name", args.dataset_name), ("output_dir", args.output_dir)] if not v]
    if missing:
        print(f"ERROR: required argument(s) missing: {', '.join(missing)}", file=sys.stderr)
        print("  Provide them on the CLI or set them in your --config YAML.", file=sys.stderr)
        sys.exit(1)

    set_seed(args.seed)

    if args.data_files:
        raw = load_dataset(
            "json",
            data_files={args.train_split: args.data_files},
            trust_remote_code=args.trust_remote_code,
        )
    else:
        raw = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            trust_remote_code=args.trust_remote_code,
            verification_mode="no_checks",
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = BertForMaskedLM.from_pretrained(args.model_name_or_path)

    # Register structure tokens as atomic specials so they're never split
    special_tokens_to_add = [
        t for t in [args.system_token, args.user_token, args.assistant_token, args.eot_token]
        if t and t not in tokenizer.all_special_tokens
    ]
    if special_tokens_to_add:
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens_to_add}
        )
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added {num_added} special tokens: {special_tokens_to_add}")
        print(f"Vocab size now: {len(tokenizer)}")

    # Frame tokens: [USER], [ASSISTANT], [SYSTEM] — structural landmarks, never masked.
    # EOT is handled separately: when eot_as_target=True it can be masked in assistant
    # regions (label is real), so it must NOT appear in special_ids in that case.
    frame_token_strs = [t for t in [args.system_token, args.user_token, args.assistant_token] if t]
    special_ids = set(
        tokenizer.encode(t, add_special_tokens=False)[0]
        for t in frame_token_strs
        if tokenizer.encode(t, add_special_tokens=False)
    )
    special_ids.update(tokenizer.all_special_ids)

    # If EOT is not a training target, block it from masking too
    eot_id = None
    if args.eot_token:
        eot_enc = tokenizer.encode(args.eot_token, add_special_tokens=False)
        if eot_enc:
            eot_id = eot_enc[0]
            if not getattr(args, "eot_as_target", True):
                special_ids.add(eot_id)

    # Boundary token IDs used for masking-probability boosting in the collator
    asst_id = None
    if args.assistant_token:
        asst_enc = tokenizer.encode(args.assistant_token, add_special_tokens=False)
        if asst_enc:
            asst_id = asst_enc[0]
    boundary_ids = {i for i in [asst_id, eot_id] if i is not None}

    def build_text(example):
        if args.format == "sharegpt":
            return {"text": build_sharegpt_string(example, args)}
        if args.format == "qa":
            return {"text": build_qa_string(example, args)}
        return {"text": build_text_string(example, args)}

    if args.train_split not in raw:
        raise ValueError(f"Train split '{args.train_split}' not found in dataset")
    train_ds = raw[args.train_split]
    if args.max_train_samples is not None:
        train_ds = train_ds.shuffle(seed=args.seed).select(
            range(min(args.max_train_samples, len(train_ds)))
        )
    train_ds = train_ds.map(build_text, remove_columns=train_ds.column_names)

    eval_ds = None
    if args.eval_split is not None and args.eval_split in raw:
        eval_ds = raw[args.eval_split]
        if args.max_eval_samples is not None:
            eval_ds = eval_ds.shuffle(seed=args.seed).select(
                range(min(args.max_eval_samples, len(eval_ds)))
            )
        eval_ds = eval_ds.map(build_text, remove_columns=eval_ds.column_names)

    def tokenize(example):
        return build_token_ids_and_labels(
            example["text"], tokenizer, args.max_seq_length, args
        )

    train_dataset = train_ds.map(tokenize, remove_columns=["text"], load_from_cache_file=False)
    eval_dataset = (
        eval_ds.map(tokenize, remove_columns=["text"], load_from_cache_file=False)
        if eval_ds is not None else None
    )

    data_collator = StructureAwareMLMCollator(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
        context_mlm_probability=args.context_mlm_probability,
        special_ids=special_ids,
        assistant_only=args.assistant_only,
        boundary_ids=boundary_ids,
        boundary_mlm_bonus=args.boundary_mlm_bonus,
        boundary_window=args.boundary_window,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        remove_unused_columns=False,  # keep is_assistant so the collator can use it
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy if eval_dataset is not None else "no",
        save_total_limit=args.save_total_limit,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
