#!/usr/bin/env python3
"""
trainer_st.py

Fine-tunes a SentenceTransformer on (user, assistant) dialogue pairs extracted
from the same dataset and config used by trainer.py.

Loss: MultipleNegativesRankingLoss — each (anchor, positive) pair acts as a
negative for every other pair in the batch. Larger batches give more negatives
and stronger training signal. Aim for st_batch_size >= 32.

After training, load the output directory in qBERT.py instead of the base
model to get coherence scores calibrated to dialogue structure.

Usage:
    python trainer_st.py --config config/trainer.yaml

    python trainer_st.py \\
      --dataset_name "json" \\
      --data_files "hf://datasets/NousResearch/Hermes-3-Dataset/hermes-3-dataset.jsonl" \\
      --sentence_model_name_or_path "all-MiniLM-L6-v2" \\
      --st_output_dir "./qbert-st-hermes-v3" \\
      --st_num_train_epochs 1 \\
      --st_batch_size 64
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator


# Keys that are booleans in YAML but may be parsed as strings
_BOOL_FLAGS = {"trust_remote_code", "include_system_in_anchor", "add_role_prefixes"}


def _str_to_bool(v):
    """argparse type helper: accepts true/false/1/0/yes/no strings."""
    return str(v).lower() not in ("false", "0", "no", "off")


def _load_yaml_defaults(path: str) -> Dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    # Unwrap one optional nesting level
    if len(data) == 1 and isinstance(next(iter(data.values())), dict):
        data = next(iter(data.values()))
    return {k: v for k, v in data.items() if v is not None}


def parse_args():
    # ── Pre-parse: --config only ──────────────────────────────────────────────
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    known, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(
        description="Fine-tune SentenceTransformer on dialogue pairs"
    )
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config (CLI flags override)")

    # ── Dataset (shared with trainer.py) ─────────────────────────────────────
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--dataset_config_name", type=str, default=None)
    p.add_argument("--data_files", type=str, default=None,
                   help="Direct file path/URL (use when the repo contains non-data files)")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--max_train_samples", type=int, default=20000,
                   help="Examples to load. 0 = full dataset.")

    # ── Format (shared with trainer.py) ──────────────────────────────────────
    p.add_argument("--format", type=str, default="sharegpt",
                   choices=["sharegpt", "qa"])
    p.add_argument("--conversation_column", type=str, default="conversations")
    p.add_argument("--role_field",          type=str, default="from")
    p.add_argument("--content_field",       type=str, default="value")
    p.add_argument("--system_role",         type=str, default="system")
    p.add_argument("--user_role",           type=str, default="human")
    p.add_argument("--assistant_role",      type=str, default="gpt")
    p.add_argument("--question_column",     type=str, default="question")
    p.add_argument("--answer_column",       type=str, default="answer")

    # ── Special tokens (shared with trainer.py) ───────────────────────────────
    p.add_argument("--system_token",    type=str, default="[SYSTEM]")
    p.add_argument("--user_token",      type=str, default="[USER]")
    p.add_argument("--assistant_token", type=str, default="[ASSISTANT]")

    # ── SentenceTransformer ───────────────────────────────────────────────────
    p.add_argument("--sentence_model_name_or_path", type=str,
                   default="all-MiniLM-L6-v2")
    p.add_argument("--st_output_dir", type=str, default=None)

    # ── Pair extraction ───────────────────────────────────────────────────────
    p.add_argument("--include_system_in_anchor", action="store_true",
                   help="Prepend the system prompt to the user turn to form the anchor")
    p.add_argument("--add_role_prefixes", type=_str_to_bool, default=True,
                   metavar="BOOL",
                   help="Prefix anchor/positive with role tokens ([USER] ..., [ASSISTANT] ...)")
    p.add_argument("--st_max_pairs", type=int, default=50000,
                   help="Maximum (anchor, positive) pairs to extract. 0 = unlimited.")
    p.add_argument("--st_eval_fraction", type=float, default=0.05,
                   help="Fraction of pairs held out for evaluation.")

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--st_num_train_epochs", type=int,   default=1)
    p.add_argument("--st_batch_size",       type=int,   default=64)
    p.add_argument("--st_learning_rate",    type=float, default=2e-5)
    p.add_argument("--st_warmup_ratio",     type=float, default=0.1)
    p.add_argument("--st_eval_steps",       type=int,   default=500)
    p.add_argument("--st_save_steps",       type=int,   default=1000)
    p.add_argument("--st_save_total_limit", type=int,   default=2)
    p.add_argument("--seed",                type=int,   default=42)

    # ── Fold YAML defaults before final parse ─────────────────────────────────
    if known.config:
        cfg = _load_yaml_defaults(known.config)
        for k in _BOOL_FLAGS:
            if k in cfg:
                cfg[k] = bool(cfg[k])
        p.set_defaults(**cfg)

    return p.parse_args()


# ── Dataset loading ────────────────────────────────────────────────────────────

def _load_raw(args) -> List:
    kwargs = dict(
        streaming=True,
        trust_remote_code=args.trust_remote_code,
        verification_mode="no_checks",
    )
    if args.data_files:
        ds = load_dataset(
            "json",
            data_files={args.train_split: args.data_files},
            split=args.train_split,
            **kwargs,
        )
    else:
        ds = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=args.train_split,
            **kwargs,
        )
    n = args.max_train_samples if args.max_train_samples > 0 else 999_999_999
    return list(ds.take(n))


# ── Pair extraction ────────────────────────────────────────────────────────────

def _extract_pairs(ex: dict, args) -> List[dict]:
    """
    Return a list of {"anchor": str, "positive": str} from one example.

    anchor   = [SYSTEM] <sys> [USER] <user>  (system optional)
    positive = [ASSISTANT] <response>

    The role-token prefixes teach the SentenceTransformer the asymmetry between
    query context and assistant response — mirroring what trainer.py encodes in
    the BERT vocabulary.
    """
    pairs: List[dict] = []

    if args.format == "sharegpt":
        turns = ex.get(args.conversation_column, [])
        if not isinstance(turns, list):
            return pairs

        system_content = ""
        last_user = ""

        for turn in turns:
            role    = str(turn.get(args.role_field, ""))
            content = str(turn.get(args.content_field, "")).strip()
            if not content:
                continue

            if role == args.system_role:
                system_content = content

            elif role == args.user_role:
                last_user = content

            elif role == args.assistant_role and last_user:
                anchor_parts = []

                if args.include_system_in_anchor and system_content:
                    pfx = f"{args.system_token} " if args.add_role_prefixes else ""
                    anchor_parts.append(f"{pfx}{system_content}")

                user_pfx = f"{args.user_token} " if args.add_role_prefixes else ""
                anchor_parts.append(f"{user_pfx}{last_user}")

                asst_pfx = f"{args.assistant_token} " if args.add_role_prefixes else ""

                pairs.append({
                    "anchor":   " ".join(anchor_parts),
                    "positive": f"{asst_pfx}{content}",
                })
                last_user = ""   # reset; system_content persists across turns

    elif args.format == "qa":
        q = str(ex.get(args.question_column, "")).strip()
        a = str(ex.get(args.answer_column, "")).strip()
        if q and a:
            user_pfx = f"{args.user_token} " if args.add_role_prefixes else ""
            asst_pfx = f"{args.assistant_token} " if args.add_role_prefixes else ""
            pairs.append({
                "anchor":   f"{user_pfx}{q}",
                "positive": f"{asst_pfx}{a}",
            })

    return pairs


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not args.dataset_name:
        print("ERROR: --dataset_name is required (or set via --config)")
        sys.exit(1)
    if not args.st_output_dir:
        print("ERROR: --st_output_dir is required (or set via --config)")
        sys.exit(1)

    Path(args.st_output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load raw examples ─────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset_name}")
    if args.data_files:
        print(f"  data_files: {args.data_files}")
    raw = _load_raw(args)
    print(f"  loaded {len(raw):,} examples")

    # ── Extract pairs ─────────────────────────────────────────────────────────
    print("Extracting (anchor, positive) pairs...")
    all_pairs: List[dict] = []
    limit = args.st_max_pairs if args.st_max_pairs > 0 else None

    for ex in raw:
        all_pairs.extend(_extract_pairs(ex, args))
        if limit and len(all_pairs) >= limit:
            break

    if limit:
        all_pairs = all_pairs[:limit]

    print(f"  {len(all_pairs):,} pairs extracted")

    if not all_pairs:
        print("ERROR: no pairs found — check --format and column settings")
        sys.exit(1)

    # ── Split pairs ───────────────────────────────────────────────────────────
    random.seed(args.seed)
    random.shuffle(all_pairs)
    n_eval = max(1, int(len(all_pairs) * args.st_eval_fraction))
    eval_pairs  = all_pairs[:n_eval]
    train_pairs = all_pairs[n_eval:]
    print(f"  train: {len(train_pairs):,}  eval: {len(eval_pairs):,}")

    # ── Model and loss ────────────────────────────────────────────────────────
    print(f"\nLoading SentenceTransformer: {args.sentence_model_name_or_path}")
    model = SentenceTransformer(args.sentence_model_name_or_path)
    loss  = MultipleNegativesRankingLoss(model)

    # ── DataLoader ────────────────────────────────────────────────────────────
    # Use classic model.fit() to avoid transformers-version compatibility issues
    # with SentenceTransformerTrainer.compute_loss() and num_items_in_batch.
    train_examples = [
        InputExample(texts=[p["anchor"], p["positive"]]) for p in train_pairs
    ]
    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=args.st_batch_size
    )

    # ── Evaluator ─────────────────────────────────────────────────────────────
    # Score each (anchor, positive) pair against a target similarity of 1.0.
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=[p["anchor"]   for p in eval_pairs],
        sentences2=[p["positive"] for p in eval_pairs],
        scores=[1.0] * len(eval_pairs),
        name="dialogue-coherence",
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    warmup_steps = int(
        len(train_dataloader) * args.st_num_train_epochs * args.st_warmup_ratio
    )
    print(f"\nTraining ({warmup_steps} warmup steps)...")
    # evaluation_steps is intentionally omitted: passing it sets eval_strategy="steps"
    # inside SentenceTransformerTrainer which requires eval_dataset, not just an evaluator.
    # Without it, the evaluator runs at the end of each epoch instead.
    model.fit(
        train_objectives=[(train_dataloader, loss)],
        evaluator=evaluator,
        epochs=args.st_num_train_epochs,
        warmup_steps=warmup_steps,
        output_path=args.st_output_dir,
        optimizer_params={"lr": args.st_learning_rate},
        show_progress_bar=True,
        save_best_model=True,
    )

    print(f"\nSaved to {args.st_output_dir}")
    print(f"To use in qBERT.py, set ModelConfig.sentence_model = \"{args.st_output_dir}\"")


if __name__ == "__main__":
    main()
