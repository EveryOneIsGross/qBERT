#!/usr/bin/env python3
"""
audit_dataset.py

Audits a HuggingFace dataset before training. Checks structure, role
distribution, sequence length, turn counts, and flags malformed examples.

Usage:
    python audit_dataset.py --dataset_name NousResearch/Hermes-3-Dataset \\
                            --data_files hf://datasets/NousResearch/Hermes-3-Dataset/hermes-3-dataset.jsonl

    python audit_dataset.py --dataset_name teknium/OpenHermes-2.5 --max_samples 500

    # QA format dataset
    python audit_dataset.py --dataset_name tatsu-lab/alpaca \\
                            --format qa \\
                            --question_column instruction \\
                            --answer_column output
"""

import argparse
import sys
from collections import Counter, defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    # Dataset loading
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--dataset_config_name", type=str, default=None)
    p.add_argument("--data_files", type=str, default=None,
                   help="Direct file path/URL (use when the dataset repo contains non-data files)")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--trust_remote_code", action="store_true")

    # Format
    p.add_argument("--format", type=str, default="sharegpt",
                   choices=["sharegpt", "qa", "text"])
    p.add_argument("--conversation_column", type=str, default="conversations")
    p.add_argument("--role_field",          type=str, default="from")
    p.add_argument("--content_field",       type=str, default="value")
    p.add_argument("--system_role",         type=str, default="system")
    p.add_argument("--user_role",           type=str, default="human")
    p.add_argument("--assistant_role",      type=str, default="gpt")
    p.add_argument("--question_column",     type=str, default="question")
    p.add_argument("--answer_column",       type=str, default="answer")
    p.add_argument("--text_column",         type=str, default="text")

    # Structure tokens (must match trainer invocation)
    p.add_argument("--system_token",    type=str, default="[SYSTEM]")
    p.add_argument("--user_token",      type=str, default="[USER]")
    p.add_argument("--assistant_token", type=str, default="[ASSISTANT]")
    p.add_argument("--eot_token",       type=str, default="[EOT]")

    # Tokenizer
    p.add_argument("--model_name_or_path", type=str, default="bert-base-cased")

    # Audit controls
    p.add_argument("--max_samples",   type=int, default=2000,
                   help="Examples to inspect. 0 = full dataset.")
    p.add_argument("--max_seq_length", type=int, default=512,
                   help="Flag examples that exceed this token length.")
    p.add_argument("--show_examples", type=int, default=2,
                   help="Print N malformed examples in full.")
    return p.parse_args()


def fmt_hist(counter, total, bins=10, width=40):
    if not counter:
        return "  (empty)"
    keys = sorted(counter)
    mn, mx = keys[0], keys[-1]
    if mn == mx:
        return f"  all values = {mn}"
    step = max(1, (mx - mn) // bins)
    buckets = defaultdict(int)
    for v, c in counter.items():
        b = ((v - mn) // step) * step + mn
        buckets[b] += c
    bmax = max(buckets.values())
    lines = []
    for b in sorted(buckets):
        bar = int(buckets[b] / bmax * width)
        pct = buckets[b] / total * 100
        lines.append(f"  {b:>5}-{b+step:<5} │{'█'*bar:<{width}}│ {buckets[b]:>6} ({pct:4.1f}%)")
    return "\n".join(lines)


def load_streaming(args):
    kwargs = dict(streaming=True, trust_remote_code=args.trust_remote_code)
    if args.data_files:
        return load_dataset("json", data_files={args.split: args.data_files},
                            split=args.split, **kwargs)
    return load_dataset(args.dataset_name, args.dataset_config_name,
                        split=args.split, **kwargs)


def get_text_for_length(ex, args, role_map, special_tokens):
    """Build the formatted string that trainer.py would produce, for length estimation."""
    if args.format == "sharegpt":
        convs = ex.get(args.conversation_column, [])
        parts = []
        for turn in convs:
            role = turn.get(args.role_field, "")
            content = str(turn.get(args.content_field, "")).strip()
            tok = role_map.get(role, role)
            parts.append(f"{tok} {content} {args.eot_token}")
        return "\n".join(parts)
    elif args.format == "qa":
        q = str(ex.get(args.question_column, "")).strip()
        a = str(ex.get(args.answer_column, "")).strip()
        return f"{args.user_token} {q} {args.eot_token}\n{args.assistant_token} {a} {args.eot_token}"
    else:
        return str(ex.get(args.text_column, "")).strip()


def main():
    args = parse_args()

    known_roles = {args.system_role, args.user_role, args.assistant_role}
    role_map = {
        args.system_role:    args.system_token,
        args.user_role:      args.user_token,
        args.assistant_role: args.assistant_token,
    }
    special_tokens = [t for t in [
        args.system_token, args.user_token, args.assistant_token, args.eot_token
    ] if t]

    print(f"Loading: {args.dataset_name} [{args.split}]")
    if args.data_files:
        print(f"  data_files: {args.data_files}")
    ds = load_streaming(args)

    # Schema discovery
    peek = list(ds.take(5))
    print(f"\n--- RAW SCHEMA ---")
    if peek:
        print(f"  columns: {list(peek[0].keys())}")
        for k, v in peek[0].items():
            vtype = type(v).__name__
            preview = str(v)[:120].replace("\n", " ")
            print(f"  [{k}] ({vtype}): {preview}")
    print("---\n")

    if not peek:
        print("ERROR: dataset returned no examples.")
        sys.exit(1)

    actual_cols = set(peek[0].keys())

    # Format-specific column check
    if args.format == "sharegpt" and args.conversation_column not in actual_cols:
        print(f"ERROR: conversation column '{args.conversation_column}' not found.")
        print(f"  Available: {sorted(actual_cols)}")
        sys.exit(1)
    elif args.format == "qa":
        missing = [c for c in [args.question_column, args.answer_column] if c not in actual_cols]
        if missing:
            print(f"ERROR: QA columns not found: {missing}")
            print(f"  Available: {sorted(actual_cols)}")
            sys.exit(1)
    elif args.format == "text" and args.text_column not in actual_cols:
        print(f"ERROR: text column '{args.text_column}' not found.")
        print(f"  Available: {sorted(actual_cols)}")
        sys.exit(1)

    n_limit = args.max_samples if args.max_samples > 0 else 999_999_999
    sample = list(ds.take(n_limit))
    n = len(sample)
    print(f"Auditing {n:,} examples\n")

    print(f"Loading tokenizer: {args.model_name_or_path}")
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tok.add_special_tokens({"additional_special_tokens": special_tokens})

    # -----------------------------------------------------------------------
    # Collect stats
    # -----------------------------------------------------------------------
    role_counter     = Counter()
    turns_counter    = Counter()
    token_len_ctr    = Counter()
    missing_field    = []
    unknown_roles    = []
    empty_content    = []
    over_length      = []
    no_assistant     = []
    consecutive_same = []

    for idx, ex in enumerate(sample):
        if args.format == "sharegpt":
            convs = ex.get(args.conversation_column)
            if not isinstance(convs, list) or len(convs) == 0:
                missing_field.append(idx)
                continue

            roles_in_ex = []
            has_assistant = False

            for turn in convs:
                role = str(turn.get(args.role_field, ""))
                content = turn.get(args.content_field, "")
                role_counter[role] += 1
                if role not in known_roles:
                    unknown_roles.append((idx, role))
                if not str(content).strip():
                    empty_content.append((idx, role))
                if role == args.assistant_role:
                    has_assistant = True
                roles_in_ex.append(role)

            if not has_assistant:
                no_assistant.append(idx)

            for i in range(len(roles_in_ex) - 1):
                if roles_in_ex[i] == roles_in_ex[i + 1]:
                    consecutive_same.append((idx, roles_in_ex[i], i))
                    break

            turns_counter[len(convs)] += 1

        elif args.format == "qa":
            q = str(ex.get(args.question_column, "")).strip()
            a = str(ex.get(args.answer_column, "")).strip()
            if not q:
                missing_field.append(idx)
                continue
            if not a:
                no_assistant.append(idx)
            turns_counter[2] += 1

        else:  # text
            t = str(ex.get(args.text_column, "")).strip()
            if not t:
                missing_field.append(idx)
                continue
            turns_counter[1] += 1

        text = get_text_for_length(ex, args, role_map, special_tokens)
        tlen = len(tok.encode(text, add_special_tokens=True, truncation=False))
        token_len_ctr[(tlen // 64) * 64] += 1
        if tlen > args.max_seq_length:
            over_length.append((idx, tlen))

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    sep = "─" * 60

    print(sep)
    print("COLUMN STRUCTURE")
    print(sep)
    first = sample[0]
    print(f"  Columns: {list(first.keys())}")
    if args.format == "sharegpt" and first.get(args.conversation_column):
        convs = first[args.conversation_column]
        print(f"  Turn fields: {list(convs[0].keys())}")
        for t in convs[:2]:
            preview = str(t[args.content_field])[:80].replace("\n", " ")
            print(f"    [{t[args.role_field]}] {preview}...")

    if args.format == "sharegpt":
        print(f"\n{sep}")
        print("ROLE DISTRIBUTION")
        print(sep)
        total_turns = sum(role_counter.values())
        for role, count in role_counter.most_common():
            pct = count / total_turns * 100
            marker = "" if role in known_roles else "  ← UNKNOWN"
            print(f"  {role:<20} {count:>8,}  ({pct:5.1f}%){marker}")

        print(f"\n{sep}")
        print("TURN COUNT DISTRIBUTION")
        print(sep)
        print(fmt_hist(turns_counter, n))
        avg = sum(k * v for k, v in turns_counter.items()) / n
        print(f"  avg turns/conversation: {avg:.1f}")

    print(f"\n{sep}")
    print(f"TOKEN LENGTH DISTRIBUTION ({args.model_name_or_path}, after special tokens)")
    print(sep)
    print(fmt_hist(token_len_ctr, n))
    over_pct = len(over_length) / n * 100
    print(f"  exceeding {args.max_seq_length}: {len(over_length):,} ({over_pct:.1f}%)")
    if over_length:
        lengths = [l for _, l in over_length]
        print(f"  longest: {max(lengths)}  |  median over-length: {sorted(lengths)[len(lengths)//2]}")

    print(f"\n{sep}")
    print("QUALITY FLAGS")
    print(sep)

    def flag(label, items, warn_threshold=0):
        pct = len(items) / n * 100
        sym = "OK " if len(items) <= warn_threshold else "!  "
        print(f"  [{sym}] {label}: {len(items):,} ({pct:.2f}%)")

    flag("Missing/empty entries",          missing_field)
    flag("Unknown roles",                  unknown_roles)
    flag("Empty content turns",            empty_content,   warn_threshold=10)
    flag("No assistant turn",              no_assistant)
    flag("Consecutive same-role turns",    consecutive_same, warn_threshold=5)
    flag(f"Over max_seq_length ({args.max_seq_length})", over_length)

    if unknown_roles and args.show_examples > 0:
        print(f"\n  Unknown role examples (first {args.show_examples}):")
        for idx, role in unknown_roles[:args.show_examples]:
            print(f"    row {idx}: role='{role}'")
            for t in sample[idx].get(args.conversation_column, []):
                print(f"      [{t[args.role_field]}] {str(t[args.content_field])[:60]}")

    if empty_content and args.show_examples > 0:
        print(f"\n  Empty content examples (first {args.show_examples}):")
        for idx, role in empty_content[:args.show_examples]:
            print(f"    row {idx}, role='{role}'")

    # -----------------------------------------------------------------------
    # Recommended trainer command
    # -----------------------------------------------------------------------
    sorted_lens = sorted(v for b, c in token_len_ctr.items() for v in [b] * c)
    p90 = sorted_lens[int(len(sorted_lens) * 0.90)] + 64 if sorted_lens else 512
    p90 = min(p90, 512)

    data_files_flag = (f"\\\n  --data_files \"{args.data_files}\" " if args.data_files else "")
    role_flags = ""
    if args.user_role != "human":
        role_flags += f"\\\n  --user_role \"{args.user_role}\" "
    if args.assistant_role != "gpt":
        role_flags += f"\\\n  --assistant_role \"{args.assistant_role}\" "

    print(f"\n{sep}")
    print("RECOMMENDED TRAINER COMMAND")
    print(sep)
    print(f"""
python trainer.py \\
  --dataset_name "{args.dataset_name}" {data_files_flag}\\
  --format {args.format} {role_flags}\\
  --output_dir "./qbert-finetuned" \\
  --model_name_or_path "{args.model_name_or_path}" \\
  --max_seq_length {p90} \\
  --per_device_train_batch_size 8 \\
  --num_train_epochs 3 \\
  --learning_rate 5e-5 \\
  --warmup_ratio 0.03 \\
  --logging_steps 100 \\
  --save_steps 1000 \\
  --save_total_limit 2 \\
  --assistant_only""")

    print(f"\n  p90 token length from sample: {p90}  |  {over_pct:.1f}% of examples exceed {args.max_seq_length} and will be truncated.")

    hard_failures = len(missing_field) + len(unknown_roles) + len(no_assistant)
    if hard_failures > 0:
        print(f"\n  {hard_failures} hard issue(s) found — review before training.")
        sys.exit(1)
    else:
        print("\n  Dataset looks clean. Safe to train.")
        sys.exit(0)


if __name__ == "__main__":
    main()
