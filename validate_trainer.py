#!/usr/bin/env python3
"""
validate_trainer.py

Post-training validation suite for qBERT's structure-aware BERT fine-tuning.
Checks tokenizer integrity, label masking logic, collator behaviour, and
model MLM output. Import-safe: only runs on __main__.

Usage:
    python validate_trainer.py --model_dir ./qbert-finetuned
    python validate_trainer.py --model_dir ./qbert-finetuned --verbose
"""

import argparse
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, BertForMaskedLM

# Import trainer internals directly so we test the real code paths
from trainer import build_token_ids_and_labels, StructureAwareMLMCollator


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
WARN = "\033[93m WARN\033[0m"
HEAD = "\033[1m{}\033[0m"

_failures = []
_warnings = []


def check(name, condition, detail="", warn_only=False):
    tag = WARN if (not condition and warn_only) else (PASS if condition else FAIL)
    print(f"  [{tag}] {name}")
    if detail and (not condition or args.verbose):
        print(f"         {detail}")
    if not condition:
        if warn_only:
            _warnings.append(name)
        else:
            _failures.append(name)
    return condition


def section(title):
    print(f"\n{HEAD.format(title)}")


# ---------------------------------------------------------------------------
# Fake args object so we can call trainer functions without argparse
# ---------------------------------------------------------------------------

class FakeArgs:
    system_token  = "[SYSTEM]"
    user_token    = "[USER]"
    assistant_token = "[ASSISTANT]"
    eot_token     = "[EOT]"


# ---------------------------------------------------------------------------
# 1. Tokenizer integrity
# ---------------------------------------------------------------------------

def test_tokenizer(tok, special_tokens):
    section("1. Tokenizer integrity")

    # All expected specials are registered
    registered = set(tok.additional_special_tokens)
    for t in special_tokens:
        check(
            f"'{t}' in additional_special_tokens",
            t in registered,
            f"registered: {registered}",
        )

    # Each special encodes to exactly one token ID
    for t in special_tokens:
        ids = tok.encode(t, add_special_tokens=False)
        check(
            f"'{t}' encodes to single id",
            len(ids) == 1,
            f"got {len(ids)} ids: {ids}",
        )

    # Tokenising a conversation string keeps specials atomic
    conv = "[USER] hello there [ASSISTANT] hi back [EOT]"
    toks = tok.tokenize(conv)
    for t in ["[USER]", "[ASSISTANT]", "[EOT]"]:
        check(
            f"'{t}' appears as single token in tokenize()",
            t in toks,
            f"tokens: {toks}",
        )

    # No special appears split (e.g. '[', 'AS', '##SI', ...]
    brackets = [x for x in toks if x in ("[", "]") and x not in special_tokens]
    check(
        "No bare '[' or ']' fragments from splitting specials",
        len(brackets) == 0,
        f"fragments: {brackets}",
    )

    # Roundtrip: encode then decode recovers the string
    sample = "[USER] the quick brown fox [ASSISTANT] jumped [EOT]"
    ids = tok.encode(sample, add_special_tokens=False)
    decoded = tok.decode(ids, skip_special_tokens=False)
    check(
        "Encode/decode roundtrip preserves structure tokens",
        all(t in decoded for t in ["[USER]", "[ASSISTANT]", "[EOT]"]),
        f"decoded: {decoded!r}",
    )


# ---------------------------------------------------------------------------
# 2. Embedding integrity
# ---------------------------------------------------------------------------

def test_embeddings(tok, model):
    section("2. Embedding integrity")

    vocab_len = len(tok)
    embed_rows = model.bert.embeddings.word_embeddings.weight.shape[0]

    check(
        f"Embedding rows ({embed_rows}) == tokenizer vocab ({vocab_len})",
        embed_rows == vocab_len,
        f"mismatch: model has {embed_rows}, tokenizer has {vocab_len}",
    )

    # New special embedding rows should be non-zero (they've been initialised
    # at resize time; even untrained they shouldn't be exactly zero)
    for t in tok.additional_special_tokens:
        ids = tok.encode(t, add_special_tokens=False)
        if not ids:
            continue
        row = model.bert.embeddings.word_embeddings.weight[ids[0]]
        check(
            f"Embedding row for '{t}' is non-zero",
            row.abs().sum().item() > 0,
            f"row norm: {row.norm().item():.6f}",
        )

    check(
        "model.config.vocab_size matches tokenizer",
        model.config.vocab_size == vocab_len,
        f"config: {model.config.vocab_size}, tok: {vocab_len}",
    )


# ---------------------------------------------------------------------------
# 3. Label masking logic
# ---------------------------------------------------------------------------

def test_label_masking(tok):
    section("3. Label masking (build_token_ids_and_labels)")

    fargs = FakeArgs()
    max_len = 128

    # --- single-turn conversation ---
    text = "[USER] what is the capital of France [EOT]\n[ASSISTANT] Paris is the capital [EOT]"
    result = build_token_ids_and_labels(text, tok, max_len, fargs)

    input_ids     = result["input_ids"]
    labels        = result["labels"]
    attention_mask = result["attention_mask"]

    check("output has input_ids, labels, is_assistant, attention_mask",
          set(result.keys()) >= {"input_ids", "labels", "is_assistant", "attention_mask"})

    check("input_ids length == max_len",
          len(input_ids) == max_len, f"got {len(input_ids)}")

    check("labels length == max_len",
          len(labels) == max_len, f"got {len(labels)}")

    is_assistant = result["is_assistant"]

    # At least some labels should be real (assistant + user/system content)
    real_labels = [l for l in labels if l != -100]
    check("Some labels are real (content trained)",
          len(real_labels) > 0,
          f"all labels are -100 — no content was unlocked")

    # is_assistant marks only assistant positions
    asst_count = sum(1 for a in is_assistant if a)
    check("Some positions flagged as assistant",
          asst_count > 0,
          f"is_assistant is all False")

    # All padding positions should be -100
    pad_id = tok.pad_token_id
    for i, (iid, lab, am) in enumerate(zip(input_ids, labels, attention_mask)):
        if iid == pad_id:
            check(f"Padding at pos {i} has label -100",
                  lab == -100,
                  f"got label {lab}",
                  warn_only=True)
            break  # just check first pad we find

    # Special tokens should always be -100
    special_ids = set(tok.additional_special_tokens_ids) | set(tok.all_special_ids)
    bad_specials = [
        i for i, (iid, lab) in enumerate(zip(input_ids, labels))
        if iid in special_ids and lab != -100
    ]
    check("No special token has a real label",
          len(bad_specials) == 0,
          f"positions with bad labels: {bad_specials}")

    # Frame tokens themselves should still be -100
    asst_tok_id = tok.encode("[ASSISTANT]", add_special_tokens=False)[0]
    user_tok_id = tok.encode("[USER]", add_special_tokens=False)[0]
    for tid, name in [(asst_tok_id, "[ASSISTANT]"), (user_tok_id, "[USER]")]:
        bad = [i for i, (iid, lab) in enumerate(zip(input_ids, labels))
               if iid == tid and lab != -100]
        check(f"Frame token {name} always has label -100",
              len(bad) == 0,
              f"found real labels at positions: {bad}")

    # User content positions should be trainable (real labels) but not flagged as assistant
    asst_pos = next((i for i, iid in enumerate(input_ids) if iid == asst_tok_id), None)
    if asst_pos is not None:
        user_real = [(i, l) for i, (l, a) in enumerate(zip(labels[:asst_pos], is_assistant[:asst_pos]))
                     if l != -100 and not a]
        check("User content has real labels (context MLM)",
              len(user_real) > 0,
              f"no real labels found in user region before pos {asst_pos}")

    # --- multi-turn ---
    multi = (
        "[USER] first question [EOT]\n"
        "[ASSISTANT] first answer [EOT]\n"
        "[USER] follow up [EOT]\n"
        "[ASSISTANT] follow up answer [EOT]"
    )
    mr = build_token_ids_and_labels(multi, tok, 256, fargs)
    real_multi = [l for l in mr["labels"] if l != -100]
    check("Multi-turn: real labels exist across both assistant turns",
          len(real_multi) > 0,
          f"real label count: {len(real_multi)}")

    # --- empty assistant content ---
    empty = "[USER] hello [EOT]\n[ASSISTANT] [EOT]"
    er = build_token_ids_and_labels(empty, tok, 64, fargs)
    check("Empty assistant content doesn't crash",
          "input_ids" in er)


# ---------------------------------------------------------------------------
# 4. Collator behaviour
# ---------------------------------------------------------------------------

def test_collator(tok, special_tokens):
    section("4. StructureAwareMLMCollator")

    fargs = FakeArgs()
    max_len = 128
    mlm_prob = 0.15

    special_ids = set(tok.additional_special_tokens_ids) | set(tok.all_special_ids)

    collator = StructureAwareMLMCollator(
        tokenizer=tok,
        mlm_probability=mlm_prob,
        context_mlm_probability=0.10,
        special_ids=special_ids,
        assistant_only=False,
    )

    # Build a batch of two examples
    texts = [
        "[USER] what colour is the sky [EOT]\n[ASSISTANT] the sky is blue [EOT]",
        "[USER] how many days in a week [EOT]\n[ASSISTANT] there are seven days [EOT]",
    ]
    features = [
        build_token_ids_and_labels(t, tok, max_len, fargs)
        for t in texts
    ]

    batch = collator(features)

    check("Batch has input_ids, labels, attention_mask",
          set(batch.keys()) >= {"input_ids", "labels", "attention_mask"})

    input_ids = batch["input_ids"]   # (B, L)
    labels    = batch["labels"]      # (B, L)
    mask_id   = tok.mask_token_id

    # 4a. Special tokens must never be masked
    for sid in special_ids:
        positions = (input_ids == mask_id) & (
            torch.tensor([
                [f["input_ids"][j] == sid for j in range(max_len)]
                for f in features
            ])
        )
        # The original position had a special token AND the collator put [MASK] there
        # We need to check original ids, not the masked batch
        orig_ids = torch.tensor([f["input_ids"] for f in features])
        bad = ((orig_ids == sid) & (input_ids == mask_id)).any()
        check(
            f"Special id {sid} ({tok.convert_ids_to_tokens([sid])[0]!r}) never replaced with [MASK]",
            not bad.item(),
        )

    # 4b. Padding never masked
    pad_id = tok.pad_token_id
    orig_ids = torch.tensor([f["input_ids"] for f in features])
    pad_masked = ((orig_ids == pad_id) & (input_ids == mask_id)).any()
    check("Padding positions never replaced with [MASK]",
          not pad_masked.item())

    # 4c. Frame token / padding positions (-100 labels) are never masked
    orig_labels = torch.tensor([f["labels"] for f in features])
    non_content_masked = ((orig_labels == -100) & (input_ids == mask_id)).any()
    check("Frame/padding positions (label=-100) never masked",
          not non_content_masked.item())

    # 4d. At least some positions were masked overall (with enough samples
    # and 15% rate this is nearly guaranteed)
    total_masked = (input_ids == mask_id).sum().item()
    check("At least one position was masked in the batch",
          total_masked > 0,
          f"masked count: {total_masked}",
          warn_only=True)

    # 4e. Masked positions have real labels (not -100)
    masked_positions = input_ids == mask_id
    if masked_positions.any():
        masked_labels = labels[masked_positions]
        all_real = (masked_labels != -100).all()
        check("All [MASK] positions have real labels (non -100)",
              all_real.item(),
              f"found -100 labels at masked positions")


# ---------------------------------------------------------------------------
# 5. Model MLM smoke test
# ---------------------------------------------------------------------------

def test_mlm_output(tok, model, special_tokens):
    section("5. MLM output smoke test")

    model.eval()
    special_ids = set(tok.additional_special_tokens_ids) | set(tok.all_special_ids)
    mask_id = tok.mask_token_id

    # Construct a sequence where we mask a known assistant token
    prompt = "[USER] what is the colour of the sky [EOT] [ASSISTANT] the sky is [MASK] [EOT]"
    enc = tok(prompt, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Find the [MASK] position
    mask_positions = (input_ids[0] == mask_id).nonzero(as_tuple=True)[0]
    check("Prompt contains exactly one [MASK]",
          len(mask_positions) == 1,
          f"found {len(mask_positions)} mask positions")

    if len(mask_positions) == 0:
        return

    mask_pos = mask_positions[0].item()

    with torch.inference_mode():
        logits = model(**enc).logits  # (1, seq, vocab)

    pred_logits = logits[0, mask_pos]  # (vocab,)

    # Top prediction should not be a special token
    top_id = pred_logits.argmax().item()
    top_tok = tok.convert_ids_to_tokens([top_id])[0]
    check("Top-1 prediction is not a special token",
          top_id not in special_ids,
          f"top prediction: {top_tok!r} (id={top_id})")

    # Top-5 predictions — at least one should be a common word (alphabetic)
    top5_ids = pred_logits.topk(5).indices.tolist()
    top5_toks = tok.convert_ids_to_tokens(top5_ids)
    has_alpha = any(any(c.isalpha() for c in t.replace("##", "").replace("▁", ""))
                    for t in top5_toks)
    check("Top-5 predictions contain at least one alphabetic token",
          has_alpha,
          f"top-5: {top5_toks}")

    if args.verbose:
        print(f"         top-5 predictions: {list(zip(top5_toks, top5_ids))}")

    # Verify model does NOT predict [USER]/[ASSISTANT] in this slot
    for t in ["[USER]", "[ASSISTANT]", "[SYSTEM]", "[EOT]"]:
        t_ids = tok.encode(t, add_special_tokens=False)
        if not t_ids:
            continue
        tid = t_ids[0]
        rank = (pred_logits >= pred_logits[tid]).sum().item()
        check(
            f"'{t}' is not in top-10 predictions (rank={rank})",
            rank > 10,
            f"'{t}' ranked {rank} — model may hallucinate role tokens",
            warn_only=True,
        )


# ---------------------------------------------------------------------------
# 6. qBERT integration: special IDs excluded from generation candidates
# ---------------------------------------------------------------------------

def test_qbert_valid_mask(tok, model):
    section("6. Generation candidate filtering (qBERT._mask_preds gap)")

    special_ids = set(tok.additional_special_tokens_ids) | set(tok.all_special_ids)

    # Replicate the valid mask as written in qBERT._mask_preds
    vocab_size = model.config.vocab_size
    valid_qbert = torch.ones(vocab_size, dtype=torch.bool)
    bert_specials = {
        tok.pad_token_id,
        getattr(tok, "cls_token_id", None),
        getattr(tok, "sep_token_id", None),
        getattr(tok, "mask_token_id", None),
        getattr(tok, "unk_token_id", None),
    }
    bert_specials = {t for t in bert_specials if t is not None}
    if bert_specials:
        valid_qbert[list(bert_specials)] = False

    # Check whether the new special token IDs are still marked valid
    leaking = []
    for sid in tok.additional_special_tokens_ids:
        if sid < vocab_size and valid_qbert[sid]:
            leaking.append((sid, tok.convert_ids_to_tokens([sid])[0]))

    check(
        "No added special tokens leak through qBERT's valid mask",
        len(leaking) == 0,
        f"Leaking IDs (will be generatable): {leaking}\n"
        f"         Fix: add tok.additional_special_tokens_ids to the specials set in _mask_preds",
        warn_only=True,  # warn not fail — qBERT.py isn't modified yet
    )

    if leaking:
        print(f"         Suggested fix for qBERT.py _mask_preds (~line 297):")
        print(f"           specials |= set(self.tok.additional_special_tokens_ids)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_validate_args():
    p = argparse.ArgumentParser(description="Validate a qBERT fine-tuned checkpoint")
    p.add_argument("--model_dir", type=str, required=True,
                   help="Path to fine-tuned model directory (output_dir from trainer.py)")
    p.add_argument("--base_model", type=str, default=None,
                   help="Original base model (for comparison checks). Optional.")
    p.add_argument("--system_token",    type=str, default="[SYSTEM]")
    p.add_argument("--user_token",      type=str, default="[USER]")
    p.add_argument("--assistant_token", type=str, default="[ASSISTANT]")
    p.add_argument("--eot_token",       type=str, default="[EOT]")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_validate_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: model_dir '{model_dir}' does not exist.")
        sys.exit(1)

    special_tokens = [t for t in [
        args.system_token, args.user_token, args.assistant_token, args.eot_token
    ] if t]

    print(f"\nLoading tokenizer and model from: {model_dir}")
    tok   = AutoTokenizer.from_pretrained(str(model_dir))
    model = BertForMaskedLM.from_pretrained(str(model_dir))
    model.eval()
    print(f"Vocab size: {len(tok)}  |  Specials: {tok.additional_special_tokens}")

    # Run all sections
    test_tokenizer(tok, special_tokens)
    test_embeddings(tok, model)
    test_label_masking(tok)
    test_collator(tok, special_tokens)
    test_mlm_output(tok, model, special_tokens)
    test_qbert_valid_mask(tok, model)

    # Summary
    print(f"\n{'='*50}")
    if _failures:
        print(f"\033[91mFAILED {len(_failures)} check(s):\033[0m")
        for f in _failures:
            print(f"  - {f}")
    if _warnings:
        print(f"\033[93mWARNINGS {len(_warnings)} check(s):\033[0m")
        for w in _warnings:
            print(f"  - {w}")
    if not _failures and not _warnings:
        print("\033[92mAll checks passed.\033[0m")
    elif not _failures:
        print("\033[92mNo failures.\033[0m Warnings above are informational.")

    sys.exit(1 if _failures else 0)
