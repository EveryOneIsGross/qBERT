import warnings; warnings.filterwarnings("ignore")
import logging; logging.getLogger("transformers").setLevel(logging.ERROR)

from dataclasses import dataclass
from typing import Any, List, Tuple, Union, Iterator, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer


@dataclass
class GenerationConfig:
    max_length: int = 512
    batch_size: int = 8
    num_candidates: int = 128
    embedding_dim: int = 768
    context_window: int = 256
    phrase_window: Optional[int] = None
    base_temperature: float = 0.7
    min_threshold: float = 0.5
    top_k: int = 32
    compression_ratio: float = 0.2
    max_cache_size: int = 16
    sequence_cache_size: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_semantic_memory: bool = True
    use_bidirectional_context: bool = True
    use_adaptive_temperature: bool = True
    use_coherence_scoring: bool = True
    use_sequence_memory: bool = True
    use_learned_gate: bool = False
    tau_coherence: float = 1.7
    p_nucleus: float = 0.9
    repetition_penalty: float = 0.8
    repetition_window: int = 64
    entity_boost: float = 1.05
    use_entity_bias: bool = True
    gibbs_every_M: int = 0
    gibbs_span_L: int = 0
    use_space_alignment: bool = False
    cache_limit: int = 1024
    dynamic_window_gamma: float = 0.0
    memory_alpha: float = 0.002
    memory_beta: float = 0.15
    # Coherence anchoring: blend user-input embedding into semantic scoring context.
    # 0.0 = pure generated-tail context (original behaviour)
    # 1.0 = pure user-input anchor
    # Values around 0.4-0.6 keep generation grounded without being too rigid.
    coherence_anchor_alpha: float = 0.5
    # Taper anchor strength as generation progresses (True = strong anchor early,
    # fades to coherence_anchor_alpha by end of response).
    coherence_anchor_taper: bool = True
    # Homeostasis: dynamically modulate tau_coherence, p_nucleus, and anchor_alpha
    # based on a running EMA of per-step coherence scores.
    use_coherence_homeostasis: bool = True
    coherence_ema_alpha: float = 0.15       # EMA decay rate (higher = more reactive)
    coherence_low_threshold: float = 0.40   # below this: tighten all pressures
    coherence_high_threshold: float = 0.65  # above this: loosen slightly
    # Entropy-gated Gibbs: instead of fixed intervals, target positions where the
    # model was most uncertain during the forward pass.
    use_entropy_gibbs: bool = True
    # Optional prompt-aware denoise pass. This runs after draft generation and
    # resamples generated tokens with the full left/right BERT context available.
    use_denoise: bool = False
    denoise_passes: int = 1
    denoise_top_k: int = 32
    denoise_accept_margin: float = 0.15
    denoise_temperature: float = 0.6
    stream_denoise: bool = True
    # Noise pattern for the denoise pass:
    #   sequential   - left-to-right (reversed on odd passes); original behaviour
    #   random       - shuffle positions each pass (standard MLM-style diffusion)
    #   entropy      - prioritise positions where draft generation was most uncertain
    #   span         - mask random contiguous spans (phrase-level re-evaluation)
    #   checkerboard - alternate even/odd positions each pass
    denoise_pattern: str = "sequential"
    # How many positions to batch into one BERT forward pass during denoise.
    # Higher = faster but more VRAM.  Set to 0 to use all positions at once.
    denoise_batch_size: int = 32

    def __post_init__(self):
        if self.phrase_window is None:
            self.phrase_window = max(32, self.context_window // 2)


class ModelConfig:
    def __init__(
        self,
        bert_model_name: str = "bert-base-cased",
        tokenizer_name: Optional[str] = None,
        sentence_transformer_name: str = "all-MiniLM-L6-v2",
        attn_implementation: str = "eager",
    ):
        self.bert_model_name = bert_model_name
        self.tokenizer_name = tokenizer_name or bert_model_name
        self.sentence_transformer_name = sentence_transformer_name
        self.attn_implementation = attn_implementation


class LRU:
    def __init__(self, limit: int):
        self.limit = limit
        self.store: Dict[str, torch.Tensor] = {}
        self.order: List[str] = []

    def get(self, k: str) -> Optional[torch.Tensor]:
        v = self.store.get(k)
        if v is not None:
            try:
                self.order.remove(k)
            except ValueError:
                pass
            self.order.append(k)
        return v

    def set(self, k: str, v: torch.Tensor) -> None:
        if k in self.store:
            self.store[k] = v
            try:
                self.order.remove(k)
            except ValueError:
                pass
            self.order.append(k)
            return
        if len(self.order) >= self.limit:
            old = self.order.pop(0)
            self.store.pop(old, None)
        self.store[k] = v
        self.order.append(k)


class SemanticMemoryCache:
    def __init__(self, config: GenerationConfig):
        self.config = config
        cd = int(config.embedding_dim * config.compression_ratio)
        self.compress = nn.Linear(config.embedding_dim, cd).to(config.device)
        self.decompress = nn.Linear(cd, config.embedding_dim).to(config.device)
        self.bank: Dict[str, torch.Tensor] = {}

    def compress_4d(self, M: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            T, B, K, D = M.shape
            X = M.reshape(T * B * K, D)
            Z = self.compress(X)
            return Z.reshape(T, B, K, -1)

    def decompress_4d(self, Z: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            T, B, K, C = Z.shape
            X = Z.reshape(T * B * K, C)
            Y = self.decompress(X)
            return Y.reshape(T, B, K, -1)

    def influence(self, M: torch.Tensor, key: str) -> torch.Tensor:
        if not self.config.use_semantic_memory or key not in self.bank:
            return M
        Z = self.bank[key].to(self.config.device)
        D = self.decompress_4d(Z)
        alpha = self.config.memory_alpha
        beta = self.config.memory_beta
        return (1 - alpha) * M + alpha * (beta * D + (1 - beta) * M)

    def update(self, M: torch.Tensor, key: str) -> None:
        if not self.config.use_semantic_memory:
            return
        Z = self.compress_4d(M)
        self.bank[key] = Z.detach().cpu()


class SequenceMemoryCache:
    def __init__(self, config: GenerationConfig, encoder: SentenceTransformer):
        self.config = config
        self.encoder = encoder
        self.store: Dict[str, List[Dict[str, Union[str, torch.Tensor]]]] = {}
        self.th = 0.7

    def enc(self, text: str) -> torch.Tensor:
        with torch.inference_mode():
            e = self.encoder.encode(text, convert_to_tensor=True)
            return e.to(self.config.device)

    def add(self, key: str, text: str) -> None:
        if not self.config.use_sequence_memory:
            return
        v = {"text": text, "embedding": self.enc(text)}
        L = self.store.setdefault(key, [])
        L.append(v)
        if len(L) > self.config.sequence_cache_size:
            L.pop(0)

    def similar(self, query: str, key: Optional[str] = None) -> List[str]:
        if not self.config.use_sequence_memory:
            return []
        if key is None:
            if not self.store:
                return []
            key = next(iter(self.store))
        L = self.store.get(key, [])
        if not L:
            return []
        with torch.inference_mode():
            q = self.enc(query)
            r: List[str] = []
            for s in L:
                sim = F.cosine_similarity(q, s["embedding"], dim=-1).item()
                if sim >= self.th:
                    r.append(s["text"])
            return r


class Punct:
    SENT_END = {".", "!", "?"}
    PUNCT = {".", "!", "?", ",", ";", ":", ")", "]", "}", "…", "—", "–"}

    @staticmethod
    def fmt_wp(tok: str, prev: Optional[str] = None) -> str:
        if not tok:
            return ""
        sub = tok.startswith("##")
        tok = tok.replace("##", "").strip()
        if not tok:
            return ""
        if tok in Punct.SENT_END:
            return tok + "\n"
        if tok in Punct.PUNCT:
            return tok
        return tok if sub else " " + tok

    @staticmethod
    def fmt_sp(tok: str, prev: Optional[str] = None) -> str:
        if not tok:
            return ""
        new = tok.startswith("▁")
        tok = tok.replace("▁", "").replace("##", "").strip()
        if not tok:
            return ""
        if tok in Punct.SENT_END:
            return tok + "\n"
        if tok in Punct.PUNCT:
            return tok
        return (" " + tok) if new else tok


class Gate(nn.Module):
    """Learned blend policy. Input: (z_mean, c_mean, t/T, coh_ema, entropy, anchor_dist).
    Output: scalar weight w ∈ (0,1) controlling BERT-score vs coherence-score fusion.
    The extra signals let the gate learn to tighten coherence when drifting (low coh_ema),
    boost BERT confidence when entropy is low, and re-anchor when anchor distance grows."""

    def __init__(self):
        super().__init__()
        self.m = nn.Sequential(
            nn.Linear(6, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward(
        self,
        z_mean: torch.Tensor,
        c_mean: torch.Tensor,
        t: torch.Tensor,
        coh_ema: torch.Tensor,
        entropy: torch.Tensor,
        anchor_dist: torch.Tensor,
    ) -> torch.Tensor:
        v = torch.stack([
            z_mean.reshape(()),
            c_mean.reshape(()),
            t.reshape(()),
            coh_ema.reshape(()),
            entropy.reshape(()),
            anchor_dist.reshape(()),
        ]).unsqueeze(0)
        return torch.sigmoid(self.m(v)).squeeze()


class ParallelBERTGenerator(nn.Module):
    def __init__(self, config: GenerationConfig, model_config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        self.mc = model_config

        self.bert = AutoModelForMaskedLM.from_pretrained(
            model_config.bert_model_name,
            attn_implementation=model_config.attn_implementation,
        ).to(config.device)
        self.bert.eval()

        self.tok = AutoTokenizer.from_pretrained(model_config.tokenizer_name)
        self.sent = SentenceTransformer(model_config.sentence_transformer_name, trust_remote_code=True)

        self.fproj = nn.Linear(config.embedding_dim, config.embedding_dim).to(config.device)
        self.bproj = nn.Linear(config.embedding_dim, config.embedding_dim).to(config.device)
        self.xproj = nn.Linear(config.embedding_dim, config.embedding_dim).to(config.device)
        self.cproj = nn.Linear(3 * config.embedding_dim, config.embedding_dim).to(config.device)

        self.sem = SemanticMemoryCache(config)
        self.seq = SequenceMemoryCache(config, self.sent)

        probe = self.tok.tokenize("hello world")
        self.fmt = Punct.fmt_sp if any("▁" in x for x in probe) else Punct.fmt_wp

        self.cache = LRU(config.cache_limit)
        self.gate = Gate().to(config.device) if config.use_learned_gate else None

        self.sem2bert = (
            nn.Linear(384, config.embedding_dim, bias=False).to(config.device).eval()
            if config.use_space_alignment
            else None
        )

        self._pw = config.phrase_window
        self._progress = 0.0
        self._last_mean_coh: Optional[float] = None

        # Homeostasis state — updated each token, read by _coherence/_fuse_sample
        self._coh_ema: float = 0.5          # running coherence estimate
        self._last_entropy: float = 1.0     # entropy of last sampling distribution
        self._tau_eff: float = config.tau_coherence      # effective (modulated) tau
        self._p_nucleus_eff: float = config.p_nucleus    # effective nucleus threshold
        self._anchor_alpha_eff: float = config.coherence_anchor_alpha
        # Per-position entropy log for entropy-gated Gibbs (cleared each generate call)
        self._entropy_log: List[Tuple[int, float]] = []

        vocab = self.bert.config.vocab_size
        self._token_prior = torch.ones(vocab, device=config.device)
        for i in range(vocab):
            s = self.tok.convert_ids_to_tokens([i])[0] or ""
            c = s.replace("##", "").replace("▁", "")
            a = any(ch.isalpha() for ch in c)
            d = any(ch.isdigit() for ch in c)
            if not a and d:
                self._token_prior[i] = 0.5
            if not a and not d:
                self._token_prior[i] = 0.3

    def encode_cached(self, texts: Union[str, List[str]]) -> torch.Tensor:
        key = f"S|{texts}|E" if isinstance(texts, str) else f"L|{'\u241f'.join(texts)}|E"
        v = self.cache.get(key)
        if v is not None:
            return v
        with torch.inference_mode():
            e = self.sent.encode(texts, convert_to_tensor=True)
            e = e.to(self.config.device)
            self.cache.set(key, e)
            return e

    def _mask_preds(self, inp: torch.Tensor, att: torch.Tensor, pos: List[int]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        with torch.inference_mode():
            X = inp.clone()
            for b in range(inp.size(0)):
                X[b, pos[b]] = self.tok.mask_token_id
            out = self.bert(X, attention_mask=att).logits

            valid = torch.ones(out.size(-1), device=self.config.device, dtype=torch.bool)
            specials = {
                self.tok.pad_token_id,
                getattr(self.tok, "cls_token_id", None),
                getattr(self.tok, "sep_token_id", None),
                getattr(self.tok, "mask_token_id", None),
                getattr(self.tok, "unk_token_id", None),
            }
            specials = {t for t in specials if t is not None}
            if specials:
                valid[list(specials)] = False

            R: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for b in range(inp.size(0)):
                logits = out[b, pos[b]]
                logits[~valid] = float("-inf")
                logits = logits + torch.log(self._token_prior + 1e-8)
                k = max(self.config.top_k, self.config.batch_size)
                if self.config.use_coherence_scoring and self._last_mean_coh is not None and self._last_mean_coh < 0.65:
                    k = int(min(k * 2, max(16, int(k * 1.5))))
                topk = min(k, int(valid.sum().item()))
                tk_vals, tk_idx = torch.topk(logits, topk)
                R.append((tk_vals, tk_idx))
            return R

    def _valid_token_mask(self, size: int) -> torch.Tensor:
        valid = torch.ones(size, device=self.config.device, dtype=torch.bool)
        specials = {
            self.tok.pad_token_id,
            getattr(self.tok, "cls_token_id", None),
            getattr(self.tok, "sep_token_id", None),
            getattr(self.tok, "mask_token_id", None),
            getattr(self.tok, "unk_token_id", None),
        }
        specials = {t for t in specials if t is not None}
        if specials:
            valid[list(specials)] = False
        return valid

    def _answer_from_span(self, ids: torch.Tensor, start: int, end: int) -> str:
        if end <= start:
            return ""
        return self.tok.decode(ids[0, start:end], skip_special_tokens=True)

    def _init_matrix(self, init_text: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not init_text:
            init_text = " "
        batch_texts = [init_text]
        T = self.tok(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        ids = T["input_ids"].to(self.config.device)
        att = T["attention_mask"].to(self.config.device)

        seq_len = ids.size(1)
        # Cap matrix size at BERT's limit instead of unbounded growth
        max_total = 512
        total = min(seq_len + self.config.max_length, max_total)
        M = torch.zeros(
            total,
            1,
            self.config.num_candidates,
            self.config.embedding_dim,
            device=self.config.device,
        )

        with torch.inference_mode():
            h = self.bert.bert(ids, attention_mask=att).last_hidden_state
            L = int(att[0].sum().item())
            if L > 0:
                M[:L, 0, 0, :] = h[0, :L]

        key = init_text[:50]
        if self.config.use_semantic_memory:
            M = self.sem.influence(M, key)
        return M, ids, att

    def _bidir(self, pos: int, emb: torch.Tensor, M: torch.Tensor, W: int) -> torch.Tensor:
        if not self.config.use_bidirectional_context:
            return emb
        B = emb.shape[0]
        T = M.shape[0]
        s = max(0, pos - W)
        e = min(T, pos + W + 1)

        f = M[s:pos, :, 0, :].transpose(0, 1)
        b = M[pos + 1 : e, :, 0, :].transpose(0, 1)

        if f.size(1) == 0:
            f = torch.empty((B, 0, self.config.embedding_dim), device=self.config.device)
        if b.size(1) == 0:
            b = torch.empty((B, 0, self.config.embedding_dim), device=self.config.device)

        with torch.inference_mode():
            fa = (
                F.scaled_dot_product_attention(
                    emb.unsqueeze(1),
                    self.fproj(f),
                    self.fproj(f),
                    attn_mask=None,
                    is_causal=False,
                ).squeeze(1)
                if f.size(1) > 0
                else torch.zeros_like(emb)
            )
            ba = (
                F.scaled_dot_product_attention(
                    emb.unsqueeze(1),
                    self.bproj(b),
                    self.bproj(b),
                    attn_mask=None,
                    is_causal=False,
                ).squeeze(1)
                if b.size(1) > 0
                else torch.zeros_like(emb)
            )
            ic = M[max(0, pos - W) : pos, :, 0, :].transpose(0, 1)
            ca = (
                F.scaled_dot_product_attention(
                    emb.unsqueeze(1),
                    self.xproj(ic),
                    self.xproj(ic),
                    attn_mask=None,
                    is_causal=False,
                ).squeeze(1)
                if ic.size(1) > 0
                else torch.zeros_like(emb)
            )
        return self.cproj(torch.cat([fa, ba, ca], dim=-1))

    def _coherence(
        self,
        cands: List[str],
        ctx: str,
        anchor_emb: Optional[torch.Tensor] = None,
        anchor_alpha: float = 0.0,
    ) -> torch.Tensor:
        if not cands:
            return torch.tensor([], device=self.config.device)
        if not self.config.use_coherence_scoring:
            return torch.ones(len(cands), device=self.config.device)

        with torch.inference_mode():
            toks = self.tok.tokenize(ctx)
            w = int(min(len(toks), self._pw))
            ctx_tail = self.tok.convert_tokens_to_string(toks[-w:]) if w > 0 else ""
            lens = torch.tensor(
                [1.0 if len(self.tok.tokenize(x)) > 1 else 0.7 for x in cands],
                device=self.config.device,
            )
            phrases = [
                self.tok.convert_tokens_to_string(toks[-w:] + [x]) if w > 0 else x
                for x in cands
            ]
            if not ctx_tail:
                return lens

            sim_seqs: List[str] = []
            if self.config.use_sequence_memory:
                key = ctx_tail[:50]
                sim_seqs = self.seq.similar(ctx_tail, key)
                if sim_seqs:
                    boost = torch.ones(len(cands), device=self.config.device)
                    for i, x in enumerate(cands):
                        for s in sim_seqs:
                            if x in s:
                                boost[i] *= 1.2
                                break
                    lens = lens * boost

            P = self.encode_cached(phrases)
            C0 = self.encode_cached(ctx_tail)
            if P.dim() == 1:
                P = P.unsqueeze(0)
            elif P.dim() != 2:
                P = P.reshape(len(cands), -1)
            C0 = C0.reshape(-1)
            if self.sem2bert is not None:
                P = self.sem2bert(P)
                C0 = self.sem2bert(C0)

            # Blend in the user-input anchor to prevent the context embedding
            # from drifting entirely into the model's own generated output.
            if anchor_emb is not None and anchor_alpha > 0.0:
                anchor = anchor_emb.to(C0.device).reshape(-1)
                if self.sem2bert is not None and anchor.numel() != C0.numel():
                    anchor = self.sem2bert(anchor)
                C0 = F.normalize(
                    (1.0 - anchor_alpha) * C0 + anchor_alpha * anchor,
                    dim=-1,
                )

            abs_sim = F.cosine_similarity(P, C0.unsqueeze(0).expand(len(cands), -1), dim=-1)
            abs01 = ((abs_sim + 1) / 2).clamp(0, 1)
            gain = abs_sim - 1.0
            c = 0.7 * gain.abs() + 0.3 * abs01
            c = torch.pow(c.clamp(1e-6, 1 - 1e-6), 1 / max(1e-6, self._tau_eff))
            return c * lens

    def _left_to_right_bias(self, cands: List[str], ctx: str) -> torch.Tensor:
        if not cands:
            return torch.tensor([], device=self.config.device)
        toks = self.tok.tokenize(ctx)
        last = toks[-1] if toks else ""
        bias = torch.ones(len(cands), device=self.config.device)
        for i, t in enumerate(cands):
            s = t.replace("##", "").replace("▁", "")
            if last.startswith("##") and t.startswith("##"):
                bias[i] *= 1.2
            if last and last not in Punct.SENT_END and s in {",", ";", ":"}:
                bias[i] *= 0.7
        return bias

    def _entity_bias(self, ctx: str, cand_tokens: List[str]) -> torch.Tensor:
        if not self.config.use_entity_bias:
            return torch.ones(len(cand_tokens), device=self.config.device)
        toks = self.tok.tokenize(ctx)
        ents = [t for t in toks if t[0].isupper() and t[1:].islower()]
        if not ents:
            return torch.ones(len(cand_tokens), device=self.config.device)
        boost = torch.ones(len(cand_tokens), device=self.config.device)
        for i, t in enumerate(cand_tokens):
            x = t.replace("##", "")
            if x in ents:
                boost[i] *= self.config.entity_boost
        return boost

    def _style_weights(self, cands: List[str], step: int) -> torch.Tensor:
        v = torch.ones(len(cands), device=self.config.device)
        early = step < 8
        for i, t in enumerate(cands):
            s = t.replace("##", "").replace("▁", "").strip()
            a = any(ch.isalpha() for ch in s)
            end = s in {".", "!", "?", ";", ":"}
            if early and not a:
                v[i] *= 0.2
            if early and end:
                v[i] *= 0.5
        return v

    def _fuse_sample(
        self,
        top_ids: torch.Tensor,
        top_logits: torch.Tensor,
        coh: torch.Tensor,
        history_ids: List[int],
        entity_boost: torch.Tensor,
        position_ratio: float,
    ) -> int:
        z = F.log_softmax(top_logits, dim=-1)
        z = (z - z.mean()) / (z.std() + 1e-6)
        c = (coh - coh.mean()) / (coh.std() + 1e-6)
        c = c * entity_boost

        if self.gate is not None:
            dev = z.device
            w = self.gate(
                z_mean=z.mean(),
                c_mean=c.mean(),
                t=torch.tensor(position_ratio, device=dev),
                coh_ema=torch.tensor(self._coh_ema, device=dev),
                entropy=torch.tensor(self._last_entropy, device=dev),
                anchor_dist=torch.tensor(1.0 - self._coh_ema, device=dev),
            )
            p = z.exp() * w + F.softmax(c, dim=-1) * (1 - w)
        else:
            p = z.exp() * c.clamp(min=0.1)

        if history_ids:
            win = (
                history_ids[-self.config.repetition_window :]
                if self.config.repetition_window > 0
                else history_ids
            )
            counts: Dict[int, int] = {}
            for h in win:
                counts[h] = counts.get(h, 0) + 1
            rp = torch.ones_like(p)
            for i, tid in enumerate(top_ids.tolist()):
                if tid in counts:
                    rp[i] *= self.config.repetition_penalty ** counts[tid]
            p = p * rp

        p = F.softmax(p / max(self.config.base_temperature, 1e-3), dim=-1)
        if self.config.use_adaptive_temperature:
            ent = -(p * (p + 1e-8).log()).sum()
            self._last_entropy = float(ent.item())
            scale = 1.0 + 0.5 * (1.0 - ent.clamp(0.0, 4.0) / 4.0)
            p = F.softmax(p / scale, dim=-1)

        if self._p_nucleus_eff < 1.0:
            sorted_p, sorted_idx = torch.sort(p, descending=True)
            cum = torch.cumsum(sorted_p, dim=-1)
            mask = cum <= self._p_nucleus_eff
            mask[..., 0] = True
            filtered = sorted_p * mask
            filtered = filtered / filtered.sum()
            idx_in_sorted = torch.multinomial(filtered, 1).item()
            return sorted_idx[idx_in_sorted].item()
        else:
            idx = torch.multinomial(p, 1).item()
            return idx

    def _update_homeostasis(self, coh_mean: float) -> None:
        """EMA-track coherence and modulate effective tau/nucleus/anchor in response.
        Below the low threshold the system tightens (sharper semantic veto, stronger
        anchor, narrower nucleus). Above the high threshold it relaxes slightly.
        Changes are incremental so the system can't oscillate too violently."""
        if not self.config.use_coherence_homeostasis:
            return
        a = self.config.coherence_ema_alpha
        self._coh_ema = (1.0 - a) * self._coh_ema + a * coh_mean
        lo = self.config.coherence_low_threshold
        hi = self.config.coherence_high_threshold

        if self._coh_ema < lo:
            # Drifting — tighten all pressures
            self._tau_eff = max(0.5, self._tau_eff * 0.95)
            self._p_nucleus_eff = max(0.6, self._p_nucleus_eff - 0.01)
            self._anchor_alpha_eff = min(1.0, self._anchor_alpha_eff + 0.03)
        elif self._coh_ema > hi:
            # Stable — ease off toward config values (don't over-tighten)
            self._tau_eff = min(self.config.tau_coherence, self._tau_eff * 1.02)
            self._p_nucleus_eff = min(self.config.p_nucleus, self._p_nucleus_eff + 0.005)
            self._anchor_alpha_eff = max(
                self.config.coherence_anchor_alpha, self._anchor_alpha_eff - 0.01
            )

    def generate_stream(self, initial_text: str, num_tokens: int) -> Iterator[str]:
        M, ids, att = self._init_matrix(initial_text)

        # Sliding window: enforce BERT's 512-token limit
        max_bert_ctx = 510  # Leave room for special tokens
        if ids.size(1) > max_bert_ctx:
            ids = ids[:, -max_bert_ctx:]
            att = att[:, -max_bert_ctx:]

        L0 = ids.size(1)

        # Compute available generation capacity
        avail = max_bert_ctx - L0
        T_gen = min(num_tokens, max(0, avail))

        if T_gen <= 0:
            return  # No room to generate

        P_ids = torch.full(
            (1, L0 + T_gen),
            self.tok.pad_token_id,
            device=self.config.device,
        )
        P_ids[:, :L0] = ids
        P_att = torch.zeros((1, L0 + T_gen), device=self.config.device)
        P_att[:, :L0] = 1

        seq = ids[0].tolist()
        prev: Optional[str] = None
        out = ""

        key = initial_text[:50]
        simseq = self.seq.similar(initial_text, key) if self.config.use_sequence_memory else []

        # Reset homeostasis to config defaults at the start of each response
        self._coh_ema = 0.5
        self._tau_eff = self.config.tau_coherence
        self._p_nucleus_eff = self.config.p_nucleus
        self._anchor_alpha_eff = self.config.coherence_anchor_alpha
        self._entropy_log = []

        # Encode the full prompt once as a stable semantic anchor.
        # This keeps coherence scoring tethered to the user's input rather than
        # drifting into the model's own generated context as output accumulates.
        anchor_emb: Optional[torch.Tensor] = None
        if self.config.coherence_anchor_alpha > 0.0:
            anchor_emb = self.encode_cached(initial_text)

        for pos in range(L0, L0 + T_gen):
            P_att[:, pos] = 1
            preds = self._mask_preds(P_ids, P_att, [pos])
            with torch.inference_mode():
                h = self.bert.bert(input_ids=P_ids, attention_mask=P_att).last_hidden_state[:, pos]
                attn = self._bidir(pos, h, M, self.config.context_window)

            self._progress = (pos - L0) / max(1, T_gen)
            self._pw = min(
                self.config.phrase_window,
                int(self.config.phrase_window * (1.0 + self.config.dynamic_window_gamma * (pos - L0))),
            )

            top_logits, top_ids = preds[0]
            if top_ids.numel() == 0:
                continue
            cand = [self.tok.convert_ids_to_tokens([i.item()])[0] for i in top_ids]
            ctx_text = out

            # Taper anchor alpha: start strong (grounded in user input),
            # ease off as the response develops its own coherent thread.
            # _anchor_alpha_eff is also modulated upward by homeostasis when drifting.
            alpha = self._anchor_alpha_eff
            if self.config.coherence_anchor_taper:
                early_boost = 1.0 + (1.0 - self._progress)
                alpha = min(1.0, alpha * early_boost)

            coh = self._coherence(cand, ctx_text, anchor_emb=anchor_emb, anchor_alpha=alpha)
            l2r = self._left_to_right_bias(cand, ctx_text)
            coh = coh * l2r

            if simseq:
                sboost = torch.ones_like(coh)
                cur = out
                inf = max(0.0, 1.0 - self._progress)
                for i, t in enumerate(cand):
                    for s in simseq:
                        if cur and cur in s:
                            p = s.find(cur) + len(cur)
                            if p < len(s) and t in s[p : p + len(t) + 5]:
                                sboost[i] *= 1.0 + 0.3 * inf
                                break
                coh = coh * sboost

            self._last_mean_coh = float(coh.mean().item())
            self._update_homeostasis(self._last_mean_coh)
            eboost = self._entity_bias(out, cand) * self._style_weights(cand, (pos - L0))
            sel_tid = self._fuse_sample(top_ids, top_logits, coh, seq, eboost, (pos - L0) / max(1, T_gen))
            tid = top_ids[sel_tid].item()

            P_ids[0, pos] = tid
            seq.append(tid)
            M[pos, 0, 0, :] = attn[0]
            self._entropy_log.append((pos, self._last_entropy))

            tok = self.tok.convert_ids_to_tokens([tid])[0]
            fmt = self.fmt(tok, prev)
            if fmt:
                out += fmt
                yield fmt
            prev = tok

        self.sem.update(M, key)
        if self.config.use_sequence_memory:
            self.seq.add(key, initial_text + out)

    def _denoise_pattern_positions(self, gen_start: int, gen_end: int, pass_idx: int) -> List[int]:
        """Return ordered positions to denoise for one pass according to denoise_pattern."""
        positions = list(range(gen_start, gen_end))
        pattern = self.config.denoise_pattern

        if pattern == "random":
            import random as _random
            _random.shuffle(positions)

        elif pattern == "entropy":
            # Sort by draft entropy descending — most uncertain positions first.
            # _entropy_log contains (absolute_pos, entropy) pairs from generate_stream.
            entropy_map = {p: e for p, e in getattr(self, "_entropy_log", [])}
            positions.sort(key=lambda p: entropy_map.get(p, 0.0), reverse=True)

        elif pattern == "span":
            # Pick a random contiguous span covering ~half the positions each pass;
            # alternate which half by offsetting with pass_idx so full coverage across passes.
            import random as _random
            n = len(positions)
            span = max(1, n // 2)
            max_start = max(0, n - span)
            # Deterministic offset per pass so successive passes cover different regions
            offset = (pass_idx * span) % max(1, max_start + 1)
            positions = positions[offset: offset + span]

        elif pattern == "checkerboard":
            # Even-indexed on even passes, odd-indexed on odd passes
            parity = pass_idx % 2
            positions = [p for i, p in enumerate(positions) if (p - gen_start) % 2 == parity]

        else:  # sequential (default)
            if pass_idx % 2 == 1:
                positions.reverse()

        return positions

    def _denoise_batch(
        self,
        ids: torch.Tensor,
        att: torch.Tensor,
        positions: List[int],
        gen_start: int,
        gen_end: int,
        anchor_emb: Optional[torch.Tensor],
        pass_idx: int,
    ) -> Iterator[Dict[str, Any]]:
        """
        Batched denoise: group positions into chunks, run one BERT forward pass per
        chunk (each row masks a different position), then score and accept/reject
        per position.  This replaces the old one-forward-pass-per-token loop.
        """
        if not positions:
            return

        valid = self._valid_token_mask(self.bert.config.vocab_size)
        topk_k = min(max(1, int(self.config.denoise_top_k)), int(valid.sum().item()))
        temp = max(float(self.config.denoise_temperature), 1e-3)
        alpha = self.config.coherence_anchor_alpha
        margin = float(self.config.denoise_accept_margin)
        seq_len = ids.size(1)

        # chunk_size == 0 means process all positions at once
        chunk_size = int(self.config.denoise_batch_size) or len(positions)

        for chunk_start in range(0, len(positions), chunk_size):
            chunk = positions[chunk_start: chunk_start + chunk_size]

            # Build (chunk_len, seq_len) batch — each row masks one position
            batch_ids = ids.expand(len(chunk), -1).clone()  # (C, S)
            batch_att = att.expand(len(chunk), -1)          # (C, S) — no clone needed
            for i, pos in enumerate(chunk):
                batch_ids[i, pos] = self.tok.mask_token_id

            with torch.no_grad():
                all_logits = self.bert(batch_ids, attention_mask=batch_att).logits  # (C, S, V)

            for i, pos in enumerate(chunk):
                old_id = int(ids[0, pos].item())
                if old_id >= valid.numel() or not bool(valid[old_id].item()):
                    continue

                logits = all_logits[i, pos].clone()
                logits[~valid] = float("-inf")
                logits = logits + torch.log(self._token_prior + 1e-8)

                tk_vals, tk_idx = torch.topk(logits, topk_k)
                cand = [self.tok.convert_ids_to_tokens([j.item()])[0] for j in tk_idx]
                ctx = self.tok.decode(ids[0, :pos], skip_special_tokens=True)

                coh = self._coherence(cand, ctx, anchor_emb=anchor_emb, anchor_alpha=alpha)
                coh = coh * self._left_to_right_bias(cand, ctx)
                eb = self._entity_bias(ctx, cand) * self._style_weights(cand, pos - gen_start)

                fused = (tk_vals / temp) + torch.log(coh.clamp(min=1e-6)) + torch.log(eb.clamp(min=1e-6))
                best_idx = int(torch.argmax(fused).item())
                new_id = int(tk_idx[best_idx].item())

                if new_id == old_id:
                    continue

                # Score the incumbent token for margin check (reuse logits already computed)
                old_tok = self.tok.convert_ids_to_tokens([old_id])[0]
                old_coh = self._coherence([old_tok], ctx, anchor_emb=anchor_emb, anchor_alpha=alpha)
                old_coh = old_coh * self._left_to_right_bias([old_tok], ctx)
                old_eb = self._entity_bias(ctx, [old_tok]) * self._style_weights([old_tok], pos - gen_start)
                old_score = (logits[old_id] / temp) + torch.log(old_coh[0].clamp(min=1e-6)) + torch.log(old_eb[0].clamp(min=1e-6))
                new_score = fused[best_idx]

                if float(new_score.item()) <= float(old_score.item()) + margin:
                    continue

                old_text = self.tok.decode([old_id], skip_special_tokens=True)
                new_text = self.tok.decode([new_id], skip_special_tokens=True)
                ids[0, pos] = new_id

                yield {
                    "phase": "denoise",
                    "position": pos - gen_start,
                    "absolute_position": pos,
                    "old_token": old_text,
                    "new_token": new_text,
                    "old_score": float(old_score.item()),
                    "new_score": float(new_score.item()),
                    "pass": pass_idx + 1,
                    "pattern": self.config.denoise_pattern,
                    "text": self._answer_from_span(ids, gen_start, gen_end),
                }

    def denoise_stream(self, initial_text: str, draft_text: str, passes: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        passes = self.config.denoise_passes if passes is None else passes
        if passes <= 0 or not draft_text:
            return

        full_text = initial_text + draft_text
        ids = self.tok.encode(full_text, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        prompt_ids = self.tok.encode(initial_text, return_tensors="pt", add_special_tokens=True).to(self.config.device)

        if ids.size(1) > 512 or prompt_ids.size(1) > ids.size(1):
            return

        att = torch.ones_like(ids)
        gen_start = max(1, int(prompt_ids.size(1)) - 1)
        gen_end = max(gen_start, int(ids.size(1)) - 1)
        if gen_start >= gen_end:
            return

        old_pw = getattr(self, "_pw", self.config.phrase_window)
        self._pw = self.config.phrase_window
        anchor_emb = self.encode_cached(initial_text) if self.config.coherence_anchor_alpha > 0.0 else None

        try:
            for pass_idx in range(int(passes)):
                positions = self._denoise_pattern_positions(gen_start, gen_end, pass_idx)
                yield from self._denoise_batch(ids, att, positions, gen_start, gen_end, anchor_emb, pass_idx)
        finally:
            self._pw = old_pw

    def generate_with_denoise_stream(self, initial_text: str, num_tokens: int) -> Iterator[Dict[str, Any]]:
        draft = ""

        for token in self.generate_stream(initial_text, num_tokens):
            draft += token
            yield {
                "phase": "draft",
                "token": token,
                "text": draft,
            }

        if not self.config.use_denoise:
            return

        final_text = draft
        edits = 0
        yield {
            "phase": "denoise_start",
            "text": final_text,
            "passes": int(self.config.denoise_passes),
        }

        for event in self.denoise_stream(initial_text, draft, self.config.denoise_passes):
            final_text = event["text"]
            edits += 1
            if self.config.stream_denoise:
                yield event

        yield {
            "phase": "denoise_complete",
            "text": final_text,
            "edits": edits,
        }

    def _backedit(self, initial_text: str, full_text: str) -> str:
        M = self.config.gibbs_every_M
        L = self.config.gibbs_span_L
        if M <= 0 or L <= 0:
            return full_text

        ids = self.tok.encode(full_text, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        att = torch.ones_like(ids)
        T = ids.size(1)

        # Entropy-gated: resample the highest-uncertainty spans rather than fixed intervals
        if self.config.use_entropy_gibbs and self._entropy_log:
            sorted_by_entropy = sorted(self._entropy_log, key=lambda x: -x[1])
            # Take the top-L highest-entropy positions as refinement targets
            target_positions = sorted(p for p, _ in sorted_by_entropy[:L])
            spans = []
            if target_positions:
                span_start = target_positions[0]
                prev = span_start
                for p in target_positions[1:]:
                    if p > prev + 2:  # gap — close current span, start new
                        spans.append((span_start, prev + 1))
                        span_start = p
                    prev = p
                spans.append((span_start, prev + 1))
            for a, b in spans:
                a = max(1, a)
                b = min(T - 1, b)
                for pos in range(a, b):
                    X = ids.clone()
                    X[0, pos] = self.tok.mask_token_id
                    with torch.no_grad():
                        logits = self.bert(X, attention_mask=att).logits[0, pos]
                    valid = torch.ones_like(logits, dtype=torch.bool)
                    spec = [t for t in [
                        self.tok.pad_token_id,
                        getattr(self.tok, "cls_token_id", None),
                        getattr(self.tok, "sep_token_id", None),
                        self.tok.mask_token_id,
                        getattr(self.tok, "unk_token_id", None),
                    ] if t is not None]
                    valid[spec] = False
                    logits[~valid] = float("-inf")
                    logits = logits + torch.log(self._token_prior + 1e-8)
                    topk = min(self.config.top_k, int(valid.sum().item()))
                    tk_vals, tk_idx = torch.topk(logits, topk)
                    cand = [self.tok.convert_ids_to_tokens([i.item()])[0] for i in tk_idx]
                    ctx = self.tok.decode(ids[0, :pos])
                    coh = self._coherence(cand, ctx) * self._left_to_right_bias(cand, ctx)
                    eb = self._entity_bias(ctx, cand)
                    sel_tid = self._fuse_sample(tk_idx, tk_vals, coh, ids[0, :pos].tolist(), eb, 0.5)
                    ids[0, pos] = tk_idx[sel_tid]
            return self.tok.decode(ids[0], skip_special_tokens=True)

        # Fixed-interval fallback when entropy_gibbs is off
        for s in range(1, T - 1, M):
            a = max(1, s)
            b = min(T - 1, s + L)
            for pos in range(a, b):
                X = ids.clone()
                X[0, pos] = self.tok.mask_token_id
                with torch.no_grad():
                    logits = self.bert(X, attention_mask=att).logits[0, pos]
                valid = torch.ones_like(logits, dtype=torch.bool)
                spec = [
                    self.tok.pad_token_id,
                    getattr(self.tok, "cls_token_id", None),
                    getattr(self.tok, "sep_token_id", None),
                    getattr(self.tok, "mask_token_id", None),
                    getattr(self.tok, "unk_token_id", None),
                ]
                spec = [t for t in spec if t is not None]
                if spec:
                    valid[spec] = False
                logits[~valid] = float("-inf")
                logits = logits + torch.log(self._token_prior + 1e-8)
                topk = min(self.config.top_k, int(valid.sum().item()))
                tk_vals, tk_idx = torch.topk(logits, topk)
                cand = [self.tok.convert_ids_to_tokens([i.item()])[0] for i in tk_idx]
                ctx = self.tok.decode(ids[0, :pos])
                coh = self._coherence(cand, ctx) * self._left_to_right_bias(cand, ctx)
                eb = self._entity_bias(ctx, cand)
                sel_tid = self._fuse_sample(tk_idx, tk_vals, coh, ids[0, :pos].tolist(), eb, 0.5)
                ids[0, pos] = tk_idx[sel_tid]

        return self.tok.decode(ids[0], skip_special_tokens=True)

    def generate(self, initial_text: str, num_tokens: int) -> str:
        s = ""
        for event in self.generate_with_denoise_stream(initial_text, num_tokens):
            phase = event.get("phase")
            if phase in {"draft", "denoise", "denoise_complete"}:
                s = str(event.get("text", s))
        if not self.config.use_denoise:
            s = self._backedit(initial_text, s)
        if self.config.use_sequence_memory:
            key = initial_text[:50]
            self.seq.add(key, initial_text + s)
        return s
