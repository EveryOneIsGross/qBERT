"""
It might not be tracking config states correctly between truns f or the teacher to maintain and update.
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import shutil
import torch
from typing import Union, Optional, Dict, Any, List, Tuple, Literal
from datetime import datetime
from pathlib import Path
import json
import time
import os
import math
import ollama
from pydantic import BaseModel, Field

from qBERT import ParallelBERTGenerator as qBERTGenerator, GenerationConfig as qBERTConfig, ModelConfig
import re
from colorama import init, Fore, Back, Style
import yaml
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
import asyncio
import pickle
from urllib.parse import urlparse


MODEL_CONFIG_PATH = Path("config/model_config.yaml")


def load_model_config(path: Path = MODEL_CONFIG_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _config_section(config_data: Dict[str, Any], section: str) -> Dict[str, Any]:
    values = (config_data or {}).get(section, {})
    return values if isinstance(values, dict) else {}


def _drop_empty(values: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in values.items()
        if value is not None and value != ""
    }


def _generation_config_kwargs(config_data: Dict[str, Any]) -> Dict[str, Any]:
    section = _config_section(config_data, "qbert")
    valid_fields = set(getattr(qBERTConfig, "__dataclass_fields__", {}).keys())
    kwargs: Dict[str, Any] = {}
    for key, value in section.items():
        if key not in valid_fields or value is None:
            continue
        if key == "device" and str(value).strip().lower() in {"", "auto"}:
            continue
        kwargs[key] = value
    return kwargs


def create_qbert_config(config_data: Optional[Dict[str, Any]] = None) -> qBERTConfig:
    return qBERTConfig(**_generation_config_kwargs(config_data or load_model_config()))


def _model_config_kwargs(config_data: Dict[str, Any]) -> Dict[str, Any]:
    defaults = _config_section(config_data, "model_defaults")
    aliases = {
        "bert_model_name": ("bert_model_name", "bert_model"),
        "tokenizer_name": ("tokenizer_name", "tokenizer"),
        "sentence_transformer_name": ("sentence_transformer_name", "sentence_transformer"),
        "attn_implementation": ("attn_implementation", "attention_implementation"),
    }
    kwargs: Dict[str, Any] = {}
    for target, names in aliases.items():
        for name in names:
            value = defaults.get(name)
            if value is not None and value != "":
                kwargs[target] = value
                break
    return kwargs


def create_model_config(config_data: Optional[Dict[str, Any]] = None) -> ModelConfig:
    return ModelConfig(**_model_config_kwargs(config_data or load_model_config()))


def _pydantic_field_names(model_cls: Any) -> set:
    return set(getattr(model_cls, "model_fields", None) or getattr(model_cls, "__fields__", {}))


def create_auto_chat(config_data: Optional[Dict[str, Any]] = None) -> "AutoChat":
    section = _config_section(config_data or load_model_config(), "autochat")
    fields = _pydantic_field_names(AutoChat)
    return AutoChat(**{key: value for key, value in _drop_empty(section).items() if key in fields})


def _bool_default(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def autochat_session_defaults(config_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    section = _config_section(config_data or load_model_config(), "autochat_session")
    return {
        "num_tokens": int(section.get("num_tokens", 64)),
        "stream_mode": _bool_default(section.get("stream_mode"), True),
        "model_type": str(section.get("model_type", "qbert")),
    }


def ollama_kwargs_from_config(config_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config_data = config_data or load_model_config()
    ollama_section = _drop_empty(_config_section(config_data, "ollama"))
    memory_section = _drop_empty(_config_section(config_data, "memory"))
    allowed = {"max_history", "embed_model", "history_path", "host", "model_name"}
    kwargs = {key: value for key, value in ollama_section.items() if key in allowed}
    memory_aliases = {
        "vector_path": "memory_vector_path",
        "bm25_path": "memory_bm25_path",
        "dimension": "memory_dimension",
        "reflection_backfill_limit": "memory_reflection_backfill_limit",
        "vector_weight": "memory_vector_weight",
        "bm25_weight": "memory_bm25_weight",
    }
    for source, target in memory_aliases.items():
        if source in memory_section:
            kwargs[target] = memory_section[source]
    return kwargs


def teacher_kwargs_from_config(
    config_data: Optional[Dict[str, Any]],
    ollama_model_name: str,
) -> Tuple[Dict[str, Any], bool]:
    section = _config_section(config_data or load_model_config(), "teacher")
    use_ollama_model = _bool_default(section.get("use_ollama_model"), True)
    allowed = {"model_name", "min_confidence", "history_size", "bm25_path", "related_attempts"}
    kwargs = {
        key: value
        for key, value in _drop_empty(section).items()
        if key in allowed
    }
    if use_ollama_model or "model_name" not in kwargs:
        kwargs["model_name"] = ollama_model_name
    return kwargs, use_ollama_model


def _visual_line_count(text: str, width: int) -> int:
    width = max(20, width)
    lines = text.splitlines() or [""]
    return sum(max(1, (len(line) + width - 1) // width) for line in lines)


def normalize_ollama_host(host: Optional[str] = None) -> str:
    raw = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").strip()
    if not raw:
        return "http://127.0.0.1:11434"
    if "://" not in raw:
        raw = f"http://{raw}"

    parsed = urlparse(raw)
    scheme = parsed.scheme or "http"
    hostname = parsed.hostname or "127.0.0.1"
    if hostname in {"0.0.0.0", "::"}:
        hostname = "127.0.0.1"

    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = f"{hostname}:{parsed.port}" if parsed.port else hostname
    path = parsed.path if parsed.path not in {"", "/"} else ""
    return f"{scheme}://{netloc}{path}"


def redraw_denoise_preview(previous_lines: int, text: str, status: str) -> int:
    if previous_lines > 0:
        sys.stdout.write(f"\033[{previous_lines}A\033[J")
    else:
        sys.stdout.write("\n")

    title = "Denoise preview (not saved to conversation history)"
    plain = f"{title}\n{status}\n{text}"
    rendered = (
        f"{Fore.CYAN}{title}{Style.RESET_ALL}\n"
        f"{Fore.MAGENTA}{status}{Style.RESET_ALL}\n"
        f"{text}"
    )
    sys.stdout.write(rendered + "\n")
    sys.stdout.flush()
    cols = shutil.get_terminal_size((80, 24)).columns
    return _visual_line_count(plain, cols)


class ConversationEntry(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    input_text: str
    output_text: str
    model_name: str
    model_type: str
    config: Dict[str, Any]
    tokens_generated: int
    stream_mode: bool = True
    generation_time: float = 0.0


class SystemUpdate(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    update_type: str
    previous_value: Any
    new_value: Any
    model_type: str


class ModelResponse(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    model_name: str
    model_type: str
    response: str
    input_text: str
    full_response: Optional[str] = None
    system_prompt: Optional[str] = None
    input_prompt: Optional[str] = None
    reflection_context: Optional[str] = None
    config: Dict[str, Any]
    tokens: int
    generation_time: float = 0.0


class Logger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.response_log = self.log_dir / "responses.jsonl"
        self.system_log = self.log_dir / "system_updates.jsonl"
        self.context_log = self.log_dir / "context.jsonl"

    def log_response(self, entry: ModelResponse):
        with open(self.response_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.dict()) + "\n")

    def log_system_update(self, update: SystemUpdate):
        with open(self.system_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(update.dict()) + "\n")

    def log_context(self, context_data: Dict[str, Any]):
        with open(self.context_log, "a", encoding="utf-8") as f:
            entry = {
                "timestamp": datetime.now().isoformat(),
                **context_data
            }
            f.write(json.dumps(entry) + "\n")


class AutoChat(BaseModel):
    pause_duration: float = Field(default=2.0, description="Seconds to wait between exchanges")
    is_paused: bool = Field(default=False, description="Chat pause state")
    max_context_length: int = Field(default=4096, description="Maximum context length to maintain")
    conversation_history: List[str] = Field(default_factory=list)
    max_chunk_size: int = Field(default=512, description="Maximum tokens per chunk for Ollama responses")
    max_bert_context: int = Field(default=256, description="Maximum context tokens for BERT input")
    max_bert_response: int = Field(default=256, description="Maximum tokens for BERT response")


class Reflection(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    input_text: str
    reflection_text: str
    response_text: str
    model_name: str
    embedding: List[float]
    keywords: List[str]


class VectorIndex:
    def __init__(self, dimension: int = 1024, save_path: str = "data/vector_index.pkl"):
        self.dimension = dimension
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        if self.save_path.exists():
            with open(self.save_path, 'rb') as f:
                loaded = pickle.load(f)
                self.reflections = loaded.reflections
                self.embeddings = loaded.embeddings
                print(f"{Fore.CYAN}Loaded {len(self.reflections)} reflections from index{Style.RESET_ALL}")
        else:
            self.reflections: List[Reflection] = []
            self.embeddings: List[List[float]] = []

    def add_reflection(self, reflection: Reflection):
        if not reflection.embedding or all(x == 0 for x in reflection.embedding):
            print("Warning: Adding reflection with zero embedding")
        self.reflections.append(reflection)
        self.embeddings.append(reflection.embedding)

        with open(self.save_path, 'wb') as f:
            pickle.dump(self, f)

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Reflection]:
        return [item["reflection"] for item in self.search_with_scores(query_embedding, top_k)]

    def search_with_scores(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.embeddings:
            return []

        similarities = cosine_similarity(
            np.array(query_embedding).reshape(1, -1),
            np.array(self.embeddings)
        )[0]

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [
            {
                "index": int(i),
                "reflection": self.reflections[int(i)],
                "score": float(similarities[int(i)]),
                "source": "vector",
            }
            for i in top_indices
        ]


class PersistentBM25Index:
    def __init__(
        self,
        save_path: str = "data/teacher_bm25_index.json",
        k1: float = 1.5,
        b: float = 0.75,
        max_docs: int = 2000,
    ):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.k1 = k1
        self.b = b
        self.max_docs = max_docs
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.inverted_index: Dict[str, Dict[str, int]] = {}
        self.doc_order: List[str] = []
        self.avgdl = 0.0
        self._load()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9_][a-z0-9_'-]*", text.lower())

    def _load(self):
        if not self.save_path.exists():
            return
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.documents = data.get("documents", {})
            self.inverted_index = data.get("inverted_index", {})
            self.doc_order = data.get("doc_order", list(self.documents.keys()))
            self.avgdl = float(data.get("avgdl", 0.0))
            self._recompute_stats()
            print(f"{Fore.CYAN}Loaded {len(self.documents)} BM25 memory records{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not load BM25 memory index: {e}{Style.RESET_ALL}")
            self.documents = {}
            self.inverted_index = {}
            self.doc_order = []
            self.avgdl = 0.0

    def _save(self):
        payload = {
            "documents": self.documents,
            "inverted_index": self.inverted_index,
            "doc_order": self.doc_order,
            "avgdl": self.avgdl,
        }
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def _recompute_stats(self):
        total_len = sum(int(doc.get("length", 0)) for doc in self.documents.values())
        self.avgdl = total_len / max(1, len(self.documents))

    def _remove_document(self, doc_id: str):
        doc = self.documents.pop(doc_id, None)
        if not doc:
            return
        for term in (doc.get("term_freq") or {}).keys():
            postings = self.inverted_index.get(term)
            if not postings:
                continue
            postings.pop(doc_id, None)
            if not postings:
                self.inverted_index.pop(term, None)
        try:
            self.doc_order.remove(doc_id)
        except ValueError:
            pass
        self._recompute_stats()

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        persist: bool = True,
    ):
        tokens = self.tokenize(text)
        if not tokens:
            return
        if doc_id in self.documents:
            self._remove_document(doc_id)

        term_freq: Dict[str, int] = {}
        for tok in tokens:
            term_freq[tok] = term_freq.get(tok, 0) + 1

        self.documents[doc_id] = {
            "text": text,
            "metadata": metadata or {},
            "length": len(tokens),
            "term_freq": term_freq,
        }
        self.doc_order.append(doc_id)
        for term, count in term_freq.items():
            self.inverted_index.setdefault(term, {})[doc_id] = count

        while len(self.doc_order) > self.max_docs:
            self._remove_document(self.doc_order[0])

        self._recompute_stats()
        if persist:
            self._save()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_terms = self.tokenize(query)
        if not query_terms or not self.documents:
            return []

        scores: Dict[str, float] = {}
        n_docs = len(self.documents)
        avgdl = self.avgdl or 1.0
        for term in set(query_terms):
            postings = self.inverted_index.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for doc_id, freq in postings.items():
                doc = self.documents.get(doc_id)
                if not doc:
                    continue
                dl = max(1, int(doc.get("length", 1)))
                denom = freq + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (freq * (self.k1 + 1.0)) / denom

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            {
                "doc_id": doc_id,
                "score": score,
                "text": self.documents[doc_id].get("text", ""),
                "metadata": self.documents[doc_id].get("metadata", {}),
            }
            for doc_id, score in ranked
        ]


class HybridMemoryDB:
    def __init__(
        self,
        vector_path: str = "data/vector_index.pkl",
        bm25_path: str = "data/hybrid_memory_bm25_index.json",
        dimension: int = 1024,
        reflection_backfill_limit: Optional[int] = None,
        vector_weight: float = 0.65,
        bm25_weight: float = 0.35,
    ):
        self.vector_index = VectorIndex(dimension=dimension, save_path=vector_path)
        self.bm25 = PersistentBM25Index(save_path=bm25_path)
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self._backfill_reflections(reflection_backfill_limit)

    @staticmethod
    def reflection_doc_id(index: int) -> str:
        return f"reflection:{index}"

    @staticmethod
    def reflection_text(reflection: Reflection) -> str:
        return "\n".join([
            "conversation reflection",
            f"input: {reflection.input_text}",
            f"reflection: {reflection.reflection_text}",
            f"response: {reflection.response_text}",
            f"model: {reflection.model_name}",
            f"keywords: {' '.join(reflection.keywords)}",
        ])

    def _backfill_reflections(self, limit: Optional[int] = None):
        total = len(self.vector_index.reflections)
        if total <= 0:
            return
        count = 0
        for index, reflection in enumerate(self.vector_index.reflections):
            if limit is not None and count >= limit:
                break
            doc_id = self.reflection_doc_id(index)
            if doc_id in self.bm25.documents:
                continue
            self.bm25.add_document(
                doc_id,
                self.reflection_text(reflection),
                metadata={
                    "kind": "reflection",
                    "reflection_index": index,
                    "timestamp": reflection.timestamp,
                    "model_name": reflection.model_name,
                    "notes": reflection.reflection_text[:240],
                    "updates": "",
                    "quality": "",
                },
                persist=False,
            )
            count += 1
        if count:
            self.bm25._recompute_stats()
            self.bm25._save()
            print(f"{Fore.CYAN}Backfilled {count} reflections into BM25 memory{Style.RESET_ALL}")

    def add_reflection(self, reflection: Reflection):
        self.vector_index.add_reflection(reflection)
        index = len(self.vector_index.reflections) - 1
        self.bm25.add_document(
            self.reflection_doc_id(index),
            self.reflection_text(reflection),
            metadata={
                "kind": "reflection",
                "reflection_index": index,
                "timestamp": reflection.timestamp,
                "model_name": reflection.model_name,
                "notes": reflection.reflection_text[:240],
                "updates": "",
                "quality": "",
            },
        )

    def search_reflection_matches(
        self,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 3,
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        vector_weight = self.vector_weight if vector_weight is None else vector_weight
        bm25_weight = self.bm25_weight if bm25_weight is None else bm25_weight
        combined: Dict[int, Dict[str, Any]] = {}

        if query_embedding and not all(float(x) == 0.0 for x in query_embedding):
            try:
                vector_hits = self.vector_index.search_with_scores(query_embedding, top_k=max(top_k * 3, top_k))
                for hit in vector_hits:
                    index = int(hit["index"])
                    score = max(0.0, min(1.0, (float(hit["score"]) + 1.0) / 2.0))
                    combined[index] = {
                        "reflection": hit["reflection"],
                        "score": vector_weight * score,
                        "vector_score": float(hit["score"]),
                        "bm25_score": 0.0,
                        "sources": {"vector"},
                    }
            except Exception as e:
                print(f"{Fore.YELLOW}Vector memory search failed: {e}{Style.RESET_ALL}")

        bm25_hits = self.bm25.search(query_text, top_k=max(top_k * 5, top_k))
        reflection_hits = [
            hit for hit in bm25_hits
            if hit.get("metadata", {}).get("kind") == "reflection"
            and hit.get("metadata", {}).get("reflection_index") is not None
        ]
        max_bm25 = max((float(hit.get("score", 0.0)) for hit in reflection_hits), default=0.0)
        for hit in reflection_hits:
            index = int(hit["metadata"]["reflection_index"])
            if index < 0 or index >= len(self.vector_index.reflections):
                continue
            norm = float(hit.get("score", 0.0)) / max_bm25 if max_bm25 > 0 else 0.0
            item = combined.setdefault(
                index,
                {
                    "reflection": self.vector_index.reflections[index],
                    "score": 0.0,
                    "vector_score": 0.0,
                    "bm25_score": 0.0,
                    "sources": set(),
                },
            )
            item["score"] += bm25_weight * norm
            item["bm25_score"] = float(hit.get("score", 0.0))
            item["sources"].add("bm25")

        ranked = sorted(combined.values(), key=lambda item: item["score"], reverse=True)[:top_k]
        for item in ranked:
            item["sources"] = sorted(item["sources"])
        return ranked

    def search_reflections(
        self,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 3,
    ) -> List[Reflection]:
        return [
            item["reflection"]
            for item in self.search_reflection_matches(query_text, query_embedding, top_k)
        ]


class OllamaChat:
    def __init__(
        self,
        max_history: int = 4,
        embed_model: str = "nomic-embed-text",
        history_path: str = "data/chat_history.pkl",
        host: Optional[str] = None,
        model_name: str = "gemma4:12b",
        memory_vector_path: str = "data/vector_index.pkl",
        memory_bm25_path: str = "data/hybrid_memory_bm25_index.json",
        memory_dimension: int = 1024,
        memory_reflection_backfill_limit: Optional[int] = None,
        memory_vector_weight: float = 0.65,
        memory_bm25_weight: float = 0.35,
    ):
        self.max_history = max_history
        self.model_name = model_name
        env_host = os.environ.get("OLLAMA_HOST")
        self.host = normalize_ollama_host(host)
        self.client = ollama.Client(host=self.host)
        if env_host and normalize_ollama_host(env_host) != env_host:
            print(
                f"{Fore.YELLOW}Using Ollama client host {self.host} "
                f"instead of OLLAMA_HOST={env_host!r}{Style.RESET_ALL}"
            )
        self.embed_model = embed_model
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

        config_path = Path("config/prompts.yaml")
        with open(config_path) as f:
            self.prompts = yaml.safe_load(f)

        if self.history_path.exists():
            with open(self.history_path, 'rb') as f:
                self.messages = pickle.load(f)
                print(f"{Fore.CYAN}Loaded {len(self.messages)-1} messages from history{Style.RESET_ALL}")
        else:
            self.messages = [{
                "role": "system",
                "content": self.prompts["ollama_system_prompt"]
            }]

        self.reflection_prompt = self.prompts["reflection_prompt"]
        self.response_prompt = self.prompts["response_prompt"]

        self.memory = HybridMemoryDB(
            vector_path=memory_vector_path,
            bm25_path=memory_bm25_path,
            dimension=memory_dimension,
            reflection_backfill_limit=memory_reflection_backfill_limit,
            vector_weight=memory_vector_weight,
            bm25_weight=memory_bm25_weight,
        )
        self.vector_index = self.memory.vector_index
        self.logger = Logger()

    def set_host(self, host: str):
        self.host = normalize_ollama_host(host)
        self.client = ollama.Client(host=self.host)

    @staticmethod
    def _message_text(message: Dict[str, str]) -> str:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        return f"{role}: {content}"

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

        if len(self.messages) > self.max_history + 1:
            self.messages = [self.messages[0]] + self.messages[-(self.max_history):]

        with open(self.history_path, 'wb') as f:
            pickle.dump(self.messages, f)

    def build_context_messages(self, bert_response: str, reflections: List[Reflection], conversation_history: str) -> List[Dict[str, str]]:
        reflection_context = "\n".join([
            f"Previous reflection: {r.reflection_text}"
            for r in reflections
        ])

        formatted_prompt = self.response_prompt.format(
            reflection_context=reflection_context,
            conversation_history=conversation_history,
            bert_response=bert_response
        )

        return [
            {"role": "system", "content": self.prompts["ollama_system_prompt"]},
            {"role": "user", "content": formatted_prompt}
        ]

    async def generate_reflection(self, bert_response: str) -> Reflection:
        query_embedding = await self.generate_embedding(bert_response)
        relevant_reflections = self.memory.search_reflections(bert_response, query_embedding)
        reflection_context = "\n".join([r.reflection_text for r in relevant_reflections])

        formatted_prompt = self.reflection_prompt.format(
            bert_response=bert_response,
            reflection_context=reflection_context
        )

        reflection_messages = [
            {"role": "system", "content": self.prompts["ollama_system_prompt"]},
            {"role": "user", "content": formatted_prompt}
        ]

        self.logger.log_context({
            "type": "reflection",
            "bert_response": bert_response,
            "relevant_reflections": [r.dict() for r in relevant_reflections],
            "formatted_prompt": formatted_prompt,
            "messages": reflection_messages
        })

        reflection = self.client.chat(
            model=self.model_name,
            messages=reflection_messages
        )

        reflection_text = reflection['message']['content']
        embedding = await self.generate_embedding(reflection_text)
        keywords = self.extract_keywords(reflection_text)

        return Reflection(
            input_text=bert_response,
            reflection_text=reflection_text,
            response_text="",
            model_name=self.model_name,
            embedding=embedding,
            keywords=keywords
        )

    async def generate_response(self, bert_response: str, relevant_reflections: List[Reflection]) -> str:
        reflection_context = "\n".join([f"Previous reflection: {r.reflection_text}"
                                        for r in relevant_reflections])

        conversation_history = "\n".join(self._message_text(m) for m in self.messages[-4:])

        formatted_prompt = self.response_prompt.format(
            reflection_context=reflection_context,
            conversation_history=conversation_history,
            bert_response=bert_response
        )

        response_messages = [
            {"role": "system", "content": self.prompts["ollama_system_prompt"]},
            {"role": "user", "content": formatted_prompt}
        ]

        self.logger.log_context({
            "type": "response",
            "bert_response": bert_response,
            "relevant_reflections": [r.dict() for r in relevant_reflections],
            "conversation_history": conversation_history,
            "formatted_prompt": formatted_prompt,
            "messages": response_messages
        })

        response = self.client.chat(
            model=self.model_name,
            messages=response_messages
        )

        response_text = truncate_context(response['message']['content'], 256)

        self.logger.log_context({
            "type": "final_response",
            "raw_response": response['message']['content'],
            "truncated_response": response_text
        })

        return response_text

    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        tokens = self.tokenizer.encode(text.lower())
        words = [self.tokenizer.decode([token]) for token in tokens]

        filtered_words = [word.strip() for word in words
                          if len(word.strip()) > 1 and not word.strip().isnumeric()]

        word_freq: Dict[str, int] = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1

        return sorted(word_freq, key=word_freq.get, reverse=True)[:top_k]

    async def generate_embedding(self, text: str) -> List[float]:
        response = None
        try:
            response = self.client.embed(
                model=self.embed_model,
                input=text
            )
            if hasattr(response, 'embedding'):
                return response.embedding
            elif hasattr(response, 'embeddings'):
                return response.embeddings[0]
            elif isinstance(response, dict):
                if 'embedding' in response:
                    return response['embedding']
                elif 'embeddings' in response:
                    return response['embeddings'][0]

            print(f"Unexpected response structure: {type(response)}")
            if hasattr(response, '__dict__'):
                print(f"Available attributes: {dir(response)}")
            return [0.0] * self.memory.vector_index.dimension

        except Exception as e:
            print(f"Embedding error: {str(e)}")
            if response is not None:
                print(f"Response type: {type(response)}")
            return [0.0] * self.memory.vector_index.dimension

    async def get_relevant_reflections(self, text: str) -> List[Reflection]:
        embedding = await self.generate_embedding(text)
        return self.memory.search_reflections(text, embedding)


class TeacherFeedback(BaseModel):
    base_temperature: Optional[float] = Field(default=None, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: Optional[int] = Field(default=None, ge=1, le=100, description="Number of candidate tokens")
    context_window: Optional[int] = Field(default=None, ge=16, le=512, description="Bidirectional memory attention window")
    tau_coherence: Optional[float] = Field(default=None, ge=0.5, le=3.0, description="Semantic coherence score temperature")
    p_nucleus: Optional[float] = Field(default=None, ge=0.5, le=1.0, description="Nucleus sampling threshold")
    repetition_penalty: Optional[float] = Field(default=None, ge=0.4, le=1.0, description="Penalty multiplier for repeated tokens")
    repetition_window: Optional[int] = Field(default=None, ge=8, le=256, description="Recent-token window for repetition penalty")
    coherence_anchor_alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Prompt-anchor strength for coherence scoring")
    use_denoise: Optional[bool] = Field(default=None, description="Enable post-draft denoise pass")
    denoise_passes: Optional[int] = Field(default=None, ge=0, le=4, description="Number of denoise passes")
    denoise_pattern: Optional[Literal["sequential", "random", "entropy", "span", "checkerboard"]] = Field(default=None, description="Denoise position order")
    denoise_top_k: Optional[int] = Field(default=None, ge=1, le=100, description="Candidate count for denoise replacements")
    denoise_accept_margin: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Required score improvement for denoise edits")
    denoise_temperature: Optional[float] = Field(default=None, ge=0.1, le=1.5, description="Denoise scoring temperature")
    notes: str = Field(default="", description="Explanation of adjustments")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in recommendations")

    class Config:
        json_schema_extra = {
            "example": {
                "base_temperature": 0.7,
                "top_k": 32,
                "context_window": 256,
                "tau_coherence": 1.7,
                "p_nucleus": 0.9,
                "repetition_penalty": 0.8,
                "coherence_anchor_alpha": 0.5,
                "notes": "Adjusted for better coherence",
                "confidence": 0.8
            }
        }


class ParameterUpdate(BaseModel):
    parameter: str
    old_value: Any
    new_value: Any
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class TeacherHistory(BaseModel):
    attempt_id: str
    suggestion: TeacherFeedback
    applied_updates: List[ParameterUpdate]
    configuration: Dict[str, Any]
    generation_quality: Dict[str, float] = Field(default_factory=dict)
    generated_text: str = ""
    context: str = ""
    reflection_text: str = ""
    outcome_text: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class TeacherAgent:
    def __init__(self,
                 ollama_client: Any,
                 model_name: str = "mistral:latest",
                 min_confidence: float = 0.6,
                 logger: Optional[Logger] = None,
                 history_size: int = 5,
                 bm25_path: str = "data/hybrid_memory_bm25_index.json",
                 related_attempts: int = 4,
                 bm25_index: Optional[PersistentBM25Index] = None):
        self.client = ollama_client
        self.model_name = model_name
        self.min_confidence = min_confidence
        self.logger = logger or Logger()
        self.history_size = history_size
        self.history: List[TeacherHistory] = []
        self.bm25 = bm25_index or PersistentBM25Index(save_path=bm25_path)
        self.related_attempts = related_attempts
        self.last_attempt_id: Optional[str] = None

        try:
            with open("config/prompts.yaml") as f:
                prompts = yaml.safe_load(f)
                self.evaluation_prompt = prompts.get("teacher_evaluation_prompt")
                if not self.evaluation_prompt:
                    raise ValueError("Teacher prompt not found in prompts.yaml")
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not load teacher prompt: {e}{Style.RESET_ALL}")
            self.evaluation_prompt = """
                Analyze this text and suggest parameter updates as JSON:
                Generated text: {generated_text}
                Context: {context}
                Return valid JSON with qBERT parameter updates plus notes and confidence.
            """

    def _make_attempt_id(self) -> str:
        return f"attempt-{datetime.now().strftime('%Y%m%d%H%M%S')}-{time.time_ns()}"

    @staticmethod
    def _quality_summary(metrics: Dict[str, float]) -> str:
        if not metrics:
            return "quality unavailable"
        return ", ".join(f"{k}: {v:.2f}" for k, v in metrics.items())

    @staticmethod
    def _updates_summary(updates: List[ParameterUpdate]) -> str:
        if not updates:
            return "No changes applied"
        return ", ".join(f"{u.parameter}: {u.old_value} -> {u.new_value}" for u in updates)

    @staticmethod
    def _feedback_patch(feedback: TeacherFeedback) -> Dict[str, Any]:
        return feedback.dict(exclude={"notes", "confidence"}, exclude_none=True)

    def _build_retrieval_query(
        self,
        generated_text: str,
        context: str,
        metrics: Dict[str, float],
        current_config: Dict[str, Any],
    ) -> str:
        tags = []
        punct = metrics.get("punct_ratio", 0.0)
        unique = metrics.get("unique_ratio", 1.0)
        length = metrics.get("length_ratio", 0.0)
        if punct > 0.18:
            tags.append("symbol operator punctuation salad")
        if unique < 0.45:
            tags.append("repetition loop low diversity")
        if length < 0.2:
            tags.append("short fragment underdeveloped")
        if length > 2.0:
            tags.append("overlong drifting verbose")
        return "\n".join([
            "current qbert state",
            context,
            generated_text,
            " ".join(tags),
            self._quality_summary(metrics),
            " ".join(f"{k}_{v}" for k, v in current_config.items()),
        ])

    def _format_related_attempts(self, related: List[Dict[str, Any]]) -> str:
        if not related:
            return "No retrieved similar attempts."
        lines = ["Retrieved similar records from persistent hybrid memory:"]
        for item in related:
            meta = item.get("metadata", {})
            lines.append(
                "\n".join([
                    f"- id: {item.get('doc_id')} score: {item.get('score', 0.0):.2f}",
                    f"  kind: {meta.get('kind', 'unknown')}",
                    f"  notes: {meta.get('notes', '')}",
                    f"  updates: {meta.get('updates', 'No changes recorded')}",
                    f"  quality: {meta.get('quality', '')}",
                    f"  excerpt: {item.get('text', '')[:500]}",
                ])
            )
        return "\n".join(lines)

    def _format_history_context(self, related: Optional[List[Dict[str, Any]]] = None) -> str:
        context: List[str] = []
        if not self.history:
            context.append("No in-process teacher adjustments yet.")
        else:
            context.append("Recent in-process adjustments and their effects:")

            for entry in self.history[-self.history_size:]:
                updates = self._updates_summary(entry.applied_updates)

                context.append(f"\nTimestamp: {entry.timestamp}")
                context.append(f"Attempt id: {entry.attempt_id}")
                context.append(f"Suggestion: {entry.suggestion.notes}")
                context.append(f"Changes: {updates}")
                if entry.generation_quality:
                    context.append(f"Quality Metrics: {self._quality_summary(entry.generation_quality)}")
                if entry.reflection_text:
                    context.append(f"Post-turn reflection: {entry.reflection_text}")

        context.append("")
        context.append(self._format_related_attempts(related or []))
        return "\n".join(context)

    def _index_attempt(self, entry: TeacherHistory):
        patch = self._feedback_patch(entry.suggestion)
        updates = self._updates_summary(entry.applied_updates)
        text = "\n".join([
            "teacher tuning attempt",
            f"attempt_id: {entry.attempt_id}",
            f"context: {entry.context}",
            f"qbert_output: {entry.generated_text}",
            f"quality: {self._quality_summary(entry.generation_quality)}",
            f"feedback_notes: {entry.suggestion.notes}",
            f"config_before: {' '.join(f'{k}_{v}' for k, v in entry.configuration.items())}",
            f"feedback_patch: {json.dumps(patch, sort_keys=True)}",
            f"applied_updates: {updates}",
            f"reflection: {entry.reflection_text}",
            f"outcome: {entry.outcome_text}",
        ])
        self.bm25.add_document(
            entry.attempt_id,
            text,
            metadata={
                "kind": "teacher_attempt",
                "timestamp": entry.timestamp,
                "notes": entry.suggestion.notes,
                "updates": updates,
                "quality": self._quality_summary(entry.generation_quality),
                "confidence": entry.suggestion.confidence,
            },
        )

    def record_post_turn_reflection(
        self,
        reflection: Reflection,
        outcome_text: str,
        attempt_id: Optional[str] = None,
    ):
        attempt_id = attempt_id or self.last_attempt_id
        if not attempt_id:
            return
        entry = next((h for h in reversed(self.history) if h.attempt_id == attempt_id), None)
        if not entry:
            return
        entry.reflection_text = reflection.reflection_text
        entry.outcome_text = outcome_text
        self._index_attempt(entry)

        reflection_text = "\n".join([
            "post change reflection",
            f"attempt_id: {attempt_id}",
            f"qbert_output: {entry.generated_text}",
            f"teacher_notes: {entry.suggestion.notes}",
            f"applied_updates: {self._updates_summary(entry.applied_updates)}",
            f"reflection: {reflection.reflection_text}",
            f"keywords: {' '.join(reflection.keywords)}",
            f"ollama_response: {outcome_text}",
        ])
        self.bm25.add_document(
            f"{attempt_id}:reflection",
            reflection_text,
            metadata={
                "kind": "post_reflection",
                "timestamp": datetime.now().isoformat(),
                "notes": entry.suggestion.notes,
                "updates": self._updates_summary(entry.applied_updates),
                "quality": self._quality_summary(entry.generation_quality),
            },
        )

    async def evaluate_and_update(self,
                                  generator: qBERTGenerator,
                                  bert_response: str,
                                  context: str,
                                  model_type: str) -> Tuple[TeacherFeedback, bool]:
        current_config = {
            param: getattr(generator.config, param)
            for param in TeacherFeedback.__fields__.keys()
            if hasattr(generator.config, param)
        }

        quality_metrics = self._calculate_quality_metrics(bert_response, context)
        retrieval_query = self._build_retrieval_query(bert_response, context, quality_metrics, current_config)
        related_attempts = self.bm25.search(retrieval_query, top_k=self.related_attempts)
        history_context = self._format_history_context(related_attempts)

        feedback = await self._get_feedback(
            generated_text=bert_response,
            context=context,
            current_config=current_config,
            history_context=history_context
        )

        self.logger.log_context({
            "type": "teacher_evaluation",
            "model_type": model_type,
            "bert_response": bert_response,
            "context": context,
            "current_config": current_config,
            "quality_metrics": quality_metrics,
            "related_attempts": related_attempts,
            "feedback": feedback.dict()
        })

        applied_updates: List[ParameterUpdate] = []
        updated = False

        if feedback.confidence >= self.min_confidence:
            updated = await self._update_parameters(
                generator, feedback, model_type, applied_updates
            )

        attempt_id = self._make_attempt_id()
        history_entry = TeacherHistory(
            attempt_id=attempt_id,
            suggestion=feedback,
            applied_updates=applied_updates,
            configuration=current_config,
            generation_quality=quality_metrics,
            generated_text=bert_response,
            context=context,
        )
        self.last_attempt_id = attempt_id
        self.history.append(history_entry)
        self._index_attempt(history_entry)

        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]

        return feedback, updated

    def _calculate_quality_metrics(self, text: str, context: str) -> Dict[str, float]:
        try:
            length_ratio = len(text.split()) / max(len(context.split()), 1)
            words = text.lower().split()
            unique_ratio = len(set(words)) / max(len(words), 1)
            punct_ratio = len([c for c in text if c in '.,!?;:']) / max(len(text), 1)

            return {
                "length_ratio": length_ratio,
                "unique_ratio": unique_ratio,
                "punct_ratio": punct_ratio
            }
        except Exception as e:
            print(f"{Fore.RED}Error calculating metrics: {e}{Style.RESET_ALL}")
            return {}

    async def _get_feedback(self,
                            generated_text: str,
                            context: str,
                            current_config: Dict[str, Any],
                            history_context: str) -> TeacherFeedback:
        prompt = self.evaluation_prompt.format(
            generated_text=generated_text,
            context=context,
            current_config=json.dumps(current_config, indent=2),
            history_context=history_context
        )

        try:
            self.logger.log_context({
                "type": "teacher_prompt",
                "prompt": prompt,
                "current_config": current_config,
                "history_context": history_context
            })

            response = self.client.chat(
                model=self.model_name,
                messages=[{
                    "role": "system",
                    "content": "You are a teaching assistant. Respond only with valid JSON."
                }, {
                    "role": "user",
                    "content": prompt
                }]
            )

            content = response['message']['content']
            json_str = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_str:
                raise ValueError("No JSON found in response")

            feedback_dict = json.loads(json_str.group())
            feedback = TeacherFeedback(**feedback_dict)

            self.logger.log_context({
                "type": "teacher_response",
                "raw_response": content,
                "feedback": feedback.dict(),
                "model_name": self.model_name
            })

            return feedback

        except Exception as e:
            print(f"{Fore.RED}Teacher evaluation failed: {str(e)}{Style.RESET_ALL}")
            return TeacherFeedback(
                notes="Failed to get feedback, using defaults",
                confidence=0.0
            )

    async def _update_parameters(self,
                                 generator: qBERTGenerator,
                                 feedback: TeacherFeedback,
                                 model_type: str,
                                 applied_updates: List[ParameterUpdate]) -> bool:
        try:
            updates = feedback.dict(exclude={'notes', 'confidence'}, exclude_none=True)
            for param, value in updates.items():
                if not hasattr(generator.config, param):
                    continue
                try:
                    old_value = getattr(generator.config, param)
                    current_type = type(old_value)
                    if current_type == bool:
                        new_value = value if isinstance(value, bool) else str(value).lower() == 'true'
                    else:
                        new_value = current_type(value)
                    setattr(generator.config, param, new_value)

                    applied_updates.append(ParameterUpdate(
                        parameter=param,
                        old_value=old_value,
                        new_value=new_value
                    ))

                    self.logger.log_system_update(SystemUpdate(
                        update_type=param,
                        previous_value=old_value,
                        new_value=new_value,
                        model_type=model_type
                    ))

                    print(f"{Fore.YELLOW}Updated {param}: {old_value} → {new_value}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}Failed to update {param}: {e}{Style.RESET_ALL}")
            return True

        except Exception as e:
            print(f"{Fore.RED}Parameter update failed: {str(e)}{Style.RESET_ALL}")
            return False


def create_generator(config: Optional[qBERTConfig] = None,
                     model_config: Optional[ModelConfig] = None,
                     config_data: Optional[Dict[str, Any]] = None) -> qBERTGenerator:
    config_data = config_data or load_model_config()
    if config is None:
        config = create_qbert_config(config_data)
    return qBERTGenerator(config, model_config=model_config or create_model_config(config_data))


def update_config(generator: qBERTGenerator, param: str, value: str) -> qBERTGenerator:
    current_config = generator.config
    config_dict = vars(current_config)

    try:
        current_type = type(config_dict[param])
        if current_type == bool:
            value_cast = value.lower() == 'true'
        else:
            value_cast = current_type(value)

        new_config_dict = vars(current_config).copy()
        new_config_dict[param] = value_cast

        new_config = qBERTConfig(**new_config_dict)
        return create_generator(config=new_config, model_config=generator.mc)

    except ValueError:
        raise ValueError(f"Invalid value for {param}. Expected type: {current_type.__name__}")
    except KeyError:
        raise KeyError(f"Unknown parameter: {param}")


def create_ollama_client(model_name: str = "deepseek-r1:8b"):
    return {"model": model_name, "client": ollama}


def truncate_context(text: str, max_tokens: int) -> str:
    response_match = re.search(r'(?<=<response>).*?(?=</response>)', text, re.DOTALL)
    if response_match:
        text = response_match.group().strip()
    else:
        paragraphs = text.split('\n\n')
        text = paragraphs[-1].strip() if paragraphs else text

    sentences = re.split(r'(?<=[.!?])\s+', text)
    tokens = 0
    result: List[str] = []

    for sentence in reversed(sentences):
        sentence_tokens = len(re.findall(r'\w+|[^\w\s]', sentence))
        if tokens + sentence_tokens > max_tokens:
            break
        result.insert(0, sentence)
        tokens += sentence_tokens

    return ' '.join(result)


def process_ollama_output(text: str, max_tokens: int = 256) -> str:
    paragraphs = text.split('\n\n')
    last_para = paragraphs[-1]

    answer_match = re.search(r'<response>(.*?)</response>', last_para, re.DOTALL)
    if answer_match:
        text = answer_match.group(1).strip()
    else:
        all_sentences: List[str] = []
        for para in reversed(paragraphs):
            para_sentences = [s.strip() for s in re.split(r'([.!?])\s+', para) if s.strip()]
            for i in range(0, len(para_sentences) - 1, 2):
                if i + 1 < len(para_sentences):
                    all_sentences.append(para_sentences[i] + para_sentences[i + 1])
            if len(all_sentences) >= 4:
                break

        text = ' '.join(all_sentences[-4:]).strip()

    tokens = re.findall(r'\w+|[^\w\s]', text)
    if len(tokens) > max_tokens:
        keep_tokens = max_tokens // 2
        start_text = ' '.join(tokens[:keep_tokens])
        end_text = ' '.join(tokens[-keep_tokens:])
        text = f"{start_text} ... {end_text}"

    return text


def format_thinking(text: str) -> str:
    match = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL)
    return match.group(1).strip() if match else ""


def print_auto_help(generator: qBERTGenerator, model_type: str, ollama_chat: OllamaChat):
    print("\nCurrent Configuration:")
    print(f"BERT Model Type: {model_type}")
    print(f"Ollama Model: {ollama_chat.model_name}")
    print(f"Ollama Host: {ollama_chat.host}")
    for key, value in vars(generator.config).items():
        print(f"{key}: {value}")
    print("\nCommands:")
    print("/help - Show this help and current configuration")
    print("/models - List available Ollama models")
    print("/ollama <model> - Switch Ollama model")
    print("/ollama_host <host> - Switch Ollama host, e.g. http://127.0.0.1:11434")
    print("/device <cpu|cuda> - Switch device")
    print("/tokens <number> - Set token count")
    print("/stream - Toggle streaming mode")
    print("/denoise - Toggle post-generation denoise pass")
    print("/denoise_passes <number> - Set denoise passes")
    print("/clear - Clear screen")
    print("/quit - Exit program")
    print("\nModel Configuration Commands:")
    print("/bert_model <name> - Change BERT model, e.g. bert-base-uncased")
    print("/sentence_model <name> - Change sentence transformer model")
    print("Or use /<param_name> <value> for any GenerationConfig field")


def print_ollama_models(ollama_chat: OllamaChat):
    try:
        models = ollama_chat.client.list()
        print("\nAvailable Ollama Models:")
        model_list = getattr(models, "models", None)
        if not model_list:
            print("No models found")
            return
        for model in model_list:
            print(f"- {getattr(model, 'model', None) or getattr(model, 'name', None) or str(model)}")
    except Exception as e:
        print(f"Error listing models from {ollama_chat.host}: {str(e)}")
        print("Make sure the Ollama server is running and the host is reachable")


async def process_exchange(bert_response: str, ollama_chat: OllamaChat):
    reflection = await ollama_chat.generate_reflection(bert_response)
    query_embedding = reflection.embedding
    relevant_reflections = ollama_chat.memory.search_reflections(reflection.reflection_text, query_embedding)
    response = await ollama_chat.generate_response(bert_response, relevant_reflections)
    reflection.response_text = response
    ollama_chat.memory.add_reflection(reflection)
    return response


async def main():
    config_data = load_model_config()
    session_defaults = autochat_session_defaults(config_data)
    num_tokens = session_defaults["num_tokens"]
    stream_mode = session_defaults["stream_mode"]
    model_type = session_defaults["model_type"]

    logger = Logger()
    generator = create_generator(config_data=config_data)
    auto_chat = create_auto_chat(config_data)
    ollama_chat = OllamaChat(**ollama_kwargs_from_config(config_data))
    teacher_kwargs, teacher_uses_ollama_model = teacher_kwargs_from_config(
        config_data,
        ollama_chat.model_name,
    )
    teacher = TeacherAgent(
        ollama_client=ollama_chat.client,
        logger=logger,
        bm25_index=ollama_chat.memory.bm25,
        **teacher_kwargs,
    )

    init(autoreset=True)

    banner = f"""
    {Fore.CYAN}╔═══════════════════════════════════════╗
    ║         {Style.BRIGHT}Autonomous BERT Chat{Style.NORMAL}          ║
    ╚═══════════════════════════════════════╝{Style.RESET_ALL}
    """
    print(banner)

    while True:
        seed_input = input(f"{Fore.GREEN}> {Style.RESET_ALL}").strip()
        if not seed_input:
            continue
        if seed_input.lower() == '/help':
            print_auto_help(generator, model_type, ollama_chat)
            continue
        if seed_input.lower() == '/models':
            print_ollama_models(ollama_chat)
            continue
        if seed_input.lower() == '/quit':
            print(f"{Fore.RED}Goodbye!{Style.RESET_ALL}")
            return
        if seed_input.startswith('/'):
            print("Enter a seed message, or use /help, /models, or /quit.")
            continue
        break
    auto_chat.conversation_history.append(seed_input)

    print(f"\n{Fore.YELLOW}Press Ctrl+C to pause/resume, Ctrl+D to quit{Style.RESET_ALL}")

    while True:
        try:
            if not auto_chat.is_paused:
                bert_input = truncate_context(auto_chat.conversation_history[-1], auto_chat.max_bert_context)
                print(f"{Fore.BLUE}{model_type.upper()}: ", end="", flush=True)
                bert_response = ""
                start_time = time.time()

                denoise_started = False
                denoise_edits = 0
                denoise_preview = ""
                denoise_preview_lines = 0
                for event in generator.generate_with_denoise_stream(
                    f"{bert_input}\n",
                    num_tokens=min(num_tokens, auto_chat.max_bert_response)
                ):
                    phase = event.get("phase")
                    if phase == "draft":
                        print(f"{event['token']}", end="", flush=True)
                        bert_response = event["text"]
                    elif phase == "denoise_start":
                        denoise_started = True
                        denoise_preview_lines = redraw_denoise_preview(
                            denoise_preview_lines,
                            bert_response,
                            "starting from streamed draft",
                        )
                    elif phase == "denoise":
                        denoise_edits += 1
                        denoise_preview = event["text"]
                        denoise_preview_lines = redraw_denoise_preview(
                            denoise_preview_lines,
                            denoise_preview,
                            (
                                f"pass {event['pass']} edit {denoise_edits}: "
                                f"{event.get('old_token', '')!r} -> {event.get('new_token', '')!r}"
                            ),
                        )
                    elif phase == "denoise_complete":
                        denoise_preview = event["text"]
                        if denoise_started and (denoise_edits or event.get("edits", 0)):
                            denoise_preview_lines = redraw_denoise_preview(
                                denoise_preview_lines,
                                denoise_preview,
                                f"complete: {event.get('edits', denoise_edits)} edits accepted",
                            )
                        elif denoise_started:
                            denoise_preview_lines = redraw_denoise_preview(
                                denoise_preview_lines,
                                bert_response,
                                "complete: no edits accepted",
                            )

                generation_time = time.time() - start_time

                logger.log_response(ModelResponse(
                    model_name=model_type,
                    model_type='bert',
                    response=bert_response,
                    input_text=bert_input,
                    config=vars(generator.config),
                    tokens=len(re.findall(r'\w+', bert_response)),
                    generation_time=generation_time
                ))

                print(Style.RESET_ALL)

                feedback, updated = await teacher.evaluate_and_update(
                    generator=generator,
                    bert_response=bert_response,
                    context="\n".join(auto_chat.conversation_history[-4:]),
                    model_type=model_type
                )

                if updated:
                    print(f"\n{Fore.CYAN}Teacher Notes: {feedback.notes}{Style.RESET_ALL}")

                reflection = await ollama_chat.generate_reflection(bert_response)

                similar_reflections = ollama_chat.memory.search_reflections(
                    f"{bert_response}\n{reflection.reflection_text}",
                    reflection.embedding,
                )
                print(f"Found {len(similar_reflections)} relevant reflections")

                messages = ollama_chat.build_context_messages(
                    bert_response=bert_response,
                    reflections=similar_reflections,
                    conversation_history="\n".join(auto_chat.conversation_history[-4:])
                )

                print(f"{Fore.MAGENTA}{ollama_chat.model_name}: ", end="", flush=True)
                raw_response = ""
                ollama_start = time.time()

                for chunk in ollama_chat.client.chat(
                    model=ollama_chat.model_name,
                    messages=messages,
                    stream=True
                ):
                    if chunk['message']['content']:
                        print(f"{chunk['message']['content']}", end="", flush=True)
                        raw_response += chunk['message']['content']
                print(Style.RESET_ALL)

                clean_response = truncate_context(raw_response, 256)

                reflection.response_text = raw_response
                ollama_chat.memory.add_reflection(reflection)
                teacher.record_post_turn_reflection(reflection, clean_response)
                ollama_chat.add_message("user", bert_response)
                ollama_chat.add_message("assistant", clean_response)
                auto_chat.conversation_history.extend([bert_response, clean_response])

                logger.log_response(ModelResponse(
                    model_name=ollama_chat.model_name,
                    model_type="ollama",
                    response=clean_response,
                    input_text=messages[-1]["content"],
                    full_response=raw_response,
                    system_prompt=messages[0]["content"],
                    input_prompt=messages[-1]["content"],
                    reflection_context="\n".join(f"Previous reflection: {r.reflection_text}"
                                                 for r in similar_reflections),
                    config={"model": ollama_chat.model_name},
                    tokens=len(re.findall(r'\w+', clean_response)),
                    generation_time=time.time() - ollama_start
                ))

                if len(auto_chat.conversation_history) > auto_chat.max_context_length:
                    auto_chat.conversation_history = auto_chat.conversation_history[-auto_chat.max_context_length:]

                time.sleep(auto_chat.pause_duration)

            else:
                user_input = input(f"\n{Fore.YELLOW}Chat paused. Enter command or press Enter to resume: {Style.RESET_ALL}")
                if not user_input:
                    auto_chat.is_paused = False
                    print("Resuming chat...")
                    continue

                if user_input.lower() == '/quit':
                    print(f"{Fore.RED}Goodbye!{Style.RESET_ALL}")
                    break

                if user_input.startswith('/'):
                    parts = user_input[1:].split(maxsplit=1)
                    command = parts[0].lower()

                    if hasattr(generator.config, command):
                        try:
                            if len(parts) != 2:
                                print(f"Please provide a value for {command}")
                                continue
                            old_config = vars(generator.config).copy()
                            generator = update_config(generator, command, parts[1])
                            logger.log_system_update(SystemUpdate(
                                update_type="config",
                                previous_value=old_config[command],
                                new_value=parts[1],
                                model_type=model_type
                            ))
                            print(f"Updated {command} to {parts[1]}")
                            continue
                        except (ValueError, KeyError) as e:
                            print(f"Error: {str(e)}")
                            continue

                if user_input.lower() == '/help':
                    print_auto_help(generator, model_type, ollama_chat)
                    continue

                if user_input.lower() == '/models':
                    print_ollama_models(ollama_chat)
                    continue

                if user_input.lower().startswith('/ollama '):
                    try:
                        _, new_model = user_input.split()
                        old_model = ollama_chat.model_name
                        ollama_chat.model_name = new_model
                        if teacher_uses_ollama_model:
                            teacher.model_name = new_model
                        logger.log_system_update(SystemUpdate(
                            update_type="ollama_model",
                            previous_value=old_model,
                            new_value=new_model,
                            model_type="ollama"
                        ))
                        print(f"Switched Ollama model to {new_model}")
                    except Exception as e:
                        print(f"Error switching Ollama model: {str(e)}")
                    continue

                if user_input.lower().startswith('/ollama_host '):
                    try:
                        _, new_host = user_input.split(maxsplit=1)
                        old_host = ollama_chat.host
                        ollama_chat.set_host(new_host)
                        teacher.client = ollama_chat.client
                        logger.log_system_update(SystemUpdate(
                            update_type="ollama_host",
                            previous_value=old_host,
                            new_value=ollama_chat.host,
                            model_type="ollama"
                        ))
                        print(f"Switched Ollama host to {ollama_chat.host}")
                    except Exception as e:
                        print(f"Error switching Ollama host: {str(e)}")
                    continue

                if user_input.lower().startswith('/device '):
                    try:
                        _, device = user_input.split()
                        if device not in ['cpu', 'cuda']:
                            raise ValueError("Device must be 'cpu' or 'cuda'")
                        if device == 'cuda' and not torch.cuda.is_available():
                            print("CUDA is not available on this system")
                            continue
                        old_device = generator.config.device
                        generator = update_config(generator, 'device', device)
                        logger.log_system_update(SystemUpdate(
                            update_type="device",
                            previous_value=old_device,
                            new_value=device,
                            model_type=model_type
                        ))
                        print(f"Switched to {device.upper()}")
                    except Exception as e:
                        print(f"Error switching device: {str(e)}")
                    continue

                if user_input.lower().startswith('/tokens '):
                    try:
                        _, value = user_input.split()
                        old_tokens = num_tokens
                        num_tokens = int(value)
                        auto_chat.max_bert_response = num_tokens
                        logger.log_system_update(SystemUpdate(
                            update_type="tokens",
                            previous_value=old_tokens,
                            new_value=num_tokens,
                            model_type=model_type
                        ))
                        print(f"Will generate {num_tokens} tokens")
                    except ValueError:
                        print("Please provide a valid number")
                    continue

                if user_input.lower() == '/stream':
                    old_stream = stream_mode
                    stream_mode = not stream_mode
                    logger.log_system_update(SystemUpdate(
                        update_type="stream_mode",
                        previous_value=old_stream,
                        new_value=stream_mode,
                        model_type=model_type
                    ))
                    print(f"Streaming mode: {'ON' if stream_mode else 'OFF'}")
                    continue

                if user_input.lower() == '/denoise':
                    old_denoise = generator.config.use_denoise
                    generator.config.use_denoise = not old_denoise
                    logger.log_system_update(SystemUpdate(
                        update_type="use_denoise",
                        previous_value=old_denoise,
                        new_value=generator.config.use_denoise,
                        model_type=model_type
                    ))
                    print(f"Denoise: {'ON' if generator.config.use_denoise else 'OFF'}")
                    continue

                if user_input.lower().startswith('/denoise_passes '):
                    try:
                        _, value = user_input.split()
                        old_passes = generator.config.denoise_passes
                        generator.config.denoise_passes = max(0, int(value))
                        logger.log_system_update(SystemUpdate(
                            update_type="denoise_passes",
                            previous_value=old_passes,
                            new_value=generator.config.denoise_passes,
                            model_type=model_type
                        ))
                        print(f"Denoise passes: {generator.config.denoise_passes}")
                    except ValueError:
                        print("Please provide a valid number")
                    continue

                if user_input.lower().startswith('/bert_model '):
                    try:
                        _, model_name = user_input.split(maxsplit=1)
                        model_name = model_name.strip('"')
                        old_model = generator.mc.bert_model_name
                        model_config = ModelConfig(
                            bert_model_name=model_name,
                            tokenizer_name=model_name,
                            sentence_transformer_name=generator.mc.sentence_transformer_name,
                            attn_implementation=generator.mc.attn_implementation
                        )
                        generator = create_generator(
                            config=generator.config,
                            model_config=model_config
                        )
                        logger.log_system_update(SystemUpdate(
                            update_type="bert_model",
                            previous_value=old_model,
                            new_value=model_name,
                            model_type=model_type
                        ))
                        print(f"Switched BERT model to {model_name}")
                    except Exception as e:
                        print(f"Error switching BERT model: {str(e)}")
                    continue

                if user_input.lower().startswith('/sentence_model '):
                    try:
                        _, model_name = user_input.split()
                        old_model = generator.mc.sentence_transformer_name
                        model_config = ModelConfig(
                            bert_model_name=generator.mc.bert_model_name,
                            tokenizer_name=generator.mc.tokenizer_name,
                            sentence_transformer_name=model_name,
                            attn_implementation=generator.mc.attn_implementation
                        )
                        generator = create_generator(
                            config=generator.config,
                            model_config=model_config
                        )
                        logger.log_system_update(SystemUpdate(
                            update_type="sentence_model",
                            previous_value=old_model,
                            new_value=model_name,
                            model_type=model_type
                        ))
                        print(f"Switched sentence transformer to {model_name}")
                    except Exception as e:
                        print(f"Error switching sentence transformer: {str(e)}")
                    continue

        except KeyboardInterrupt:
            auto_chat.is_paused = not auto_chat.is_paused
            print(f"\n{Fore.YELLOW}{'Chat paused. Press Ctrl+C again to resume.' if auto_chat.is_paused else 'Resuming chat...'}{Style.RESET_ALL}")

        except EOFError:
            print(f"\n{Fore.RED}Goodbye!{Style.RESET_ALL}")
            break

        except Exception as e:
            print(f"\n{Fore.RED}Unexpected error: {str(e)}{Style.RESET_ALL}")
            auto_chat.is_paused = True


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)
