"""
Central configuration for the OrbitDesk Support Agent.

Keeping every tunable value in one place makes the retry/threshold behaviour
easy to explain and easy to change without hunting through the node code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
KB_DIR = DATA_DIR / "knowledge_base"
RESOLVED_CASES_PATH = DATA_DIR / "resolved_cases.json"
INDEX_CACHE_DIR = PROJECT_ROOT / ".cache" / "index"
QUERY_CACHE_PATH = PROJECT_ROOT / ".cache" / "query_cache.json"


@dataclass
class ModelConfig:
    """Names/revisions are pinned so the README claim is actually reproducible."""

    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_model_revision: str = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

    # Primary generator. Falls back automatically on low-RAM / no-GPU machines.
    generation_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    generation_model_revision: str = "main"

    fallback_generation_models: tuple[str, ...] = (
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/Phi-3-mini-4k-instruct",
    )
    max_new_tokens: int = 350
    generation_temperature: float = 0.1


@dataclass
class RetrievalConfig:
    chunk_size_chars: int = 900
    chunk_overlap_chars: int = 150
    top_k_vector: int = 6
    top_k_keyword: int = 6
    top_k_final: int = 4
    vector_weight: float = 0.65
    keyword_weight: float = 0.35
    low_confidence_threshold: float = 0.34  # below this -> ask for clarification


@dataclass
class VerificationConfig:
    max_retries: int = 1
    min_answer_confidence: float = 0.45
    min_evidence_overlap: float = 0.30  # fraction of answer sentences grounded in sources (lexical fallback)
    min_semantic_similarity: float = 0.55  # cosine similarity threshold for the embedding-based guard


MODEL_CONFIG = ModelConfig()
RETRIEVAL_CONFIG = RetrievalConfig()
VERIFICATION_CONFIG = VerificationConfig()
