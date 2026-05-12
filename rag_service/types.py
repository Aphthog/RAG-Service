from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class Chunk:
    text: str
    metadata: dict
    chunk_index: int
    start_char: int
    end_char: int


@dataclass
class SearchResult:
    content: str
    metadata: dict
    score: float
    chunk_index: int


@dataclass
class Embeddings:
    dense: np.ndarray
    sparse: csr_matrix


@dataclass
class ChunkConfig:
    min_chars: int = 200
    max_chars: int = 800
    overlap: int = 80
    semantic_threshold: float = 0.6
    batch_size: int = 32
