"""
Pre-trained GloVe Embeddings Integration Module.
Loads Stanford GloVe text vectors, builds vocabulary-aligned embedding matrices,
handles out-of-vocabulary words, and reports coverage statistics.
"""

import os
import logging
from typing import Dict, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DOWNLOAD_INSTRUCTIONS = """
================================================================================
                    GLOVE EMBEDDINGS SETUP INSTRUCTIONS
================================================================================
To obtain the official pre-trained Stanford GloVe embeddings:

1. Automatic Setup:
   Run the project download script:
   python -m src.download_glove 100

2. Manual Download:
   - Download 'glove.6B.zip' from Stanford NLP:
     https://nlp.stanford.edu/data/glove.6B.zip
     or Hugging Face mirror:
     https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip
   - Extract 'glove.6B.100d.txt' (or 50d/200d/300d) into the 'embeddings/' folder.
   - Target location: 'embeddings/glove.6B.100d.txt'
================================================================================
"""


def load_glove_vectors(
    glove_path: str = "embeddings/glove.6B.100d.txt",
    expected_dim: Optional[int] = 100
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Parse the pre-trained GloVe vector text file into a dictionary of word -> float numpy array.
    Validates file integrity and vector dimensionality.
    """
    if not os.path.exists(glove_path):
        raise FileNotFoundError(
            f"GloVe embeddings file not found at: '{glove_path}'\n{DOWNLOAD_INSTRUCTIONS}"
        )

    file_size_mb = os.path.getsize(glove_path) / (1024 * 1024)
    if file_size_mb < 1.0:
        raise ValueError(
            f"GloVe file at '{glove_path}' appears corrupt or truncated ({file_size_mb:.2f} MB)."
        )

    logger.info(f"Loading GloVe vectors from '{glove_path}' ({file_size_mb:.1f} MB)...")
    embeddings_index: Dict[str, np.ndarray] = {}
    detected_dim: Optional[int] = None

    with open(glove_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f, start=1):
            values = line.strip().split(" ")
            if len(values) < 2:
                continue

            word = values[0]
            try:
                vector = np.asarray(values[1:], dtype=np.float32)
            except ValueError:
                continue

            if detected_dim is None:
                detected_dim = len(vector)
                logger.info(f"Detected GloVe vector dimension: {detected_dim}")
                if expected_dim is not None and detected_dim != expected_dim:
                    logger.warning(
                        f"Expected dimension {expected_dim}, but detected {detected_dim}. "
                        f"Updating embedding dimension to {detected_dim}."
                    )

            if len(vector) == detected_dim:
                embeddings_index[word] = vector

    logger.info(f"Loaded {len(embeddings_index):,} word vectors from GloVe.")
    return embeddings_index, detected_dim


def build_embedding_matrix(
    word_index: Dict[str, int],
    embeddings_index: Dict[str, np.ndarray],
    vocab_size: int,
    embedding_dim: int
) -> Tuple[np.ndarray, Dict]:
    """
    Construct embedding matrix aligned with tokenizer vocabulary indices.
    Index 0 is reserved for padding and remains zero-vector.
    Words missing from GloVe are left as zeros (or random uniform if specified).
    """
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    matched_words = 0
    missing_words = []

    for word, idx in word_index.items():
        if idx >= vocab_size:
            continue

        vector = embeddings_index.get(word)
        if vector is not None:
            embedding_matrix[idx] = vector
            matched_words += 1
        else:
            missing_words.append(word)

    coverage_pct = (matched_words / (vocab_size - 1)) * 100 if vocab_size > 1 else 0.0

    stats = {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "matched_words": matched_words,
        "out_of_vocab_count": len(missing_words),
        "coverage_percentage": round(coverage_pct, 2),
        "sample_missing_words": missing_words[:10]
    }

    logger.info(
        f"GloVe Embedding Matrix built: shape={embedding_matrix.shape}. "
        f"Matched {matched_words:,}/{vocab_size - 1:,} words ({coverage_pct:.2f}% coverage)."
    )
    return embedding_matrix, stats
