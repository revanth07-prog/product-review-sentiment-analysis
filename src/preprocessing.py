"""
Text Preprocessing and Tokenization Pipeline.
Provides identical cleaning, tokenization, sequence padding, and label encoding
for both training pipelines and live inference.
"""

import os
import re
import html
import json
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# Common English contractions dictionary
CONTRACTIONS = {
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
}


def clean_text(text: Union[str, float, None]) -> str:
    """
    Standard text normalization pipeline for product reviews.
    1. Handles None / non-string inputs.
    2. Unescapes HTML entities.
    3. Strips HTML tags.
    4. Converts to lowercase.
    5. Expands common contractions (preserving negation context like 'not').
    6. Removes URLs and email addresses.
    7. Cleans non-alphanumeric noise while retaining word boundaries and punctuation spacing.
    8. Collapses multiple whitespace.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Decode HTML entities (e.g., &amp;, &quot;)
    text = html.unescape(text)

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Lowercase
    text = text.lower()

    # Expand contractions
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Replace special product-review artifacts (like question mark encoding garbage ??)
    text = re.sub(r"\?\?", " ", text)

    # Keep only letters, digits, and basic whitespace
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


class TextPreprocessor:
    """
    Manages tokenizer fitting, sequence padding, and label encoding.
    Ensures zero data leakage and consistent artifact persistence.
    """

    def __init__(
        self,
        max_vocab_size: int = 20000,
        max_sequence_length: int = 100,
        oov_token: str = "<OOV>",
        padding: str = "post",
        truncating: str = "post"
    ):
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.oov_token = oov_token
        self.padding = padding
        self.truncating = truncating

        self.tokenizer: Optional[Tokenizer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.label_mapping: Dict[str, int] = {}
        self.classes_: List[str] = []

    def fit(self, texts: List[str], labels: Optional[List[str]] = None) -> "TextPreprocessor":
        """
        Fit tokenizer on training texts and label encoder on training labels.
        """
        cleaned_texts = [clean_text(t) for t in texts]

        # Fit Keras Tokenizer
        self.tokenizer = Tokenizer(
            num_words=self.max_vocab_size,
            oov_token=self.oov_token,
            lower=True
        )
        self.tokenizer.fit_on_texts(cleaned_texts)

        # Fit LabelEncoder if labels provided
        if labels is not None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(labels)
            self.classes_ = self.label_encoder.classes_.tolist()
            self.label_mapping = {cls_name: idx for idx, cls_name in enumerate(self.classes_)}
            logger.info(f"Fitted label encoder with classes: {self.label_mapping}")

        vocab_count = min(len(self.tokenizer.word_index) + 1, self.max_vocab_size)
        logger.info(f"Fitted tokenizer on {len(texts)} samples. Effective vocab size: {vocab_count}")
        return self

    def transform_texts(self, texts: List[str]) -> np.ndarray:
        """
        Transform raw review texts into padded integer sequences.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been fitted or loaded yet.")

        cleaned = [clean_text(t) for t in texts]
        sequences = self.tokenizer.texts_to_sequences(cleaned)
        padded = pad_sequences(
            sequences,
            maxlen=self.max_sequence_length,
            padding=self.padding,
            truncating=self.truncating
        )
        return padded

    def transform_labels(self, labels: List[str]) -> np.ndarray:
        """
        Encode target string labels to integers.
        """
        if self.label_encoder is None:
            raise ValueError("Label encoder has not been fitted or loaded yet.")
        return self.label_encoder.transform(labels)

    def inverse_transform_labels(self, encoded_labels: Union[np.ndarray, List[int]]) -> List[str]:
        """
        Convert encoded label indices back to original string class names.
        """
        if self.label_encoder is None:
            raise ValueError("Label encoder has not been fitted or loaded yet.")
        return self.label_encoder.inverse_transform(encoded_labels).tolist()

    def save_artifacts(self, artifacts_dir: str = "artifacts") -> None:
        """
        Persist tokenizer, label encoder, and preprocessing metadata to disk.
        """
        os.makedirs(artifacts_dir, exist_ok=True)

        tokenizer_path = os.path.join(artifacts_dir, "tokenizer.pkl")
        label_encoder_path = os.path.join(artifacts_dir, "label_encoder.pkl")
        meta_path = os.path.join(artifacts_dir, "preprocessing_meta.json")

        with open(tokenizer_path, "wb") as f:
            pickle.dump(self.tokenizer, f)

        with open(label_encoder_path, "wb") as f:
            pickle.dump(self.label_encoder, f)

        meta = {
            "max_vocab_size": self.max_vocab_size,
            "max_sequence_length": self.max_sequence_length,
            "oov_token": self.oov_token,
            "padding": self.padding,
            "truncating": self.truncating,
            "classes": self.classes_,
            "label_mapping": self.label_mapping,
            "num_classes": len(self.classes_),
            "total_words_indexed": len(self.tokenizer.word_index) if self.tokenizer else 0
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved preprocessing artifacts to '{artifacts_dir}'")

    @classmethod
    def load_artifacts(cls, artifacts_dir: str = "artifacts") -> "TextPreprocessor":
        """
        Load persisted tokenizer, label encoder, and metadata.
        """
        tokenizer_path = os.path.join(artifacts_dir, "tokenizer.pkl")
        label_encoder_path = os.path.join(artifacts_dir, "label_encoder.pkl")
        meta_path = os.path.join(artifacts_dir, "preprocessing_meta.json")

        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Missing tokenizer artifact at: {tokenizer_path}")
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"Missing label encoder artifact at: {label_encoder_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing preprocessing metadata at: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(
            max_vocab_size=meta.get("max_vocab_size", 20000),
            max_sequence_length=meta.get("max_sequence_length", 100),
            oov_token=meta.get("oov_token", "<OOV>"),
            padding=meta.get("padding", "post"),
            truncating=meta.get("truncating", "post")
        )

        with open(tokenizer_path, "rb") as f:
            instance.tokenizer = pickle.load(f)

        with open(label_encoder_path, "rb") as f:
            instance.label_encoder = pickle.load(f)

        instance.classes_ = meta.get("classes", [])
        instance.label_mapping = meta.get("label_mapping", {})

        return instance
