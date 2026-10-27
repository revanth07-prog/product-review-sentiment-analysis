"""
Unit and Integration Tests for Text Preprocessing and Tokenization.
"""

import os
import shutil
import tempfile
import pytest
import numpy as np

from src.preprocessing import clean_text, TextPreprocessor


def test_clean_text_basic():
    raw = "The product is GREAT!!!"
    assert clean_text(raw) == "the product is great"


def test_clean_text_contractions():
    raw = "I can't believe it didn't work and won't turn on."
    cleaned = clean_text(raw)
    assert "can not" in cleaned
    assert "did not" in cleaned
    assert "will not" in cleaned


def test_clean_text_html_and_urls():
    raw = "<div>Best cooler ever! Check https://example.com/item &amp; enjoy</div>"
    cleaned = clean_text(raw)
    assert "div" not in cleaned
    assert "https" not in cleaned
    assert "best cooler ever check enjoy" in cleaned


def test_clean_text_empty_and_none():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   \n\t  ") == ""


def test_text_preprocessor_fitting_and_transformation():
    texts = [
        "The product quality is excellent and delivery was fast.",
        "Very bad product, useless and terrible.",
        "Average item, okay for the price."
    ]
    labels = ["positive", "negative", "neutral"]

    preprocessor = TextPreprocessor(max_vocab_size=100, max_sequence_length=10)
    preprocessor.fit(texts, labels)

    assert preprocessor.tokenizer is not None
    assert preprocessor.label_encoder is not None
    assert set(preprocessor.classes_) == {"negative", "neutral", "positive"}

    # Test text transformation
    padded = preprocessor.transform_texts(texts)
    assert padded.shape == (3, 10)
    assert isinstance(padded, np.ndarray)

    # Test label encoding
    encoded_y = preprocessor.transform_labels(labels)
    assert len(encoded_y) == 3
    decoded_y = preprocessor.inverse_transform_labels(encoded_y)
    assert decoded_y == labels


def test_text_preprocessor_save_and_load():
    texts = ["Good product", "Bad product"]
    labels = ["positive", "negative"]

    preprocessor = TextPreprocessor(max_vocab_size=50, max_sequence_length=5)
    preprocessor.fit(texts, labels)

    with tempfile.TemporaryDirectory() as tmp_dir:
        preprocessor.save_artifacts(tmp_dir)

        loaded = TextPreprocessor.load_artifacts(tmp_dir)
        assert loaded.max_sequence_length == 5
        assert set(loaded.classes_) == {"negative", "positive"}

        padded_orig = preprocessor.transform_texts(["Good product"])
        padded_loaded = loaded.transform_texts(["Good product"])
        np.testing.assert_array_equal(padded_orig, padded_loaded)
