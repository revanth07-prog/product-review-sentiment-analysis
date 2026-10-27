"""
Inference Module for Live Product Review Sentiment Analysis.
Loads the trained LSTM model and preprocessing artifacts once at startup,
cleans live review inputs, and returns predicted sentiment, confidence, and class probabilities.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union

import numpy as np
import tensorflow as tf

from src.preprocessing import clean_text, TextPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SentimentPredictor:
    """
    Thread-safe Sentiment Predictor singleton.
    Loads Keras model and tokenizer artifacts once, providing low-latency inference.
    """

    _instance: Optional["SentimentPredictor"] = None

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self.model_path = os.path.join(artifacts_dir, "lstm_glove_model.keras")
        self.model: Optional[tf.keras.Model] = None
        self.preprocessor: Optional[TextPreprocessor] = None
        self.classes: List[str] = []
        self._load()

    def _load(self) -> None:
        """Load model and preprocessor artifacts."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Trained model not found at: '{self.model_path}'. "
                f"Please train the model first by running 'python -m src.train'."
            )

        logger.info(f"Loading trained model from '{self.model_path}'...")
        self.model = tf.keras.models.load_model(self.model_path)

        logger.info(f"Loading preprocessor from '{self.artifacts_dir}'...")
        self.preprocessor = TextPreprocessor.load_artifacts(self.artifacts_dir)
        self.classes = self.preprocessor.classes_
        logger.info(f"SentimentPredictor ready with classes: {self.classes}")

    @classmethod
    def get_instance(cls, artifacts_dir: str = "artifacts", force_reload: bool = False) -> "SentimentPredictor":
        """Singleton accessor to ensure artifacts are loaded into memory only once, unless force_reload=True."""
        if cls._instance is None or force_reload:
            cls._instance = cls(artifacts_dir=artifacts_dir)
        return cls._instance

    def predict(self, review_text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Predict sentiment for a single review or list of reviews.
        Returns:
            Dict containing predicted_sentiment, confidence, probabilities, and cleaned review.
        """
        is_single = isinstance(review_text, str)
        raw_texts = [review_text] if is_single else review_text

        cleaned_texts = [clean_text(t) for t in raw_texts]

        # Handle empty/whitespace-only input gracefully
        results = []
        texts_to_process = []
        valid_indices = []

        for i, (orig, cleaned) in enumerate(zip(raw_texts, cleaned_texts)):
            if not cleaned:
                results.append({
                    "review": orig,
                    "cleaned_review": "",
                    "predicted_sentiment": "neutral" if "neutral" in self.classes else self.classes[0],
                    "confidence": 0.0,
                    "probabilities": {c: round(1.0 / len(self.classes), 4) for c in self.classes},
                    "note": "Input text contained no usable words after cleaning."
                })
            else:
                results.append(None)  # Placeholder
                texts_to_process.append(cleaned)
                valid_indices.append(i)

        if texts_to_process:
            # Transform to padded sequence
            X = self.preprocessor.transform_texts(texts_to_process)
            preds = self.model.predict(X, verbose=0)

            for idx, pred_vector in zip(valid_indices, preds):
                orig_text = raw_texts[idx]
                cleaned_text = cleaned_texts[idx]

                if len(self.classes) == 2:
                    # Binary classification (sigmoid output)
                    pos_prob = float(pred_vector[0])
                    neg_prob = 1.0 - pos_prob
                    pred_class = self.classes[1] if pos_prob >= 0.5 else self.classes[0]
                    confidence = pos_prob if pred_class == self.classes[1] else neg_prob
                    probabilities = {
                        self.classes[0]: round(neg_prob, 4),
                        self.classes[1]: round(pos_prob, 4)
                    }
                else:
                    # Multiclass classification (softmax output)
                    pred_idx = int(np.argmax(pred_vector))
                    pred_class = self.classes[pred_idx]
                    confidence = float(pred_vector[pred_idx])
                    probabilities = {
                        cls_name: round(float(prob), 4)
                        for cls_name, prob in zip(self.classes, pred_vector)
                    }

                results[idx] = {
                    "review": orig_text,
                    "cleaned_review": cleaned_text,
                    "predicted_sentiment": pred_class,
                    "confidence": round(confidence, 4),
                    "probabilities": probabilities
                }

        return results[0] if is_single else results


def predict_sentiment(review_text: str, artifacts_dir: str = "artifacts") -> Dict:
    """Convenience function for predicting sentiment on a single review string."""
    predictor = SentimentPredictor.get_instance(artifacts_dir=artifacts_dir)
    return predictor.predict(review_text)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict sentiment for a product review.")
    parser.add_argument("--review", type=str, default="The product quality is excellent and delivery was fast.")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    args = parser.parse_args()

    try:
        res = predict_sentiment(args.review, artifacts_dir=args.artifacts_dir)
        print("\nPrediction Result:")
        print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"Prediction failed: {e}")
