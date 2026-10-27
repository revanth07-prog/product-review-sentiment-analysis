"""
Model Evaluation Script for Product Review Sentiment Analysis.
Evaluates the trained LSTM model on the held-out test split.
Generates Accuracy, Precision, Recall, Macro F1, Confusion Matrix, and detailed metrics.
"""

import os
import json
import logging
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.preprocessing import TextPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str = "artifacts/lstm_glove_model.keras",
    test_data_path: str = "artifacts/test_split.csv",
    artifacts_dir: str = "artifacts"
) -> Dict:
    """
    Load saved model, run inference on the held-out test dataset,
    calculate classification metrics, and save confusion matrix chart.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at: '{model_path}'. "
            f"Please run 'python -m src.train' before evaluating."
        )

    if not os.path.exists(test_data_path):
        raise FileNotFoundError(
            f"Test data split not found at: '{test_data_path}'. "
            f"Please run 'python -m src.train' to generate the held-out test split."
        )

    logger.info(f"Loading trained model from '{model_path}'...")
    model = tf.keras.models.load_model(model_path)

    logger.info(f"Loading preprocessing artifacts from '{artifacts_dir}'...")
    preprocessor = TextPreprocessor.load_artifacts(artifacts_dir)

    logger.info(f"Loading test split from '{test_data_path}'...")
    test_df = pd.read_csv(test_data_path, encoding="utf-8")
    texts = test_df["clean_review"].fillna("").tolist()
    true_labels = test_df["sentiment"].tolist()

    # Preprocess test texts and ground truth labels
    X_test = preprocessor.transform_texts(texts)
    y_true_indices = preprocessor.transform_labels(true_labels)
    classes = preprocessor.classes_
    num_classes = len(classes)

    logger.info(f"Running inference on {len(X_test)} test samples...")
    raw_preds = model.predict(X_test, batch_size=128, verbose=1)

    if num_classes == 2:
        # Binary model outputs sigmoid single value
        y_pred_indices = (raw_preds.flatten() >= 0.5).astype(int)
    else:
        # Multiclass model outputs softmax distribution
        y_pred_indices = np.argmax(raw_preds, axis=1)

    # Calculate real evaluation metrics
    accuracy = float(accuracy_score(y_true_indices, y_pred_indices))
    precision_macro = float(precision_score(y_true_indices, y_pred_indices, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_true_indices, y_pred_indices, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true_indices, y_pred_indices, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true_indices, y_pred_indices, average="weighted", zero_division=0))

    cls_report_dict = classification_report(
        y_true_indices,
        y_pred_indices,
        target_names=classes,
        output_dict=True,
        zero_division=0
    )
    cls_report_text = classification_report(
        y_true_indices,
        y_pred_indices,
        target_names=classes,
        zero_division=0
    )

    logger.info(f"\nClassification Report:\n{cls_report_text}")

    # Generate Confusion Matrix
    cm = confusion_matrix(y_true_indices, y_pred_indices)
    cm_path = os.path.join(artifacts_dir, "confusion_matrix.png")

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        annot_kws={"size": 12, "weight": "bold"}
    )
    plt.title("Confusion Matrix (Held-out Test Set)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Sentiment", fontsize=11, fontweight="semibold")
    plt.ylabel("Actual Sentiment", fontsize=11, fontweight="semibold")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()
    logger.info(f"Saved confusion matrix visualization to '{cm_path}'")

    # Save metrics JSON for API / Frontend consumption
    metrics = {
        "test_sample_count": len(test_df),
        "classes": classes,
        "accuracy": round(accuracy, 4),
        "precision_macro": round(precision_macro, 4),
        "recall_macro": round(recall_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": cls_report_dict,
        "per_class_metrics": {
            cls_name: {
                "precision": round(cls_report_dict[cls_name]["precision"], 4),
                "recall": round(cls_report_dict[cls_name]["recall"], 4),
                "f1_score": round(cls_report_dict[cls_name]["f1-score"], 4),
                "support": int(cls_report_dict[cls_name]["support"])
            }
            for cls_name in classes if cls_name in cls_report_dict
        }
    }

    metrics_file = os.path.join(artifacts_dir, "evaluation_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved evaluation metrics to '{metrics_file}'")
    print(f"\nEvaluation Results:\n"
          f"Accuracy:    {accuracy * 100:.2f}%\n"
          f"Precision:   {precision_macro * 100:.2f}%\n"
          f"Recall:      {recall_macro * 100:.2f}%\n"
          f"Macro F1:    {f1_macro * 100:.2f}%\n"
          f"Weighted F1: {f1_weighted * 100:.2f}%\n")

    return metrics


if __name__ == "__main__":
    evaluate_model()
