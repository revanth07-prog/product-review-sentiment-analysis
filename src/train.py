"""
Model Training Script for Product Review Sentiment Analysis.
Orchestrates data loading, GloVe embedding matrix creation, tokenizer fitting,
LSTM model compilation, model training with callbacks, and artifact persistence.
"""

import os
import sys
import json
import argparse
import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau

from src.data_loader import (
    load_raw_dataset,
    inspect_dataset,
    detect_columns,
    prepare_labeled_data,
    plot_class_distribution,
    split_dataset,
)
from src.preprocessing import TextPreprocessor
from src.glove_embeddings import load_glove_vectors, build_embedding_matrix
from src.model import build_sentiment_lstm_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def plot_training_history(history, output_path: str = "artifacts/training_history.png") -> None:
    """
    Plot and save training vs. validation accuracy and loss curves.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    epochs = range(1, len(history.history["accuracy"]) + 1)
    ax1.plot(epochs, history.history["accuracy"], "b-o", label="Training Accuracy", linewidth=2)
    ax1.plot(epochs, history.history["val_accuracy"], "g--s", label="Validation Accuracy", linewidth=2)
    ax1.set_title("Training vs Validation Accuracy", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Accuracy", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    # Loss Plot
    ax2.plot(epochs, history.history["loss"], "r-o", label="Training Loss", linewidth=2)
    ax2.plot(epochs, history.history["val_loss"], "m--s", label="Validation Loss", linewidth=2)
    ax2.set_title("Training vs Validation Loss", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Loss", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved training history curves to '{output_path}'")


def train(
    dataset_path: str = "dataset/flipkart_product.csv",
    glove_path: str = "embeddings/glove.6B.100d.txt",
    artifacts_dir: str = "artifacts",
    classification_mode: str = "multiclass",
    sample_size: Optional[int] = 30000,
    max_vocab_size: int = 20000,
    max_sequence_length: int = 100,
    embedding_dim: int = 100,
    trainable_embeddings: bool = False,
    lstm_units: int = 128,
    batch_size: int = 64,
    epochs: int = 8,
    seed: int = 42
) -> None:
    """
    Full end-to-end model training workflow.
    """
    set_seed(seed)
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Dataset Discovery & Loading
    if not os.path.exists(dataset_path):
        # Fallback to root directory if present
        if os.path.exists("flipkart_product.csv"):
            dataset_path = "flipkart_product.csv"
        else:
            raise FileNotFoundError(
                f"Dataset not found at '{dataset_path}'. Please place 'flipkart_product.csv' in dataset/ or workspace root."
            )

    logger.info(f"Loading dataset from: {dataset_path}")
    raw_df, enc = load_raw_dataset(dataset_path)

    # Save dataset inspection metadata
    inspection = inspect_dataset(raw_df)
    inspection_file = os.path.join(artifacts_dir, "dataset_inspection.json")
    with open(inspection_file, "w", encoding="utf-8") as f:
        json.dump(inspection, f, indent=2)
    logger.info(f"Saved dataset inspection metadata to '{inspection_file}'")

    # Detect review and rating columns
    text_cols, rating_col = detect_columns(raw_df)

    # Clean, map ratings to sentiments, and handle duplicates
    clean_df = prepare_labeled_data(
        raw_df,
        text_cols=text_cols,
        rating_col=rating_col,
        classification_mode=classification_mode,
        sample_size=sample_size,
        random_state=seed
    )

    # Plot class distribution
    dist_plot_path = os.path.join(artifacts_dir, "class_distribution.png")
    plot_class_distribution(clean_df, output_path=dist_plot_path)

    # Save test partition to file for independent evaluation
    train_df, val_df, test_df = split_dataset(clean_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=seed)
    test_csv_path = os.path.join(artifacts_dir, "test_split.csv")
    test_df.to_csv(test_csv_path, index=False, encoding="utf-8")
    logger.info(f"Saved held-out test split ({len(test_df)} samples) to '{test_csv_path}'")

    # 2. Fit Text Preprocessor
    logger.info("Fitting TextPreprocessor on training set...")
    preprocessor = TextPreprocessor(
        max_vocab_size=max_vocab_size,
        max_sequence_length=max_sequence_length
    )
    preprocessor.fit(texts=train_df["clean_review"].tolist(), labels=train_df["sentiment"].tolist())
    preprocessor.save_artifacts(artifacts_dir)

    # Transform datasets
    X_train = preprocessor.transform_texts(train_df["clean_review"].tolist())
    y_train = preprocessor.transform_labels(train_df["sentiment"].tolist())

    X_val = preprocessor.transform_texts(val_df["clean_review"].tolist())
    y_val = preprocessor.transform_labels(val_df["sentiment"].tolist())

    # 3. Load GloVe Embeddings
    embedding_matrix = None
    if os.path.exists(glove_path):
        try:
            embeddings_index, detected_dim = load_glove_vectors(glove_path, expected_dim=embedding_dim)
            embedding_dim = detected_dim
            vocab_size = min(len(preprocessor.tokenizer.word_index) + 1, max_vocab_size)
            embedding_matrix, glove_stats = build_embedding_matrix(
                word_index=preprocessor.tokenizer.word_index,
                embeddings_index=embeddings_index,
                vocab_size=vocab_size,
                embedding_dim=embedding_dim
            )
            glove_stats_file = os.path.join(artifacts_dir, "glove_coverage.json")
            with open(glove_stats_file, "w", encoding="utf-8") as f:
                json.dump(glove_stats, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not initialize GloVe embeddings from {glove_path}: {e}. Falling back to learned embeddings.")
    else:
        logger.warning(
            f"GloVe file not found at '{glove_path}'. Proceeding with randomly initialized trainable embeddings."
        )

    # 4. Build Model
    effective_vocab_size = min(len(preprocessor.tokenizer.word_index) + 1, max_vocab_size)
    num_classes = len(preprocessor.classes_)

    model = build_sentiment_lstm_model(
        vocab_size=effective_vocab_size,
        max_sequence_length=max_sequence_length,
        embedding_dim=embedding_dim,
        embedding_matrix=embedding_matrix,
        trainable_embeddings=trainable_embeddings or (embedding_matrix is None),
        lstm_units=lstm_units,
        num_classes=num_classes
    )

    # 5. Compute Balanced Class Weights (Essential for heavily imbalanced e-commerce reviews)
    from sklearn.utils.class_weight import compute_class_weight
    unique_classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=unique_classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(unique_classes, weights)}
    logger.info(f"Computed balanced class weights: {class_weight_dict}")

    # 6. Callbacks
    best_model_path = os.path.join(artifacts_dir, "lstm_glove_model.keras")
    history_csv = os.path.join(artifacts_dir, "training_history.csv")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=best_model_path, monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
        CSVLogger(filename=history_csv)
    ]

    # Handle binary targets format if binary classification
    train_targets = y_train.astype(np.float32) if num_classes == 2 else y_train
    val_targets = y_val.astype(np.float32) if num_classes == 2 else y_val

    # 7. Model Training
    logger.info(f"Starting model training for {epochs} epochs (batch_size={batch_size})...")
    history = model.fit(
        X_train,
        train_targets,
        validation_data=(X_val, val_targets),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model explicitly if checkpoint didn't trigger
    if not os.path.exists(best_model_path):
        model.save(best_model_path)
        logger.info(f"Saved trained model to '{best_model_path}'")

    # Plot & Save history curves
    hist_plot_path = os.path.join(artifacts_dir, "training_history.png")
    plot_training_history(history, output_path=hist_plot_path)

    # Save training configuration & summary
    training_summary = {
        "dataset_path": dataset_path,
        "total_dataset_rows": len(raw_df),
        "sampled_rows": len(clean_df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "classification_mode": classification_mode,
        "classes": preprocessor.classes_,
        "max_vocab_size": max_vocab_size,
        "effective_vocab_size": effective_vocab_size,
        "max_sequence_length": max_sequence_length,
        "embedding_dim": embedding_dim,
        "lstm_units": lstm_units,
        "epochs_trained": len(history.history["loss"]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "best_val_loss": float(min(history.history["val_loss"])),
    }
    summary_file = os.path.join(artifacts_dir, "training_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    logger.info("Training complete! Artifacts saved successfully.")

    try:
        from src.evaluate import evaluate_model
        logger.info("Running automatic independent evaluation on test set...")
        evaluate_model(artifacts_dir=artifacts_dir)
    except Exception as eval_err:
        logger.warning(f"Automatic post-training evaluation encountered an issue: {eval_err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM Sentiment Analysis model with GloVe.")
    parser.add_argument("--dataset_path", type=str, default="dataset/flipkart_product.csv")
    parser.add_argument("--glove_path", type=str, default="embeddings/glove.6B.100d.txt")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--mode", type=str, choices=["multiclass", "binary"], default="multiclass")
    parser.add_argument("--sample_size", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    train(
        dataset_path=args.dataset_path,
        glove_path=args.glove_path,
        artifacts_dir=args.artifacts_dir,
        classification_mode=args.mode,
        sample_size=args.sample_size,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
