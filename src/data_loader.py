"""
Data Loader Module for Product Review Sentiment Analysis.
Loads, validates, cleans, inspects, and splits Flipkart product review datasets.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard rating-to-sentiment mappings
DEFAULT_3CLASS_MAPPING = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}

DEFAULT_BINARY_MAPPING = {
    1: "negative",
    2: "negative",
    4: "positive",
    5: "positive",
}


def load_raw_dataset(
    file_path: str,
    encodings: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Attempt loading CSV dataset using various encodings.
    Handles encoding differences across platforms and Kaggle exports.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset file not found at: '{file_path}'. "
            f"Please ensure 'flipkart_product.csv' is placed inside 'dataset/' or project root."
        )

    if encodings is None:
        encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]

    df = None
    successful_encoding = None

    for enc in encodings:
        try:
            logger.info(f"Attempting to read '{file_path}' with encoding '{enc}'...")
            df = pd.read_csv(file_path, encoding=enc, low_memory=False)
            successful_encoding = enc
            logger.info(f"Successfully loaded dataset with encoding: {enc}")
            break
        except (UnicodeDecodeError, Exception) as e:
            logger.debug(f"Failed loading with {enc}: {e}")

    if df is None:
        raise ValueError(f"Could not read CSV file '{file_path}' with any tested encoding: {encodings}")

    return df, successful_encoding


def inspect_dataset(df: pd.DataFrame) -> Dict:
    """
    Inspect columns, missing value counts, memory usage, and shape of raw DataFrame.
    """
    inspection = {
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample_head": df.head(3).to_dict(orient="records"),
    }
    return inspection


def detect_columns(
    df: pd.DataFrame,
    text_column: Optional[Union[str, List[str]]] = None,
    rating_column: Optional[str] = None
) -> Tuple[List[str], str]:
    """
    Identify review text and rating/sentiment columns automatically or validate user inputs.
    """
    cols_lower = {col.lower(): col for col in df.columns}

    # Detect rating column
    detected_rating = None
    if rating_column and rating_column in df.columns:
        detected_rating = rating_column
    else:
        for candidate in ["rate", "rating", "stars", "score", "sentiment"]:
            if candidate in cols_lower:
                detected_rating = cols_lower[candidate]
                break

    if not detected_rating:
        raise ValueError(
            f"Could not identify a rating or sentiment column in columns: {list(df.columns)}. "
            f"Please specify 'rating_column' explicitly."
        )

    # Detect text columns
    detected_text_cols: List[str] = []
    if text_column:
        if isinstance(text_column, str):
            if text_column in df.columns:
                detected_text_cols = [text_column]
            else:
                raise ValueError(f"Specified text column '{text_column}' not found in {list(df.columns)}")
        elif isinstance(text_column, list):
            for tc in text_column:
                if tc in df.columns:
                    detected_text_cols.append(tc)
                else:
                    logger.warning(f"Specified text column '{tc}' not in columns; skipping.")
    else:
        # For Flipkart reviews, both 'Review' (title) and 'Summary' (body) are valuable
        for candidate in ["summary", "review", "review_text", "reviews", "text", "comment", "description"]:
            if candidate in cols_lower:
                col_name = cols_lower[candidate]
                if col_name not in detected_text_cols:
                    detected_text_cols.append(col_name)

    if not detected_text_cols:
        raise ValueError(
            f"Could not identify review text column(s) in {list(df.columns)}. "
            f"Please specify 'text_column' explicitly."
        )

    logger.info(f"Using text column(s): {detected_text_cols} and rating column: '{detected_rating}'")
    return detected_text_cols, detected_rating


def prepare_labeled_data(
    df: pd.DataFrame,
    text_cols: List[str],
    rating_col: str,
    classification_mode: str = "multiclass",
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Cleans raw dataframe, parses ratings into sentiment labels, combines text fields,
    removes missing values and duplicates.
    """
    data = df.copy()

    # Drop rows where rating is null
    data = data.dropna(subset=[rating_col])

    # Convert rating to numeric (handles strings like '5', and filters out malformed shifted rows)
    data["numeric_rating"] = pd.to_numeric(data[rating_col], errors="coerce")
    data = data.dropna(subset=["numeric_rating"])
    data["numeric_rating"] = data["numeric_rating"].astype(int)

    # Select mapping
    if classification_mode == "binary":
        mapping = DEFAULT_BINARY_MAPPING
        data = data[data["numeric_rating"].isin(mapping.keys())]
    else:
        mapping = DEFAULT_3CLASS_MAPPING
        data = data[data["numeric_rating"].isin(mapping.keys())]

    data["sentiment"] = data["numeric_rating"].map(mapping)

    # Combine text fields if multiple (e.g. Flipkart 'Review' title + 'Summary' body)
    if len(text_cols) == 1:
        data["clean_review"] = data[text_cols[0]].astype(str).fillna("")
    else:
        data["clean_review"] = data[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

    # Clean empty or whitespace-only reviews
    data["clean_review"] = data["clean_review"].str.strip()
    data = data[data["clean_review"].str.len() > 2]

    # Remove duplicates
    initial_len = len(data)
    data = data.drop_duplicates(subset=["clean_review"])
    logger.info(f"Removed {initial_len - len(data)} duplicate reviews; {len(data)} unique reviews remain.")

    # Optional subsampling for training efficiency while maintaining stratification
    if sample_size and sample_size < len(data):
        logger.info(f"Subsampling {sample_size} records from {len(data)} using stratified sampling...")
        data, _ = train_test_split(
            data,
            train_size=sample_size,
            stratify=data["sentiment"],
            random_state=random_state
        )

    return data[["clean_review", "sentiment", "numeric_rating"]]


def plot_class_distribution(
    df: pd.DataFrame,
    output_path: str = "artifacts/class_distribution.png"
) -> None:
    """
    Generate and save class distribution chart.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    order = ["negative", "neutral", "positive"] if "neutral" in df["sentiment"].unique() else ["negative", "positive"]
    palette = {"positive": "#10b981", "neutral": "#f59e0b", "negative": "#ef4444"}

    counts = df["sentiment"].value_counts()
    ax = sns.barplot(
        x=counts.index,
        y=counts.values,
        order=[c for c in order if c in counts.index],
        palette=palette
    )
    plt.title("Sentiment Class Distribution", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Sentiment Class", fontsize=11)
    plt.ylabel("Number of Reviews", fontsize=11)

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{int(height):,}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="semibold",
                xytext=(0, 3),
                textcoords="offset points"
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved class distribution visualization to '{output_path}'")


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split into training, validation, and test partitions without data leakage.
    """
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    # Step 1: Split into train and temporary test+val
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        stratify=df["sentiment"],
        random_state=random_state
    )

    # Step 2: Split temp into validation and test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df["sentiment"],
        random_state=random_state
    )

    logger.info(
        f"Dataset split complete: Train={len(train_df)} ({len(train_df)/len(df)*100:.1f}%), "
        f"Val={len(val_df)} ({len(val_df)/len(df)*100:.1f}%), "
        f"Test={len(test_df)} ({len(test_df)/len(df)*100:.1f}%)"
    )

    return train_df, val_df, test_df
