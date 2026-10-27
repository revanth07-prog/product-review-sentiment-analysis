"""
Application Configuration settings for Flask Backend.
"""

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class Config:
    HOST = os.environ.get("HOST", "127.0.0.1")
    PORT = int(os.environ.get("PORT", 5000))
    DEBUG = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1")
    
    ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
    FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "flipkart_product.csv")
    GLOVE_PATH = os.path.join(BASE_DIR, "embeddings", "glove.6B.100d.txt")
    
    MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lstm_glove_model.keras")
    METRICS_PATH = os.path.join(ARTIFACTS_DIR, "evaluation_metrics.json")
    TRAINING_SUMMARY_PATH = os.path.join(ARTIFACTS_DIR, "training_summary.json")
