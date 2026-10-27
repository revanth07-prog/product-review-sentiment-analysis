"""
API Blueprint routes for Product Review Sentiment Analysis backend.
"""

import os
import json
import logging
from flask import Blueprint, jsonify, request, send_from_directory, current_app
from werkzeug.exceptions import BadRequest

from backend.config import Config
from src.predict import SentimentPredictor

logger = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__)


@api_bp.route("/", methods=["GET"])
def api_root():
    """Basic API Information Endpoint."""
    model_loaded = False
    try:
        predictor = SentimentPredictor.get_instance(Config.ARTIFACTS_DIR)
        model_loaded = predictor.model is not None
    except Exception:
        model_loaded = False

    return jsonify({
        "project": "Product Review Sentiment Analysis Using LSTM with GloVe Embeddings",
        "version": "1.0.0",
        "model": "LSTM + GloVe",
        "dataset": "Flipkart Product Reviews (Kaggle)",
        "model_loaded": model_loaded,
        "endpoints": {
            "GET /": "API Information",
            "GET /api/health": "Health check & model status",
            "POST /api/predict": "Predict sentiment for review JSON",
            "GET /api/metrics": "Retrieve model evaluation metrics",
            "GET /api/artifacts/<filename>": "Retrieve evaluation visualization images"
        }
    }), 200


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Backend and Model Health Check."""
    model_exists = os.path.exists(Config.MODEL_PATH)
    metrics_exists = os.path.exists(Config.METRICS_PATH)
    glove_exists = os.path.exists(Config.GLOVE_PATH)

    model_loaded = False
    classes = []
    error_message = None

    if model_exists:
        try:
            predictor = SentimentPredictor.get_instance(Config.ARTIFACTS_DIR)
            model_loaded = predictor.model is not None
            classes = predictor.classes
        except Exception as e:
            error_message = str(e)
            logger.warning(f"Health check model load warning: {e}")
    else:
        error_message = "Trained model artifact not found. Please run 'python -m src.train' to train the model."

    return jsonify({
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_file_exists": model_exists,
        "evaluation_metrics_available": metrics_exists,
        "glove_file_exists": glove_exists,
        "supported_classes": classes,
        "message": "Model is loaded and ready for predictions." if model_loaded else error_message
    }), 200


@api_bp.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON payload: {"review": "string"}
    Returns sentiment prediction and confidence scores.
    """
    # 1. Validate JSON presence
    if not request.is_json:
        return jsonify({
            "error": "Invalid Content-Type. Request must be application/json."
        }), 415

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({
            "error": "Malformed JSON payload."
        }), 400

    # 2. Validate 'review' field
    if "review" not in data:
        return jsonify({
            "error": "Missing required field 'review' in JSON payload."
        }), 400

    review_text = data.get("review")
    if not isinstance(review_text, str) or not review_text.strip():
        return jsonify({
            "error": "Field 'review' must be a non-empty string."
        }), 400

    # 3. Model Loading & Prediction
    try:
        predictor = SentimentPredictor.get_instance(Config.ARTIFACTS_DIR)
    except FileNotFoundError as fnf:
        return jsonify({
            "error": "Model artifacts unavailable.",
            "details": str(fnf),
            "remedy": "Please execute 'python -m src.train' to train the LSTM model and generate artifacts."
        }), 503
    except Exception as e:
        logger.error(f"Error loading model for prediction: {e}")
        return jsonify({
            "error": "Model initialization failed.",
            "details": str(e)
        }), 500

    try:
        prediction_result = predictor.predict(review_text)
        return jsonify(prediction_result), 200
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return jsonify({
            "error": "Inference execution failed.",
            "details": str(e)
        }), 500


@api_bp.route("/metrics", methods=["GET"])
def get_metrics():
    """
    Return saved evaluation metrics and training summary.
    """
    if not os.path.exists(Config.METRICS_PATH):
        return jsonify({
            "status": "unavailable",
            "message": "Evaluation metrics not found. Run 'python -m src.evaluate' after training.",
            "metrics": None
        }), 200

    try:
        with open(Config.METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        training_summary = None
        if os.path.exists(Config.TRAINING_SUMMARY_PATH):
            with open(Config.TRAINING_SUMMARY_PATH, "r", encoding="utf-8") as f:
                training_summary = json.load(f)

        return jsonify({
            "status": "available",
            "metrics": metrics_data,
            "training_summary": training_summary,
            "available_charts": [
                f for f in ["confusion_matrix.png", "training_history.png", "class_distribution.png"]
                if os.path.exists(os.path.join(Config.ARTIFACTS_DIR, f))
            ]
        }), 200
    except Exception as e:
        logger.error(f"Error reading metrics: {e}")
        return jsonify({"error": "Failed to read evaluation metrics.", "details": str(e)}), 500


@api_bp.route("/artifacts/<path:filename>", methods=["GET"])
def get_artifact_file(filename):
    """
    Serve generated artifact visualization files safely.
    """
    allowed_extensions = {".png", ".jpg", ".jpeg", ".json"}
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_extensions:
        return jsonify({"error": "Access to requested file type is forbidden."}), 403

    safe_dir = Config.ARTIFACTS_DIR
    target_path = os.path.join(safe_dir, filename)
    if not os.path.exists(target_path):
        return jsonify({"error": f"Artifact '{filename}' not found."}), 404

    return send_from_directory(safe_dir, filename)
