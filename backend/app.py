"""
Main Flask Web Application and API Server.
Configures CORS, registers API blueprints, serves frontend files, and preloads model.
"""

import os
import sys
import logging
from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

from backend.config import Config
from backend.routes import api_bp
from src.predict import SentimentPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_app(config_class=Config) -> Flask:
    """Application factory for Flask backend."""
    app = Flask(
        __name__,
        static_folder=config_class.FRONTEND_DIR,
        static_url_path=""
    )
    app.config.from_object(config_class)

    # Enable Cross-Origin Resource Sharing
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register API blueprint under /api prefix
    app.register_blueprint(api_bp, url_prefix="/api")

    # Serve Frontend Single Page Application
    @app.route("/")
    def index():
        return send_from_directory(config_class.FRONTEND_DIR, "index.html")

    @app.route("/<path:path>")
    def static_files(path):
        target = os.path.join(config_class.FRONTEND_DIR, path)
        if os.path.exists(target):
            return send_from_directory(config_class.FRONTEND_DIR, path)
        return send_from_directory(config_class.FRONTEND_DIR, "index.html")

    # Error Handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Resource not found", "status_code": 404}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "Internal server error", "status_code": 500}), 500

    # Model pre-warming at server startup
    if os.path.exists(config_class.MODEL_PATH):
        try:
            logger.info("Pre-warming model and tokenizer singleton into memory...")
            SentimentPredictor.get_instance(config_class.ARTIFACTS_DIR)
            logger.info("Model pre-warming complete.")
        except Exception as e:
            logger.warning(f"Model pre-warming deferred: {e}")
    else:
        logger.info("Model artifact not yet generated. Ready to train via 'python -m src.train'.")

    return app


def main():
    app = create_app()
    logger.info(f"Starting Sentiment Analysis Backend on http://{Config.HOST}:{Config.PORT}")
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)


if __name__ == "__main__":
    main()
