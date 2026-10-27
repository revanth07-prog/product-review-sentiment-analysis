# Product Review Sentiment Analysis Using LSTM with GloVe Embeddings

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00.svg?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000.svg?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end Deep Learning and Natural Language Processing (NLP) system that classifies customer sentiment from e-commerce product reviews. The model uses a Recurrent Neural Network with **Long Short-Term Memory (LSTM)** layers initialized with **Stanford pre-trained GloVe word embeddings** (100-dimensional), trained on the **Flipkart Product Reviews dataset from Kaggle**, and served through an interactive **Flask REST API** and **HTML5/CSS3/JavaScript responsive Single Page Application (SPA)**.

---

## 📌 Project Architecture

```
                                 [ Flipkart Customer Review ]
                                               │
                                               ▼
                              [ Text Normalization & Cleaning ]
                                (HTML, Contractions, Symbols)
                                               │
                                               ▼
                                 [ Tokenization & Padding ]
                               (Vocabulary Indexing, maxlen=100)
                                               │
                                               ▼
                             [ GloVe-Initialized Embedding Layer ]
                             (100-dimensional continuous vectors)
                                               │
                                               ▼
                                     [ Recurrent LSTM Layer ]
                                   (128 units, sequence context)
                                               │
                                               ▼
                                     [ Dropout Layer (0.3) ]
                                               │
                                               ▼
                                    [ Dense Feature Layer ]
                                      (64 units, ReLU)
                                               │
                                               ▼
                                  [ Sentiment Output Layer ]
                                  (Softmax / 3 Class Output)
                                               │
                   ┌───────────────────────────┼───────────────────────────┐
                   ▼                           ▼                           ▼
          Negative (1-2★)              Neutral (3★)               Positive (4-5★)
```

---

## 🗂️ Project Directory Structure

```
product-review-sentiment-analysis/
├── dataset/                     # Dataset storage
│   └── flipkart_product.csv     # 189,874 Flipkart buyer reviews from Kaggle
├── embeddings/                  # Word vectors storage
│   ├── glove.6B.100d.txt        # 100-dimensional Stanford GloVe vectors (347 MB)
│   └── glove.6B.50d.txt         # 50-dimensional GloVe vectors (optional)
├── artifacts/                   # Persisted checkpoints, metrics, and plots
│   ├── lstm_glove_model.keras   # Trained Keras 3 LSTM model
│   ├── tokenizer.pkl            # Fitted Keras Tokenizer
│   ├── label_encoder.pkl        # Fitted Scikit-learn LabelEncoder
│   ├── preprocessing_meta.json  # Vocabulary size, sequence length, classes
│   ├── training_summary.json    # Epoch counts, loss, and training stats
│   ├── evaluation_metrics.json  # Accuracy, Precision, Recall, Macro F1
│   ├── test_split.csv           # Held-out test split for independent evaluation
│   ├── class_distribution.png   # Class distribution plot
│   ├── training_history.png     # Train vs. Validation Loss and Accuracy curves
│   └── confusion_matrix.png     # Confusion matrix heatmap on held-out test data
├── notebooks/                   # Interactive Jupyter notebooks
│   └── exploration_and_pipeline.ipynb
├── src/                         # Core Machine Learning & NLP modules
│   ├── __init__.py
│   ├── data_loader.py           # Robust CSV loader, inspection, cleaning, stratified split
│   ├── preprocessing.py         # Text cleaning, Tokenizer, LabelEncoder, sequence padding
│   ├── glove_embeddings.py      # GloVe vector parser, matrix construction, coverage reporter
│   ├── download_glove.py        # Automated GloVe downloader with fallback mirrors
│   ├── model.py                 # Keras LSTM architecture with GloVe weights
│   ├── train.py                 # Training orchestrator with early stopping & checkpoints
│   ├── evaluate.py              # Independent test evaluation and confusion matrix generator
│   └── predict.py               # Singleton inference engine for live predictions
├── backend/                     # Flask REST API server
│   ├── __init__.py
│   ├── app.py                   # App factory, static frontend serving, model pre-warming
│   ├── routes.py                # REST endpoints: /, /api/health, /api/predict, /api/metrics
│   └── config.py                # Server configuration and environment variable bindings
├── frontend/                    # Single Page Application
│   ├── index.html               # Semantic HTML5 UI (Dashboard, Live Analyzer, Performance, About)
│   ├── css/
│   │   └── style.css            # Responsive vanilla CSS styling
│   └── js/
│       └── app.js               # Frontend controller and API integration
├── tests/                       # Automated test suite
│   ├── test_preprocessing.py    # Unit tests for text cleaning and preprocessor
│   └── test_api.py              # API endpoint validation and health tests
├── requirements.txt             # Pinned Python package dependencies
├── .gitignore                   # Ignore datasets, embeddings, weights, and caches
├── .env.example                 # Sample environment variables
└── README.md                    # Comprehensive documentation
```

---

## ⚙️ Environment Setup & Installation

### 1. Prerequisites
- Python 3.10 or 3.11 installed on your system.
- PowerShell (Windows) or Terminal (macOS/Linux).

### 2. Create and Activate Virtual Environment
Open PowerShell inside the project directory:

```powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# (On Linux / macOS use: source .venv/bin/activate)
```

### 3. Install Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 📦 Dataset & GloVe Embeddings Setup

### Dataset
The **Flipkart Product Reviews dataset from Kaggle** (`flipkart_product.csv`) is automatically loaded from the `dataset/` directory.
- Rows: **189,874**
- Columns: `ProductName`, `Price`, `Rate`, `Review`, `Summary`
- Rating mapping:
  - Ratings `1` and `2` &rarr; `negative`
  - Rating `3` &rarr; `neutral`
  - Ratings `4` and `5` &rarr; `positive`

### GloVe Embeddings
Pre-trained **Stanford GloVe vectors** (`glove.6B.100d.txt`) provide 400,000 token vectors mapped into a 100-dimensional space.
If not already downloaded, fetch it automatically using:

```powershell
python -m src.download_glove 100
```
This downloads `glove.6B.zip`, extracts `glove.6B.100d.txt` directly into `embeddings/`, and cleans up the temporary archive.

---

## 🚀 Execution Commands

### 1. Model Training
Train the LSTM model with pre-trained GloVe weights:

```powershell
python -m src.train --sample_size 30000 --epochs 5 --batch_size 64
```

**Training parameters:**
- `--sample_size`: Stratified review sample count (default: `30000` for fast CPU training, or omit to train on full dataset).
- `--epochs`: Number of epochs (default: `5`, with early stopping).
- `--batch_size`: Batch size (default: `64`).
- `--mode`: `multiclass` (3 classes: negative, neutral, positive) or `binary` (negative, positive).

Training automatically saves:
- `artifacts/lstm_glove_model.keras`
- `artifacts/tokenizer.pkl`
- `artifacts/label_encoder.pkl`
- `artifacts/class_distribution.png`
- `artifacts/training_history.png`
- `artifacts/test_split.csv`

### 2. Independent Model Evaluation
Run standalone evaluation against the held-out test split:

```powershell
python -m src.evaluate
```

This outputs Accuracy, Precision, Recall, Macro F1, Weighted F1, and saves `artifacts/confusion_matrix.png` and `artifacts/evaluation_metrics.json`.

### 3. Run Automated Tests
```powershell
python -m pytest tests/
```

### 4. Launch Web Application
Start the Flask backend and web interface:

```powershell
python -m backend.app
```

Then open your browser to:
**`http://127.0.0.1:5000`**

---

## 🌐 API Reference

### Health Check
- **Endpoint:** `GET /api/health`
- **Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "supported_classes": ["negative", "neutral", "positive"],
  "evaluation_metrics_available": true
}
```

### Sentiment Prediction
- **Endpoint:** `POST /api/predict`
- **Headers:** `Content-Type: application/json`
- **Request Body:**
```json
{
  "review": "The product quality is excellent and delivery was fast."
}
```
- **Response:**
```json
{
  "review": "The product quality is excellent and delivery was fast.",
  "cleaned_review": "the product quality is excellent and delivery was fast",
  "predicted_sentiment": "positive",
  "confidence": 0.9412,
  "probabilities": {
    "negative": 0.0215,
    "neutral": 0.0373,
    "positive": 0.9412
  }
}
```

### Evaluation Metrics
- **Endpoint:** `GET /api/metrics`
- **Response:** Returns JSON of Accuracy, Macro F1, Per-Class metrics, and available chart image names.

---

## 📊 Evaluation & Performance Methodology

- **Class Imbalance Handling:** E-commerce reviews are heavily skewed toward positive reviews (5 stars make up >55% of the data). We use **Stratified Sampling** and report **Macro-F1** alongside Accuracy so performance on minority classes (Negative and Neutral) is rigorously assessed.
- **Data Leakage Prevention:** The tokenizer, vocabulary indexing, and label encoders are fitted exclusively on the training partition. Sequences in validation and test partitions are transformed using the fitted training artifacts.
- **Overfitting Prevention:** Spatial Dropout (0.3) after both LSTM recurrent units and Dense projection layers, coupled with `EarlyStopping(patience=3)` on validation loss.

---

## ⚠️ Limitations & Future Work

1. **Domain-Specific Vocabulary:** GloVe 6B was pre-trained on Wikipedia and Gigaword corpora. Colloquial phrasing, shorthand (e.g. *"vfm"* for value-for-money), and Hinglish words common on Indian e-commerce platforms can be Out-Of-Vocabulary (OOV).
2. **Contextual Polysemy:** As a static word embedding, GloVe assigns a single fixed vector per token regardless of context (unlike contextual transformer models like BERT).
3. **Aspect-Level Nuance:** The model classifies the global sentiment of the entire review. Dual-sentiment reviews (e.g., *"Amazing camera, but terrible battery life"*) are challenging without aspect-based sentiment decomposition.

---

## 📜 License
This project is open-source under the MIT License.
