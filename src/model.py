"""
LSTM Model Architecture Definition and Compilation.
Implements Input -> Embedding (GloVe) -> LSTM -> Dropout -> Dense -> Output.
Supports binary and multiclass sentiment classification.
"""

import logging
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dropout, Dense
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam

logger = logging.getLogger(__name__)


def build_sentiment_lstm_model(
    vocab_size: int,
    max_sequence_length: int,
    embedding_dim: int,
    embedding_matrix: Optional[np.ndarray] = None,
    trainable_embeddings: bool = False,
    lstm_units: int = 128,
    dropout_rate: float = 0.3,
    dense_units: int = 64,
    num_classes: int = 3,
    learning_rate: float = 0.001
) -> Model:
    """
    Constructs and compiles the LSTM Sentiment Analysis model.
    Structure:
    Input -> GloVe Embedding (mask_zero=True) -> Bidirectional LSTM -> Dropout -> Dense -> Dropout -> Output
    """
    # 1. Input Layer
    inputs = Input(shape=(max_sequence_length,), dtype="int32", name="input_review")

    # 2. Embedding Layer (Initialized with GloVe weights if available)
    if embedding_matrix is not None:
        embedding_layer = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            embeddings_initializer=Constant(embedding_matrix),
            trainable=trainable_embeddings,
            mask_zero=True,
            name="glove_embedding"
        )(inputs)
    else:
        embedding_layer = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,
            name="learned_embedding"
        )(inputs)

    # 3. Bidirectional LSTM Layer
    lstm_layer = Bidirectional(
        LSTM(units=max(32, lstm_units // 2), return_sequences=False),
        name="bidirectional_lstm"
    )(embedding_layer)

    # 4. Dropout Layer
    dropout_1 = Dropout(dropout_rate, name="dropout_lstm")(lstm_layer)

    # 5. Intermediate Dense Layer
    dense_layer = Dense(dense_units, activation="relu", name="dense_features")(dropout_1)
    dropout_2 = Dropout(dropout_rate, name="dropout_dense")(dense_layer)

    # 6. Sentiment Output Layer
    if num_classes == 2:
        # Binary sentiment output: single sigmoid probability
        outputs = Dense(1, activation="sigmoid", name="sentiment_output")(dropout_2)
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    else:
        # Multiclass sentiment output: softmax probability distribution
        outputs = Dense(num_classes, activation="softmax", name="sentiment_output")(dropout_2)
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    model = Model(inputs=inputs, outputs=outputs, name="LSTM_GloVe_Sentiment_Classifier")

    # Compile with Adam optimizer
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    logger.info(f"Built LSTM Sentiment model: num_classes={num_classes}, loss={loss}")
    model.summary(print_fn=lambda x: logger.debug(x))
    return model
