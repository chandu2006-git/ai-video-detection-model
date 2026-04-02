import os
import json
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "appearance_lstm_clean_v2_80.keras")
INFO_PATH  = os.path.join(MODEL_DIR, "model_info.json")

_model = None
_info = None

def load_model_once():
    global _model, _info

    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        _model.trainable = False

        with open(INFO_PATH, "r") as f:
            _info = json.load(f)

    return _model, _info
