from services.model_loader import load_model_once
from services.preprocess import extract_features

def predict_video(video_path):
    model, info = load_model_once()

    X = extract_features(video_path)
    prob = float(model.predict(X)[0][0])

    if prob < 0.45:
        label = "REAL VIDEO"
    elif prob > 0.60:
        label = "AI GENERATED"
    else:
        label = "UNCERTAIN"

    return {
        "label": label,
        "ai_probability": round(prob * 100, 2)
    }

