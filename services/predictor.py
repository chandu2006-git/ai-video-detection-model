from services.model_loader import load_model_once
from services.preprocess import extract_features


def predict_video(video_path):

    print("STEP 1: predict_video entered")

    print("STEP 2: loading CNN-LSTM model")
    model, info = load_model_once()

    print("STEP 3: extracting features")
    X = extract_features(video_path)

    print("STEP 4: features extracted")
    print("Feature Shape:", X.shape)

    print("STEP 5: running CNN-LSTM prediction")

    prob = float(
        model.predict(X)[0][0]
    )

    print("STEP 6: prediction complete")
    print("Probability:", prob)

    if prob < 0.45:
        label = "REAL VIDEO"

    elif prob > 0.60:
        label = "AI GENERATED"

    else:
        label = "UNCERTAIN"

    ai_prob = round(
        prob * 100,
        2
    )

    real_prob = round(
        (1 - prob) * 100,
        2
    )

    confidence = max(
        ai_prob,
        real_prob
    )

    return {
        "label": label,
        "ai_probability": ai_prob,
        "real_probability": real_prob,
        "confidence": confidence
    }