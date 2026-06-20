from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi import HTTPException

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from services.predictor import predict_video
from services.preprocess import get_resnet
from services.model_loader import load_model_once

import tempfile
import shutil
import traceback
import os

app = FastAPI()

# ---------------------------------
# CORS
# ---------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---------------------------------
# STARTUP
# ---------------------------------

@app.on_event("startup")
async def startup_event():

    print("================================")
    print("LOADING RESNET50...")
    get_resnet()

    print("LOADING CNN-LSTM MODEL...")
    load_model_once()

    print("MODELS READY")
    print("================================")


# ---------------------------------
# HEALTH CHECK
# ---------------------------------

@app.get("/")
def home():

    return {
        "status": "online",
        "model": "CNN-LSTM",
        "version": "2.0"
    }


# ---------------------------------
# TEST MODEL
# ---------------------------------

@app.get("/test-model")
def test_model():

    get_resnet()

    return {
        "status": "loaded"
    }


# ---------------------------------
# PREDICTION
# ---------------------------------

@app.post("/predict")
async def predict(
    video: UploadFile = File(...)
):

    temp_path = None

    try:

        print("========== PREDICT START ==========")

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as tmp:

            shutil.copyfileobj(
                video.file,
                tmp
            )

            temp_path = tmp.name

        print("Video Saved:", temp_path)

        print("Calling predict_video...")

        result = predict_video(
            temp_path
        )

        print("Prediction Complete")
        print(result)

        return result

    except Exception as e:

        print("========== ERROR ==========")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        if temp_path and os.path.exists(temp_path):

            try:
                os.remove(temp_path)
                print("Temporary file removed")

            except Exception:
                pass


# ---------------------------------
# FRONTEND
# ---------------------------------

app.mount(
    "/frontend",
    StaticFiles(directory="frontend"),
    name="frontend"
)

@app.get("/app")
async def index():

    return FileResponse(
        "frontend/index.html"
    )

@app.get("/test-lstm")
def test_lstm():

    from services.model_loader import load_model_once

    load_model_once()

    return {
        "status": "lstm_loaded"
    }