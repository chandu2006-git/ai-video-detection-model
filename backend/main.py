from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi.middleware.cors import CORSMiddleware

from services.predictor import predict_video

import tempfile
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/predict")
async def predict(
    video: UploadFile = File(...)
):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ) as tmp:

        shutil.copyfileobj(
            video.file,
            tmp
        )

        temp_path = tmp.name

    result = predict_video(
        temp_path
    )

    return result