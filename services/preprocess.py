import cv2
import numpy as np
import os

from tensorflow.keras.applications.resnet50 import (
ResNet50,
preprocess_input
)

IMG_SIZE = 224
SEQ_LEN = 30

_base_cnn = None

def get_resnet():


    global _base_cnn

    if _base_cnn is None:

        BASE_DIR = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        RESNET_PATH = os.path.join(
            BASE_DIR,
            "models",
            "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
        )

        print("================================")
        print("ResNet Path:", RESNET_PATH)
        print("Exists:", os.path.exists(RESNET_PATH))
        print("Loading ResNet50...")

        _base_cnn = ResNet50(
            weights=RESNET_PATH,
            include_top=False,
            pooling="avg"
        )

        _base_cnn.trainable = False

        print("ResNet50 Loaded")
        print("================================")

    return _base_cnn


def extract_frames(video_path):

    
    print("FRAME EXTRACTION START")

    cap = cv2.VideoCapture(video_path)

    frames = []

    total = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print("Total Frames:", total)

    step = max(
        total // SEQ_LEN,
        1
    )

    i = 0

    while len(frames) < SEQ_LEN:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            i
        )

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(
            frame,
            (IMG_SIZE, IMG_SIZE)
        )

        frames.append(frame)

        i += step

    cap.release()

    if len(frames) == 0:

        raise ValueError(
            "No frames could be extracted from the video."
        )

    while len(frames) < SEQ_LEN:

        frames.append(
            frames[-1]
        )

    print("Frames Extracted:", len(frames))

    return np.array(frames)

def extract_features(video_path):


    print("FEATURE EXTRACTION START")

    frames = extract_frames(
        video_path
    )

    print("Frames Shape:", frames.shape)

    frames = preprocess_input(
        frames.astype(np.float32)
    )

    print("Loading ResNet")

    cnn = get_resnet()

    print("Running ResNet50")

    feats = cnn.predict(
        frames,
        verbose=0
    )

    print("ResNet50 Complete")

    print("Features Shape:", feats.shape)

    return feats.reshape(
        1,
        SEQ_LEN,
        -1
    )
