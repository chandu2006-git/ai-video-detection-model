import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

IMG_SIZE = 224
SEQ_LEN = 30

_base_cnn = None

def get_resnet():
    global _base_cnn
    if _base_cnn is None:
        _base_cnn = ResNet50(
            weights="imagenet",
            include_top=False,
            pooling="avg"
        )
        _base_cnn.trainable = False
    return _base_cnn

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // SEQ_LEN, 1)

    i = 0
    while len(frames) < SEQ_LEN:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frames.append(frame)
        i += step

    cap.release()

    while len(frames) < SEQ_LEN:
        frames.append(frames[-1])

    return np.array(frames)

def extract_features(video_path):
    frames = extract_frames(video_path)
    frames = preprocess_input(frames.astype(np.float32))

    cnn = get_resnet()
    feats = cnn.predict(frames, verbose=0)

    return feats.reshape(1, SEQ_LEN, -1)
