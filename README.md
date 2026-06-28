#  AI Video Detection using CNN-LSTM

###  Live Demo
** https://ai-video-detection-model-7uwcszkr5zetudqpyh5hzv.streamlit.app/**


---

## Dashboard

![Dashboard](assets/dashboard.png)

---

# Overview

This project presents a **CNN-LSTM based AI Video Detection pipeline** developed for educational and research purposes. The system analyzes uploaded videos by extracting representative frames, generating deep visual features using **ResNet50**, and performing temporal sequence classification with a trained **CNN-LSTM** model.

The project demonstrates a complete deep learning workflow including preprocessing, feature extraction, sequence modeling, prediction, and deployment through a modern Streamlit interface.

> **Note**
>
> This project is intended as a **study and experimentation pipeline** for AI-generated video analysis.
>
> Modern AI-generated videos are becoming increasingly realistic, and reliable detection remains an active research problem. Therefore, predictions produced by this model should be considered experimental rather than definitive.

---

# Features

-  Upload video directly from browser
-  CNN-LSTM based prediction
-  ResNet50 feature extraction
-  Confidence score visualization
-  AI vs Real probability comparison
-  Modern Streamlit dashboard
-  Automatic preprocessing pipeline

---

# Pipeline

```text
Input Video
      │
      ▼
Frame Extraction
      │
      ▼
ResNet50 Feature Extraction
      │
      ▼
2048-D Feature Sequence
      │
      ▼
CNN-LSTM Model
      │
      ▼
Prediction
      │
      ▼
Interactive Dashboard
```

---

# Model Architecture

- **Feature Extractor**
  - ResNet50 (ImageNet pretrained)
  - Global Average Pooling
  - 2048-dimensional feature vectors

- **Sequence Model**
  - CNN-LSTM
  - Temporal video representation
  - Binary classification

---

# Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- OpenCV
- NumPy
- Pillow

---

# Project Structure

```
AI-Video-Detection/
│
├── app.py
├── style.css
├── requirements.txt
├── runtime.txt
├── packages.txt
│
├── services/
│   ├── preprocess.py
│   ├── predictor.py
│   └── model_loader.py
│
├── models/
│   ├── appearance_lstm_clean_v2_80.keras
│   ├── resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
│   └── model_info.json
│
├── assets/
│   └── dashboard.png
│
└── uploads/
```

---

# Installation

```bash
git clone https://github.com/chandu2006-git/ai-video-detection-model.git

cd ai-video-detection-model

pip install -r requirements.txt

streamlit run app.py
```

---

# Disclaimer

This project was developed for **educational, academic, and research purposes**.

Although the system demonstrates a complete AI video analysis workflow, AI-generated videos are evolving rapidly. Detection performance depends on dataset quality, model generalization, and video characteristics. Results should therefore be interpreted as **experimental predictions** and should not be used as the sole basis for determining the authenticity of media.

---

# Future Improvements

- Vision Transformer (ViT) feature extraction
- TimeSformer / VideoMAE integration
- Attention-based temporal modeling
- Explainable AI (Grad-CAM)
- Multi-class manipulation detection
- Larger benchmark datasets
- Model quantization and optimization

---

# Author

**Chandra Sekhar**

B.Tech — Artificial Intelligence & Data Science

Research & Deep Learning Enthusiast

---

⭐ If you found this project interesting, consider giving it a star!
