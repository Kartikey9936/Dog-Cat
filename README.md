# 🐾 Dog & Cat Classifier



## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Workflow](#-workflow)
- [Problems Faced During Building](#-problems-faced-during-building)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact & Acknowledgements](#-contact--acknowledgements)

---

## ❓ Problem Statement

### What real problem does this solve?

Manually identifying whether an image contains a dog or a cat is trivial for humans — but for automated systems processing thousands of images (shelter databases, pet adoption platforms, veterinary record systems, content moderation pipelines), it becomes a bottleneck.

### Who faces this problem?

- **Animal shelters** that need to auto-tag uploaded pet photos
- **Pet adoption platforms** requiring consistent image categorization
- **Developers and ML learners** building their first computer vision pipeline
- **Content platforms** filtering or categorizing user-uploaded pet images at scale

### Why existing solutions are not enough?

Most off-the-shelf solutions are either:
- Too heavy (full-blown cloud vision APIs with pricing per call)
- Too opaque (black-box services with no control over the model)
- Too complex for small teams to self-host with a clean UI

This project gives you a **fully self-hosted, transparent, open-source classifier** with both an API and a user-friendly web interface — no cloud billing required.

---

## 💡 Solution Overview

### How does this project solve the problem?

A Convolutional Neural Network (CNN) trained on labeled dog and cat images learns spatial features — edges, textures, shapes — and maps them to a binary output (dog / cat). The trained model is served via a **FastAPI** backend, and users interact through a clean **Streamlit** web interface.

### Key insight or approach taken

Rather than using a massive pre-trained model like ResNet or VGG for a simple binary task, this project trains a **compact custom CNN** using Keras — keeping inference fast, the model small, and the codebase easy to understand, extend, and deploy anywhere.

---

## ✨ Features

| Feature | Status |
|---|---|
| 🖼️ Upload any JPG/PNG image via web UI | ✅ Completed |
| 🤖 CNN model classifies dog vs cat | ✅ Completed |
| ⚡ FastAPI REST endpoint for predictions | ✅ Completed |
| 🌐 Streamlit web interface | ✅ Completed |
| 📊 Confidence score displayed with result | ✅ Completed |
| 🔄 Real-time inference (< 1 second) | ✅ Completed |
| 📱 Responsive UI layout | ✅ Completed |
| 🐳 Docker support for easy deployment | 🚧 Upcoming |
| 📈 Model retraining via UI | 🚧 Upcoming |
| 🗂️ Batch image upload & prediction | 🚧 Upcoming |
| 🔐 API key authentication | 🚧 Upcoming |
| 📉 Prediction history dashboard | 🚧 Upcoming |

---

## 🛠️ Tech Stack

### Overview

| Layer | Technology | Why chosen |
|---|---|---|
| **ML Model** | Keras + TensorFlow | High-level API, fast prototyping, mature CNN support |
| **Model Architecture** | Custom CNN | Lightweight, interpretable, no over-engineering for binary task |
| **Backend API** | FastAPI | Async-native, auto-generates Swagger docs, type-safe |
| **Frontend UI** | Streamlit | Python-native UI, zero JavaScript needed, perfect for ML demos |
| **Image Processing** | Pillow + NumPy | Standard, battle-tested preprocessing pipeline |
| **Runtime** | Python 3.9+ | Universal ML ecosystem support |

### Why not Flask or Django?

FastAPI gives async performance and automatic OpenAPI documentation out of the box — no extra boilerplate to serve a `/predict` endpoint correctly.

### Why Streamlit over React?

For an ML demo, Streamlit lets you build a working UI entirely in Python in minutes. The team stays focused on the model, not the frontend plumbing.

---

## 🏗️ System Architecture

```
                        ┌─────────────────────────────────┐
                        │         User's Browser          │
                        └──────────────┬──────────────────┘
                                       │ HTTP (image upload)
                        ┌──────────────▼──────────────────┐
                        │       Streamlit Frontend         │
                        │       (localhost:8501)           │
                        │  - File uploader widget          │
                        │  - Displays prediction result    │
                        └──────────────┬──────────────────┘
                                       │ POST /predict (multipart/form-data)
                        ┌──────────────▼──────────────────┐
                        │        FastAPI Backend           │
                        │       (localhost:8000)           │
                        │  - Receives image bytes          │
                        │  - Validates input               │
                        │  - Calls model inference         │
                        └──────────────┬──────────────────┘
                                       │ numpy array (preprocessed)
                        ┌──────────────▼──────────────────┐
                        │         CNN Model (Keras)        │
                        │       model.h5 / model.keras     │
                        │  - Resize → Normalize → Predict  │
                        │  - Returns: class + confidence   │
                        └─────────────────────────────────┘

  DATA FLOW:
  Image File → Read Bytes → Resize (150×150) → Normalize (÷255)
             → CNN Forward Pass → Sigmoid Output → Threshold (0.5)
             → {"label": "Dog", "confidence": 0.94}
```

---

## 📁 Project Structure

```
dog-cat-classifier/
│
├── backend/                        # FastAPI application
│   ├── main.py                     # App entry point, /predict route
│   ├── model_loader.py             # Loads Keras model at startup
│   └── utils.py                    # Image preprocessing helpers
│
├── frontend/                       # Streamlit application
│   └── app.py                      # UI: uploader, API call, result display
│
├── model/                          # ML training pipeline
│   ├── train.py                    # CNN architecture + training script
│   ├── evaluate.py                 # Accuracy, confusion matrix evaluation
│   └── saved_model/
│       └── dog_cat_model.keras     # Trained model weights (git-ignored if large)
│
├── data/                           # Dataset (not committed to git)
│   ├── train/
│   │   ├── dogs/                   # Training dog images
│   │   └── cats/                   # Training cat images
│   └── validation/
│       ├── dogs/                   # Validation dog images
│       └── cats/                   # Validation cat images
│
├── notebooks/                      # Jupyter exploration notebooks
│   └── exploration.ipynb           # EDA and model experimentation
│
├── tests/                          # Unit tests
│   ├── test_api.py                 # FastAPI endpoint tests
│   └── test_model.py               # Model output shape/type tests
│
├── .env.example                    # Environment variable template
├── .gitignore                      # Ignores data/, __pycache__, .env
├── requirements.txt                # All Python dependencies
└── README.md                       # This file
```

---

## 🔧 Prerequisites

Make sure you have the following installed before starting:

| Software | Minimum Version | Check command |
|---|---|---|
| Python | 3.9 | `python --version` |
| pip | 21.0 | `pip --version` |
| Git | 2.x | `git --version` |
| (Optional) CUDA | 11.2 | For GPU-accelerated training |

> **Note:** A GPU is not required. The model trains and infers fine on CPU for this scale of task.

---

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/dog-cat-classifier.git
cd dog-cat-classifier
```

### 2. Create and activate a virtual environment

```bash
# Create
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
# Edit .env with your values (see Configuration section)
```

### 5. Prepare the dataset

Download the [Kaggle Dogs vs Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data) and place it in the `data/` folder following the structure shown above.

### 6. Train the model

```bash
python model/train.py
```

This saves the trained model to `model/saved_model/dog_cat_model.keras`.

### 7. Start the Backend (FastAPI)

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 8. Start the Frontend (Streamlit)

Open a new terminal (keep the backend running):

```bash
streamlit run frontend/app.py
```

Visit `http://localhost:8501` in your browser.

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and fill in the values:

```env
# ─── Model ───────────────────────────────────────────
MODEL_PATH=model/saved_model/dog_cat_model.keras  # Absolute or relative path to trained model
IMAGE_SIZE=150                                     # Input image size (width & height in pixels)

# ─── API ─────────────────────────────────────────────
API_HOST=0.0.0.0                                   # Host for FastAPI server
API_PORT=8000                                      # Port for FastAPI server
API_BASE_URL=http://localhost:8000                 # URL Streamlit uses to reach the API

# ─── Training ────────────────────────────────────────
EPOCHS=25                                          # Training epochs
BATCH_SIZE=32                                      # Batch size during training
LEARNING_RATE=0.001                                # Adam optimizer learning rate
```

---

## 📖 Usage

### Run the full application

```bash
# Terminal 1 — Backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — Frontend
streamlit run frontend/app.py
```

### API Endpoints

**`POST /predict`** — Classify an uploaded image

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@/path/to/your/image.jpg"
```

**Response:**

```json
{
  "label": "Dog",
  "confidence": 0.9423,
  "prediction_time_ms": 47
}
```

**`GET /health`** — Check if the API and model are loaded

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Swagger UI

FastAPI auto-generates interactive API docs at:
`http://localhost:8000/docs`

```
┌──────────────────────────────────────────────────┐
│          [ App Screenshot Placeholder ]           │
│   Streamlit UI with image uploader + result card  │
└──────────────────────────────────────────────────┘
```

---

## 🔄 Workflow

Here's exactly what happens from the moment you upload an image:

```
Step 1 — User uploads image via Streamlit UI
         ↓
Step 2 — Streamlit sends POST /predict (multipart/form-data) to FastAPI
         ↓
Step 3 — FastAPI receives image bytes, validates file type (jpg/png)
         ↓
Step 4 — Image is opened with Pillow, resized to 150×150 pixels
         ↓
Step 5 — Pixel values normalized: divide by 255.0 → range [0.0, 1.0]
         ↓
Step 6 — Array reshaped to (1, 150, 150, 3) to match model input shape
         ↓
Step 7 — CNN forward pass: Conv → ReLU → Pool (×3) → Flatten → Dense → Sigmoid
         ↓
Step 8 — Sigmoid output: value close to 0 = Cat, close to 1 = Dog
         ↓
Step 9 — Threshold at 0.5 → final label assigned
         ↓
Step 10 — JSON response returned: {"label": "Dog", "confidence": 0.94}
         ↓
Step 11 — Streamlit displays result with confidence bar to the user
```

---


### How to contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run tests: `pytest tests/`
5. Commit with a clear message: `git commit -m "feat: add batch prediction endpoint"`
6. Push and open a Pull Request

### Code style guidelines

- Follow **PEP 8** for all Python code
- Use **type hints** on all function signatures
- Write **docstrings** for every function and class
- Keep functions focused — one responsibility per function
- All new features must include a corresponding test in `tests/`

```bash
# Format before committing
pip install black isort
black .
isort .
```

---

## 📄 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute this project for personal or commercial purposes with attribution.

See [`LICENSE`](./LICENSE) for the full text.

---



### Acknowledgements & References

| Resource | Purpose |
|---|---|
| [Kaggle Dogs vs Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats) | Training data |
| [Keras Documentation](https://keras.io/) | CNN architecture reference |
| [FastAPI Documentation](https://fastapi.tiangolo.com/) | API server setup |
| [Streamlit Documentation](https://docs.streamlit.io/) | Frontend UI reference |
| [Deep Learning with Python — François Chollet](https://www.manning.com/books/deep-learning-with-python) | CNN design principles |

---

<p align="center">
  Made with ❤️ and a lot of images of dogs and cats
</p>
