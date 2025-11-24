from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


app = FastAPI(
    title="Iris Classifier API",
    description="Simple ML model served with FastAPI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (fine for learning)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



MODEL_PATH = Path("models/iris_model.joblib")
model = None
target_names: List[str] = []
feature_names: List[str] = []


@app.on_event("startup")
def load_model():
    global model, target_names, feature_names

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Make sure you ran train_model.py."
        )

    data = joblib.load(MODEL_PATH)
    model = data["model"]
    target_names = list(data["target_names"])
    feature_names = list(data["feature_names"])
    print("Model loaded successfully.")


@app.get("/")
def read_root():
    return {
        "message": "Iris Classifier API is running.",
        "model_loaded": model is not None,
        "target_names": target_names,
    }

@app.get("/ui", response_class=HTMLResponse)
def get_ui():
    html_path = Path("iris_ui.html")
    if not html_path.exists():
        return HTMLResponse("<h1>UI file not found</h1>", status_code=500)
    return html_path.read_text(encoding="utf-8")


@app.post("/predict")
def predict(iris: IrisFeatures):

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    features = [
        [
            iris.sepal_length,
            iris.sepal_width,
            iris.petal_length,
            iris.petal_width,
        ]
    ]

    try:
        pred_idx = model.predict(features)[0]
        pred_proba = model.predict_proba(features)[0].tolist()
        predicted_class = target_names[pred_idx]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "predicted_class": predicted_class,
        "predicted_class_index": int(pred_idx),
        "class_probabilities": {
            target_names[i]: pred_proba[i] for i in range(len(target_names))
        },
    }
