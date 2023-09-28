import json
import sys
from functools import partial
from io import BytesIO
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from torch.nn.functional import softmax
from transformers import ViTImageProcessor


class ClassPredictions(BaseModel):
    predictions: dict[str, float]


app = FastAPI()

logger.remove()
logger.add(sys.stderr, level="INFO", format="{message}", serialize=False)
logger.info("Starting API, model version v0...")

model_name_or_path = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTImageProcessor.from_pretrained(model_name_or_path)
preprocessor = partial(feature_extractor, return_tensors="pt")


def preprocess_image(image: Image.Image) -> torch.tensor:
    return preprocessor([image])["pixel_values"]


def read_imagefile(file: bytes) -> Image.Image:
    return Image.open(BytesIO(file))


current_folder = Path(__file__).parent

MODEL_PATH = current_folder / "model.ckpt"


def load_model(model_path: str | Path = MODEL_PATH) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model = checkpoint["hyper_parameters"]["model"]
    labels = checkpoint["hyper_parameters"]["label_names"]
    model.eval()  # To set up inference (disable dropout, layernorm, etc.)
    return model, labels


model, labels = load_model()


def predict(x: torch.tensor, labels: list = labels) -> dict:
    logits = model(x).logits
    probas = softmax(logits, dim=1)

    values, indices = torch.topk(probas[0], 5)
    return_dict = {labels[int(i)]: float(v) for i, v in zip(indices, values)}
    return return_dict


@app.get("/")
def get_root() -> dict:
    logger.info("Received request on the root endpoint")
    return {"status": "ok"}


@app.post("/predict", response_model=ClassPredictions)
async def predict_api(file: UploadFile = File(...)) -> ClassPredictions:
    logger.info(f"Predict endpoint received image {file.filename}")

    file_extension = file.filename.split(".")[-1]
    valid_extensions = ("jpg", "jpeg", "png")
    if file_extension not in valid_extensions:
        raise TypeError(
            f"File extension for {file.filename} should be one of {valid_extensions}"
        )
    ### EXERCISE: read the image from the input `file` object,
    ### then preprocess the input and compute model predictions ###

    log = {
        "message": f"Predictions for {file.filename}: {predictions}",
        "top_class": list(predictions.keys())[0],
        "score": list(predictions.values())[0],
    }
    logger.info(json.dumps(log))
    return ClassPredictions(predictions=predictions)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
