import os
from functools import partial
from io import BytesIO
from pathlib import Path

import torch
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.security import OAuth2PasswordBearer
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from torch.nn.functional import softmax
from transformers import ViTImageProcessor


class ClassPredictions(BaseModel):
    predictions: dict[str, float]


# Deactivating docs
# see https://github.com/tiangolo/fastapi/issues/364#issuecomment-789711477
# for how to protect /docs with a username/password combo
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
logger.info("Starting API, model version v0...")

api_keys = [os.environ["FOODFORMER_API_KEY"]]
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def api_key_auth(api_key: str = Depends(oauth2_scheme)) -> None:
    if api_key not in api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Forbidden"
        )


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


@app.get("/", dependencies=[Depends(api_key_auth)])
def get_root() -> dict:
    logger.info("Received request on the root endpoint")
    return {"status": "ok"}


@app.post(
    "/predict", response_model=ClassPredictions, dependencies=[Depends(api_key_auth)]
)
async def predict_api(file: UploadFile = File(...)) -> ClassPredictions:
    logger.info(f"Predict endpoint received image {file.filename}")

    file_extension = file.filename.split(".")[-1]
    valid_extensions = ("jpg", "jpeg", "png")
    if file_extension not in valid_extensions:
        raise TypeError(
            f"File extension for {file.filename} should be one of {valid_extensions}"
        )

    image = read_imagefile(await file.read())
    x = preprocess_image(image)
    predictions = predict(x)

    logger.info(f"Predictions for {file.filename}: {predictions}")
    return ClassPredictions(predictions=predictions)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
