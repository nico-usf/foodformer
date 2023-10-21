# mypy: ignore-errors
import os

from datasets import load_dataset

dataset = load_dataset("Matthijs/snacks")

FOLDER = "./load_test_data"
os.makedirs(FOLDER, exist_ok=True)


def save_image(sample, folder=FOLDER):
    image_object = sample["image"]
    basename = os.path.basename(image_object.filename)
    image_object.save(f"{folder}/{basename}")


dataset["test"].map(save_image)
