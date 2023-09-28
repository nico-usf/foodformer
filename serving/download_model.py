import os
from pathlib import Path

import wandb

if not os.environ.get("WANDB_API_KEY"):
    raise ValueError(
        "You must set the WANDB_API_KEY environment variable " "to download the model."
    )

wandb_team = "mlops-usf"
wandb_project = "Foodformer"
wandb_model = "vit:v0"
wandb_model_path = f"{wandb_team}/{wandb_project}/{wandb_model}"

wandb.init()

current_folder = Path(__file__).parent
print(f"Folder: {current_folder}")
path = ### EXERCISE: download model from Weights and Biasesto local path ###
print(f"Model downloaded to: {path}")
