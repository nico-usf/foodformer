import argparse
import logging
import os
import sys
from functools import partial

import pytorch_lightning as pl
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy
from torchvision.datasets import Food101
from transformers import ViTForImageClassification, ViTImageProcessor


class ViTDataset(Dataset):
    """Package images pixel values and labels into a dictionary."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        """In PyTorch datasets have to override the length method."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        """This method defines how to feed the data during model training."""
        inputs_and_labels = dict()
        inputs_and_labels["pixel_values"] = self.dataset[index][0]["pixel_values"][0]
        inputs_and_labels["labels"] = self.dataset[index][1]
        return inputs_and_labels


class LightningVisionTransformer(pl.LightningModule):
    """Main model object: contains the model, defines how
    to run a forward pass, what the loss is, and the optimizer."""

    def __init__(self, model, label_names, learning_rate=2e-4):
        super().__init__()
        self.model = model
        self.label_names = label_names
        self.learning_rate = learning_rate

    def forward(self, batch):
        pixel_values = batch["pixel_values"]
        logits = self.model(pixel_values).logits
        return logits

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        outputs = self.forward(batch)
        loss = cross_entropy(outputs, labels)
        self.log(
            "train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        outputs = self.forward(batch)
        loss = cross_entropy(outputs, labels)
        self.log(
            "val_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )

        preds = outputs.argmax(dim=1)
        acc = accuracy(
            preds, labels, task="multiclass", num_classes=len(self.label_names)
        )
        self.log(
            "accuracy", acc, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        return loss

    def predict_step(self, pixel_values):
        logits = self.model(pixel_values).logits
        predicted_label = logits.argmax()
        return self.label_names[predicted_label]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument(
        "--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument(
        "--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"]
    )

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    model_name_or_path = "google/vit-base-patch16-224-in21k"
    feature_extractor = ViTImageProcessor.from_pretrained(model_name_or_path)

    preprocessor = partial(feature_extractor, return_tensors="pt")
    print(
        f'{os.environ["SM_CHANNEL_TRAIN"]} content: {os.listdir(os.environ["SM_CHANNEL_TRAIN"])}'
    )
    train_ds = Food101(
        root=os.environ["SM_CHANNEL_TRAIN"], split="train", transform=preprocessor
    )
    test_ds = Food101(
        root=os.environ["SM_CHANNEL_TRAIN"], split="test", transform=preprocessor
    )

    labels = train_ds.classes

    train_dataset = ViTDataset(train_ds)
    test_dataset = ViTDataset(test_ds)

    # Create a dataloader object to chunk the datasets into batches
    training_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=0
    )
    testing_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=0
    )
    # Download pre-trained model from the HuggingFace model hub
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
    )

    # Create the Pytorch Lightning model
    lightning_vit = LightningVisionTransformer(model, labels)

    trainer = pl.Trainer(
        # max_steps=100, # For debug, comment for real training
        default_root_dir="/opt/ml/checkpoints",
        accelerator="gpu",
        max_epochs=1,
        devices=-1,
        strategy="ddp",
        enable_checkpointing=True,
        # logger=logger
    )

    # Run evaluation before training
    trainer.validate(lightning_vit, testing_loader)

    # Train the model
    trainer.fit(lightning_vit, training_loader, testing_loader)
