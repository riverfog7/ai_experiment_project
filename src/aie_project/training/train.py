import argparse
from pathlib import Path

from transformers import TrainingArguments, Trainer

from .configs import EfficientClassificationConfig
from .dataset_utils import easy_load
from .models import EfficientClassificationModel


def train(data_loc: Path):
    dataset, label2id, id2label = easy_load("./datasets/recyclables_image_classification", include_all_columns=False)

    train_ds = dataset["train"]
    val_ds = dataset["validation"]
    config = EfficientClassificationConfig(
        num_labels=len(label2id),
        backbone_name="mobilenetv3_large_100",
        pretrained=True,
        img_size=224,
        classifier_dropout=0.2,
    )

    model = EfficientClassificationModel(config)

    args = TrainingArguments(
        output_dir="./train_results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        num_train_epochs=10,
        weight_decay=0.01,
        push_to_hub=False,
        remove_unused_columns=False,
        report_to=["wandb"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=None,
    )

    trainer.train()
    trainer.save_model("./models/trained_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Efficient Classification Model")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("./datasets/recyclables_image_classification"),
        help="Path to the dataset location",
    )
    train(parser.parse_args().data_path)
