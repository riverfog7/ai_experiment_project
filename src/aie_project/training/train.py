import argparse
from pathlib import Path

from transformers import TrainingArguments, Trainer

from .configs import EfficientClassificationConfig
from .dataset_utils import easy_load
from .models import EfficientClassificationModel
from .metrics import compute_metrics


def train(
        data_loc: Path,
        train: bool = True,
        model_path: Path | str = "./models/trained_model",
        output_path: Path | str = "./train_results"
):
    output_path = Path(output_path)
    model_path = Path(model_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    dataset, label2id, id2label = easy_load(data_loc, include_all_columns=False)
    train_ds = dataset["train"]
    val_ds = dataset["validation"]

    if train:
        config = EfficientClassificationConfig(
            num_labels=len(label2id),
            backbone_name="mobilenetv3_large_100",
            pretrained=True,
            img_size=224,
            classifier_dropout=0.2,
        )
        model = EfficientClassificationModel(config)
    else:
        model = EfficientClassificationModel.from_pretrained(
            model_path.absolute().as_posix(),
            low_cpu_mem_usage=False,
        )

    args = TrainingArguments(
        output_dir=output_path.absolute().as_posix(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        per_device_train_batch_size=384,
        per_device_eval_batch_size=384,
        num_train_epochs=10,
        dataloader_num_workers=6,
        dataloader_prefetch_factor=4,
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
        compute_metrics=compute_metrics,
        tokenizer=None,
    )

    if train:
        trainer.train()
        trainer.save_model(model_path.absolute().as_posix())

    print("Evaluating the model...")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Efficient Classification Model")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("./datasets/recyclables_image_classification"),
        help="Path to the dataset location",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="If set, only evaluate the model without training",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("./models/trained_model"),
        help="Path to the pre-trained model for evaluation",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("./train_results"),
        help="Path to save evaluation results",
    )

    train(
        data_loc=parser.parse_args().data_path,
        train=not parser.parse_args().eval_only,
        model_path=parser.parse_args().model_path,
        output_path=parser.parse_args().output_path,
    )
