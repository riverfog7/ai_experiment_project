import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import TrainingArguments

from .custom_trainer import VisualizationTrainer
from .dataset_utils import easy_load
from .metrics import compute_metrics
from .models import EfficientMultiTaskClassificationModel
from .train_utils import model_factory


def train(
        data_loc: Path,
        train: bool = True,
        model_path: Path | str = "./models/trained_model",
        output_path: Path | str = "./train_results",
        random_seed: int = 42,
):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    output_path = Path(output_path)
    model_path = Path(model_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    dataset, material_label2id, material_id2label, transparency_label2id, transparency_id2label = \
        easy_load("./datasets/recyclables_image_classification", include_all_columns=False, keep_in_memory=False)
    train_ds = dataset["train"]
    val_ds = dataset["validation"]

    if train:
        model = model_factory(
            num_classes_1=len(material_label2id),
            num_classes_2=len(transparency_label2id),
            label2id_1=material_label2id,
            id2label_1=material_id2label,
            label2id_2=transparency_label2id,
            id2label_2=transparency_id2label,
        )
    else:
        model = EfficientMultiTaskClassificationModel.from_pretrained(
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
        per_device_train_batch_size=512,
        per_device_eval_batch_size=2048,
        num_train_epochs=10,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=4,
        dataloader_pin_memory=True,
        weight_decay=0.01,
        push_to_hub=False,
        remove_unused_columns=False,
        report_to=["wandb"],
        label_names=["labels_1", "labels_2"],
    )

    trainer = VisualizationTrainer(
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
    outputs = trainer.predict(val_ds)
    for key, value in outputs.metrics.items():
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
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    train(
        data_loc=parser.parse_args().data_path,
        train=not parser.parse_args().eval_only,
        model_path=parser.parse_args().model_path,
        output_path=parser.parse_args().output_path,
        random_seed=parser.parse_args().random_seed,
    )
