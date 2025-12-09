import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import TrainingArguments, EarlyStoppingCallback

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
        easy_load(
            data_loc.absolute().as_posix(),
            include_all_columns=False,
            keep_in_memory=False,
            prune_train_set=False,
        )
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

    metric = "eval_combined_f1_macro"
    batch_size = 128
    # assume GPU count is 1 here
    epoch_steps = len(train_ds) // batch_size
    eval_steps = epoch_steps // 4  # evaluate 4 times per epoch
    args = TrainingArguments(
        run_name=f"ai-experiment-project-final",
        seed=random_seed,
        output_dir=output_path.absolute().as_posix(),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=0.000766037529647225,
        lr_scheduler_type="cosine",
        warmup_ratio=0.08,  # fixed to conserve compute
        weight_decay=0.07326981828565753,
        load_best_model_at_end=True,
        metric_for_best_model=metric,
        greater_is_better=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=384,  # fixed because it doesn't affect training
        num_train_epochs=16,
        dataloader_num_workers=16,
        dataloader_prefetch_factor=2,
        label_smoothing_factor=0.1,
        max_grad_norm=0.5, # stabilize training
        optim="adamw_torch",
        bf16=False, # Full precision for main training
        fp16=False,
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
        # stop with patience of 1 epoch (4 eval per epoch)
        callbacks=[EarlyStoppingCallback(early_stopping_patience=32)],
    )

    if train:
        trainer.train()
        trainer.save_model(model_path.absolute().as_posix())

    print("Evaluating the model...")
    # use evaluate. confusion matrix is handled in custom trainer
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
