import optuna
from pathlib import Path
from transformers import Trainer, TrainingArguments

from .dataset_utils import easy_load
from .metrics import compute_metrics
from .train_utils import model_factory
from .tune_utils import get_study_name, get_db_conn_str




def objective(trial: optuna.Trial):
    # easy load works with only cached dataset.
    dataset, material_label2id, material_id2label, transparency_label2id, transparency_id2label = \
        easy_load("./datasets/recyclables_image_classification", include_all_columns=False)
    train_ds = dataset["train"]
    val_ds = dataset["validation"]
    model = model_factory(num_classes_1=len(material_label2id), num_classes_2=len(transparency_label2id))

    train_dir = Path("training_results") / f"{trial.study.study_name}_{trial.number}"
    train_dir.mkdir(parents=True, exist_ok=True)

    optim_type = trial.suggest_categorical("optim", ["sgd", "adamw", "adam", "adamax"])

    args = TrainingArguments(
        output_dir=train_dir.absolute().as_posix(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        lr_scheduler_type="cosine",
        warmup_ratio=0.2, # fixed to conserve compute
        load_best_model_at_end=True,
        per_device_train_batch_size=trial.suggest_categorical("per_device_train_batch_size", [64, 128, 256, 384, 512]),
        per_device_eval_batch_size=512, # fixed because it doesn't affect training
        num_train_epochs=trial.suggest_int("num_train_epochs", 2, 16, step=2),
        dataloader_num_workers=6,
        dataloader_prefetch_factor=4,
        push_to_hub=False,
        remove_unused_columns=False,
        report_to=["wandb"], # will report to optuna in a custom callback
        label_names=["labels_1", "labels_2"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        tokenizer=None,
    )
