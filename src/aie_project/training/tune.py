import os
import tarfile
from pathlib import Path

import numpy as np
import optuna
import torch
import wandb
from optuna.artifacts import upload_artifact
from optuna.pruners import HyperbandPruner
from optuna.storages import RDBStorage, RetryFailedTrialCallback
from transformers import TrainingArguments

from .callbacks import DistributedOptunaCallback
from .custom_trainer import VisualizationTrainer
from .dataset_utils import easy_load, prune_smart_fast
from .metrics import compute_metrics
from .train_utils import model_factory
from .tune_utils import get_artifact_store, get_db_conn_str, get_study_name
from .utils import find_and_load_dotenv


def objective(trial: optuna.Trial):
    # set seed for reproducibility
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # easy load works with only cached dataset.
    dataset, material_label2id, material_id2label, transparency_label2id, transparency_id2label = \
        easy_load(
            "./datasets/recyclables_image_classification",
            include_all_columns=False,
            keep_in_memory=False,
            prune_train_set=True,
            prune_kwargs={
                "ratio": 0.1,
                "threshold": 5000,
                "seed": RANDOM_SEED,
            },
        )
    train_ds = dataset["train"]
    val_ds = dataset["validation"]
    model = model_factory(
        num_classes_1=len(material_label2id),
        num_classes_2=len(transparency_label2id),
        label2id_1=material_label2id,
        id2label_1=material_id2label,
        label2id_2=transparency_label2id,
        id2label_2=transparency_id2label,
    )
    metric = "eval_combined_f1_macro"
    optuna_callback = DistributedOptunaCallback(trial, objective_metric=metric)

    train_dir = Path("training_results") / f"{trial.study.study_name}_{trial.number}"
    train_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path("models") / f"{trial.study.study_name}_{trial.number}"
    save_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_NAME"] = f"{trial.study.study_name}_{trial.number}"

    optim_type = trial.suggest_categorical("optim", ["adamw_torch", "sgd", "adagrad"])

    # dataset is shuffled by default (Not IterableDataset)
    # see https://discuss.huggingface.co/t/does-masked-language-model-training-script-does-random-shuffle-on-the-dataset/11197/3
    args = TrainingArguments(
        seed=RANDOM_SEED,
        output_dir=train_dir.absolute().as_posix(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        lr_scheduler_type="cosine",
        warmup_ratio=0.2, # fixed to conserve compute
        load_best_model_at_end=True,
        metric_for_best_model=metric,
        greater_is_better=True,
        # for 24GB vram GPU
        # 24GB 368 OK -> 48GB 768 OK
        per_device_train_batch_size=trial.suggest_categorical("per_device_train_batch_size", [64, 128, 256, 384]),
        per_device_eval_batch_size=384, # fixed because it doesn't affect training
        num_train_epochs=trial.suggest_int("num_train_epochs", 2, 16, step=2),
        dataloader_num_workers=8,
        dataloader_prefetch_factor=2,
        optim=optim_type,
        push_to_hub=False,
        remove_unused_columns=False,
        report_to=["wandb"], # will report to optuna in a custom callback
        label_names=["labels_1", "labels_2"],
    )

    trainer = VisualizationTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        tokenizer=None,
        callbacks=[optuna_callback],
    )
    trainer.train()
    eval_result = trainer.evaluate()
    for key, value in eval_result.items():
        trial.set_user_attr(key, value)

    trainer.save_model(save_dir.absolute().as_posix())

    tarball_path = Path("models") / f"{trial.study.study_name}_{trial.number}.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(save_dir, arcname=save_dir.name)

    upload_artifact(
        artifact_store=get_artifact_store(),
        file_path=tarball_path.absolute().as_posix(),
        study_or_trial=trial,
    )
    wandb.finish()

    return eval_result[metric]

def get_study() -> optuna.Study:
    find_and_load_dotenv()

    retry_cb = RetryFailedTrialCallback(max_retry=3)
    storage = RDBStorage(
        url=get_db_conn_str(),
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=retry_cb,
    )
    study_name = get_study_name()

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        pruner=HyperbandPruner(
            min_resource=1,
            max_resource=16,
            reduction_factor=3
        )
    )
    return study

def run_study():
    study = get_study()
    study.optimize(objective, n_trials=None)

if __name__ == "__main__":
    run_study()
