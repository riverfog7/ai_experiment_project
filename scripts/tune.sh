#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." &>/dev/null && pwd)
export POSTGRES_USER=${POSTGRES_USER:-"optuna"}
export POSTGRES_HOST=${POSTGRES_HOST:-"riverfog7.com"}
export POSTGRES_PORT=${POSTGRES_PORT:-15432}
export POSTGRES_DB=${POSTGRES_DB:-"optuna"}
export OPTUNA_STUDY_NAME=${OPTUNA_STUDY_NAME:-"default"}
export OPTUNA_S3_BUCKET=${OPTUNA_S3_BUCKET:-"optuna-study"}

cd "${PROJECT_ROOT}"
export WANDB_PROJECT="ai-study-project"
export WANDB_WATCH=1
uv run -m aie_project.training.tune
