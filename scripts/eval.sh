#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." &>/dev/null && pwd)
MODEL_DIR="${PROJECT_ROOT}/models"

if [ ! -d "${MODEL_DIR}/final_model" ]; then
  echo "Model directory '${MODEL_DIR}/final_model' does not exist. Downloading the model"
  curl https://web.aws.riverfog7.com/files/ai_experiment_project/final_model.tar.gz | tar -xzf - -C "${MODEL_DIR}"
fi

cd "${PROJECT_ROOT}"
uv run -m aie_project.training.train \
  --data-path ./datasets/recyclables_image_classification \
  --model-path "${MODEL_DIR}/final_model" \
  --output-path ./train_results \
  --eval-only
