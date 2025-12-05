#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." &>/dev/null && pwd)

cd "${PROJECT_ROOT}"
uv run -m aie_project.training.train \
  --data-path ./datasets/recyclables_image_classification \
  --model-path ./models/trained_model \
  --output-path ./train_results \
  --eval-only
