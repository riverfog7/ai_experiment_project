#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." &>/dev/null && pwd)

cd "${PROJECT_ROOT}"
export WANDB_PROJECT="ai-study-project"
uv run -m aie_project.training.train --data-path ./datasets/recyclables_image_classification
