#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." &>/dev/null && pwd)
DATASET_DIR=${1:-"$PROJECT_ROOT/datasets"}
LABELS_URL="https://riverfog7.com/datasets/recyclables_image_classification_labels.tar.gz"

mkdir -p "$DATASET_DIR/cache"
echo "Preloading dataset into $DATASET_DIR..."
cd "$DATASET_DIR" || exit 1

uv tool run --with hf_transfer hf download \
  riverfog7/ai-experiment-project \
  --repo-type dataset \
  --local-dir "$DATASET_DIR/cache/easy_load_cache" \
  --max-workers 8

cd "$DATASET_DIR/datasets/cache" || exit 1
echo "Downloading labels using curl..."
curl -L "$LABELS_URL" | tar -xzf -
