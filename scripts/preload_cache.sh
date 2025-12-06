#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." &>/dev/null && pwd)
USE_ARIA2=true
DATASET_DIR=${1:-"$PROJECT_ROOT/datasets"}

TARBALL_URL="https://riverfog7.com/datasets/recyclables_image_classification_cache.tar"
TARBALL_MD5=""

if ! command -v aria2c &> /dev/null; then
    echo "Error: aria2c is not installed. Falling back to curl".
    USE_ARIA2=false
    if ! command -v curl &> /dev/null; then
        echo "Error: curl is also not installed. Please install either aria2c or curl to proceed."
        exit 1
    fi
fi

if ! command -v md5sum &> /dev/null; then
    echo "Error: md5sum is not installed. Please install md5sum to proceed."
    exit 1
fi

mkdir -p "$DATASET_DIR"
echo "Preloading dataset into $DATASET_DIR..."
if [ "$USE_ARIA2" = true ]; then
    echo "Downloading dataset using aria2c..."
    aria2c -x 8 -s 8 -d "$DATASET_DIR" -o recyclables_image_classification_cache.tar "$TARBALL_URL"
else
    echo "Downloading dataset using curl..."
    curl -L "$TARBALL_URL" -o "$DATASET_DIR/recyclables_image_classification_cache.tar"
fi

if [ $? -ne 0 ]; then
    echo "Error: Failed to download dataset from $TARBALL_URL"
    exit 1
fi
echo "Download completed."

echo "Verifying checksum..."
DOWNLOADED_MD5=$(md5sum "$DATASET_DIR/recyclables_image_classification_cache.tar" | awk '{ print $1 }')
if [ "$DOWNLOADED_MD5" != "$TARBALL_MD5" ]; then
    echo "Error: Checksum verification failed. Expected $TARBALL_MD5 but got $DOWNLOADED_MD5"
    exit 1
fi
echo "Checksum verified."

echo "Extracting dataset..."
tar -xf "$DATASET_DIR/recyclables_image_classification_cache.tar" -C "$DATASET_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Failed to extract dataset."
    exit 1
fi
rm "$DATASET_DIR/recyclables_image_classification_cache.tar"
echo "Dataset preloaded successfully into $DATASET_DIR."
