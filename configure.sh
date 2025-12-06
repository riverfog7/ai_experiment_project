#!/usr/bin/env bash

# script for cloud environment configureation
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
LOCAL_VENV_DIR="$HOME/uv_venv"
export UV_CACHE_DIR="${HOME}/.uv_cache"

if [ "$(uname)" != "Darwin" ] && [ "$(uname)" != "Linux" ]; then
    echo "Unsupported OS: $(uname). This script supports only Linux and macOS."
    exit 1
fi

if [ "$(uname)" == "Linux" ]; then
  if command -v apt $> /dev/null; then
    if ! command -v sudo &> /dev/null; then
        echo "Installing sudo..."
        apt update && apt install -y sudo
        if $? -ne 0; then
            echo "Failed to install sudo. Please install it manually and re-run the script."
            exit 1
        fi
    fi
    sudo apt update && sudo apt install -y gh btop nvtop screen git jq aria2 curl wget unzip
  fi

  if [ "$(uname -m)" == "x86_64" ]; then
    pushd /tmp || exit 1
    echo "Downloading and installing AWS CLI v2 for x86_64..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    rm -rf awscliv2.zip aws
    popd || exit 1

  elif [ "$(uname -m)" == "aarch64" || "$(uname -m)" == "arm64" ]; then
    pushd /tmp || exit 1
    echo "Downloading and installing AWS CLI v2 for aarch64..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    rm -rf awscliv2.zip aws
    popd || exit 1

  else
    echo "Unsupported architecture: $(uname -m)."
    exit 1
  fi
fi

if [ "${GITHUB_TOKEN}" != "" ]; then
  echo "Authenticating gh with provided GITHUB_TOKEN..."
  echo "${GITHUB_TOKEN}" | gh auth login --with-token
  gh auth setup-git
  GH_USER=$(gh api -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" /user | jq -r .login)
  GH_EMAIL=$(gh api -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" /user/emails | jq -r ".[0].email")
  git config --global user.name "${GH_USER}"
  git config --global user.email "${GH_EMAIL}"
  echo "Git configured with user: ${GH_USER}, email: ${GH_EMAIL}"
fi

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

UV_EXTRA="--extra cpu"
DETECTED_CUDA=""

if command -v nvcc &> /dev/null; then
    DETECTED_CUDA=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
    echo "Detected NVCC: $DETECTED_CUDA"

elif command -v nvidia-smi &> /dev/null; then
    DETECTED_CUDA=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "Detected NVIDIA Driver (Max CUDA): $DETECTED_CUDA"
fi

if [[ -n "$DETECTED_CUDA" ]]; then
    CUDA_CLEAN=$(echo "$DETECTED_CUDA" | tr -d '.')

    if [[ "$CUDA_CLEAN" -ge 130 ]]; then
        UV_EXTRA="--extra cu130"
    elif [[ "$CUDA_CLEAN" -ge 128 ]]; then
        UV_EXTRA="--extra cu128"
    elif [[ "$CUDA_CLEAN" -ge 126 ]]; then
        UV_EXTRA="--extra cu126"
    else
        echo "Warning: CUDA version $DETECTED_CUDA is too old (<12.0). Defaulting to CPU."
    fi
else
    echo "No NVIDIA GPU detected. Defaulting to CPU."
fi

if [ ! -d "$LOCAL_VENV_DIR" ]; then
    echo "Creating local venv storage at $LOCAL_VENV_DIR"
    mkdir -p "$LOCAL_VENV_DIR"
else
    echo "Local venv storage already exists at $LOCAL_VENV_DIR"
fi

rm -rf "${SCRIPT_DIR}/.venv"
echo "Symlinking venv to local storage"
ln -s "$LOCAL_VENV_DIR" "${SCRIPT_DIR}/.venv"

echo "Running: uv sync $UV_EXTRA"
cd "$SCRIPT_DIR" || exit 1
uv sync $UV_EXTRA --frozen --dev

if [ "${WANDB_API_KEY}" != "" ]; then
  echo "Configuring Weights & Biases with provided WANDB_API_KEY..."
  uv run wandb login "${WANDB_API_KEY}"
  echo "export WANDB_PROJECT=ai-experiment-project" >> ~/.bashrc
fi
