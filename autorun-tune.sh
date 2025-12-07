#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "$SCRIPT_DIR" || exit 1

bash -c "./configure.sh"
source "$HOME/.local/bin/env"

bash -c "./scripts/preload_cache_hf.sh"
bash -c "./scripts/tune.sh"
