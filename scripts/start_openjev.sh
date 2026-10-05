#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -x .openjev-venv/bin/python || ! -f models/openjev-mlx-4bit/model.safetensors.index.json || ! -f vendor/openjev/helper/shim_mlx.py ]]; then
    printf '%s\n' 'OpenJev is not installed. Run bash scripts/setup_openjev.sh first.' >&2
    exit 1
fi

export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZER="$PWD/models/openjev-mlx-4bit"
export SHIM_MODEL="$TOKENIZER"
export READOUT_T=0.85
export READOUT_NOUL_T=1.829074
export READOUT_NOUL_BIAS=0
export READOUT_TARGETED=1
export READOUT_PERMS=1
export READOUT_INSTR_STYLE=pyrepr
export SHIM_STAGGER=1
export PYTHONUNBUFFERED=1

printf '%s\n' 'Loading local OpenJev. Requests stay on this machine.'
exec .openjev-venv/bin/python vendor/openjev/helper/shim_mlx.py \
    --helper vendor/openjev/helper/shim.py --model "$TOKENIZER" \
    --host 127.0.0.1 --port 3000
