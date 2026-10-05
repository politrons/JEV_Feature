#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    printf '%s\n' 'This MLX setup requires an Apple Silicon Mac.' >&2
    exit 1
fi

printf '%s\n' 'Installing the isolated OpenJev runtime.'
python3 -m venv .openjev-venv
.openjev-venv/bin/python -m pip install -r requirements-openjev.txt

export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
printf '%s\n' 'Downloading pinned official helpers and model weights (approximately 15 GB).'
.openjev-venv/bin/hf download openjev/openjev \
    helper/shim.py helper/shim_mlx.py LICENSE-APACHE-2.0 NOTICE \
    --revision 1c341f65bfe5d50fdb935c71e9739c9e0938d6c4 --local-dir vendor/openjev
.openjev-venv/bin/hf download openjev/openjev-MLX-4bit \
    --revision c59bf1eed7d8de0eb88105a0d5517c9b4a858e16 --local-dir models/openjev-mlx-4bit

printf '%s\n' 'Verifying the published model checksums.'
(cd models/openjev-mlx-4bit && shasum -a 256 -c SHA256SUMS)
printf '%s\n' 'OpenJev setup completed. Run bash scripts/start_openjev.sh.'
