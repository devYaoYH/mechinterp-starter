#!/bin/bash
# Bootstrap on a fresh GPU box. Picks the torch CUDA build from the GPU's
# compute capability -- an old wheel installs cleanly and then dies at the
# first GPU op with "no kernel image is available for execution on the device".
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.12 .venv
source .venv/bin/activate

CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1 || echo "")"
echo "Detected GPU compute capability: ${CC:-unknown}"

# Blackwell (cc >= 10.0: RTX 50-series, B200) requires CUDA >= 12.8 wheels.
TORCH_INDEX="https://download.pytorch.org/whl/cu124"
if [ -n "${CC:-}" ] && awk -v cc="$CC" 'BEGIN{exit !(cc+0 >= 10.0)}'; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
fi
echo "Installing torch from: $TORCH_INDEX"
uv pip install torch --index-url "$TORCH_INDEX"
uv pip install -r requirements.txt

echo
echo "Setup complete. Next:"
echo "  source .venv/bin/activate"
echo "  python smoke_test.py          # verify the whole stack before writing research code"
