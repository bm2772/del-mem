#!/usr/bin/env bash
# One-time environment bootstrap for resiliente-2003.
#
# The box ships Python 3.10 and no conda, but the pinned flash-attn wheel is
# cp311-only, so a 3.11 conda env is not optional here.
#
#   bash setup_env.sh
#
# Safe to re-run: every step is skipped if it already happened.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"   # resolves CONDA_PREFIX_DIR

echo "=============================================="
echo " C-AIMMS environment bootstrap"
echo " bundle : ${CAIMMS_ROOT}"
echo " env    : ${CONDA_ENV_NAME} (python 3.11)"
echo "=============================================="

# ── 1. conda ──────────────────────────────────────────────────────────────────
if ! command -v conda >/dev/null 2>&1 && [ ! -d "${CONDA_PREFIX_DIR}" ]; then
    echo "[1/4] No conda found. Installing Miniconda to ${CONDA_PREFIX_DIR}..."
    INSTALLER="/tmp/miniconda_${USER}.sh"
    wget -q --show-progress -O "${INSTALLER}" \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash "${INSTALLER}" -b -p "${CONDA_PREFIX_DIR}"
    rm -f "${INSTALLER}"
else
    echo "[1/4] conda already present, skipping install."
fi

# shellcheck disable=SC1091
source "${CONDA_PREFIX_DIR}/etc/profile.d/conda.sh"

# ── 2. env ────────────────────────────────────────────────────────────────────
if conda env list | grep -qE "^${CONDA_ENV_NAME}\s"; then
    echo "[2/4] Env '${CONDA_ENV_NAME}' exists, skipping create."
else
    echo "[2/4] Creating env '${CONDA_ENV_NAME}' with Python 3.11 from conda-forge..."
    # conda-forge + --override-channels deliberately: Anaconda's default
    # channels (repo.anaconda.com/pkgs/*) now gate on a Terms of Service accept
    # that carries commercial licensing obligations for larger organisations.
    # Nothing here needs them -- conda supplies only the 3.11 interpreter, and
    # every real dependency comes from pip below.
    conda create -n "${CONDA_ENV_NAME}" python=3.11 -y -c conda-forge --override-channels
fi
conda activate "${CONDA_ENV_NAME}"

# Keep any later `conda install` in this env off the default channels too.
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict

python -c "import sys; assert sys.version_info[:2]==(3,11), sys.version; print('  python', sys.version.split()[0])"

# ── 3. packages ───────────────────────────────────────────────────────────────
# One resolver pass over the full freeze. If pip cannot solve it, fall back to
# the staged install documented in README.md rather than relaxing the pins --
# the pins are what the current LoCoMo numbers were produced with.
echo "[3/4] Installing pinned packages (this pulls ~6GB of wheels)..."
pip install --upgrade pip
pip install -r "${CAIMMS_ROOT}/requirements-cluster.txt"

# ── 4. verify ─────────────────────────────────────────────────────────────────
echo "[4/4] Verifying..."
python - <<'PY'
import torch, importlib.metadata as md
print("  torch       ", torch.__version__)
print("  cuda avail  ", torch.cuda.is_available())
print("  gpu count   ", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"    [{i}] {p.name}  {p.total_memory/1024**3:.1f} GiB  sm_{p.major}{p.minor}")
for pkg in ("vllm", "transformers", "pydantic", "sentence-transformers", "peft", "flash_attn"):
    try:
        print(f"  {pkg:<20} {md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"  {pkg:<20} MISSING")
assert torch.cuda.is_available(), "CUDA not visible from torch -- check the driver"
PY

echo
echo "Environment ready. Next:"
echo "  bash ${CAIMMS_ROOT}/download_assets.sh     # ~8GB of Qwen3-4B weights"
echo "  bash ${CAIMMS_ROOT}/run_pipeline.sh --smoke"
