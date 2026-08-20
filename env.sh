# Source this file, don't execute it:  source env.sh
#
# Single place where every path lives, for ALL environments (local Mac,
# Mahamathi A100/SLURM, resiliente-2003 workstation). Nothing here is
# machine-specific -- both roots are derived from this file's own location, so
# the repo works wherever it is cloned or unpacked.
#
# Two roots, because code and data live at different levels:
#
#   CAIMMS_ROOT       the repo itself (this directory). Holds delta-Mem/,
#                     IterRet/, scripts/, docs/.
#   CAIMMS_WORKSPACE  the parent, holding the big things deliberately kept OUT
#                     of git: models/ (~8GB of weights) and outputs/ (run
#                     checkpoints, logs, graph_cache).
#
# Expected layout, identical everywhere:
#
#   <workspace>/
#     models/                Qwen3-4B-Instruct-2507/ + delta-mem-adapter/
#     outputs/               checkpoints, logs, graph_cache/
#     workmem-vertical/      <- CAIMMS_ROOT, this repo
#
# Override CAIMMS_WORKSPACE before sourcing if models/outputs live elsewhere.

CAIMMS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CAIMMS_ROOT
export CAIMMS_WORKSPACE="${CAIMMS_WORKSPACE:-$(cd "${CAIMMS_ROOT}/.." && pwd)}"

# Read directly by deltamem/workmem/*.py. Those files fall back to the original
# Mahamathi absolute paths when these are unset, so they still run bare there.
export CAIMMS_MODEL_PATH="${CAIMMS_WORKSPACE}/models/Qwen3-4B-Instruct-2507"
export CAIMMS_ADAPTER_DIR="${CAIMMS_WORKSPACE}/models/delta-mem-adapter"
export CAIMMS_DATA_FILE="${CAIMMS_ROOT}/delta-Mem/data/locomo10.json"
export CAIMMS_OUTPUT_DIR="${CAIMMS_WORKSPACE}/outputs"

# Port 8000 is not reserved for you on a shared box. Override with
# VLLM_PORT=8xxx before sourcing if something else is already bound to it.
export VLLM_PORT="${VLLM_PORT:-8000}"
export CAIMMS_VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"

# IterRet is VENDORED at ${CAIMMS_ROOT}/IterRet -- there is exactly one copy.
# It used to be a sibling of the repo, which meant two copies that could drift.
export PYTHONPATH="${CAIMMS_ROOT}/delta-Mem:${CAIMMS_ROOT}/IterRet:${PYTHONPATH:-}"

# Keep HF downloads next to the weights rather than in ~/.cache.
export HF_HOME="${HF_HOME:-${CAIMMS_WORKSPACE}/.hf}"

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-workmem}"

# Locate the conda install rather than assuming ~/miniconda3.
if [ -z "${CONDA_PREFIX_DIR:-}" ]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_PREFIX_DIR="$(conda info --base 2>/dev/null || true)"
    fi
    for _c in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" "/opt/conda"; do
        [ -n "${CONDA_PREFIX_DIR:-}" ] && break
        [ -f "${_c}/etc/profile.d/conda.sh" ] && CONDA_PREFIX_DIR="${_c}"
    done
    unset _c
fi
export CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:-$HOME/miniconda3}"

caimms_activate() {
    local hook="${CONDA_PREFIX_DIR}/etc/profile.d/conda.sh"
    if [ ! -f "${hook}" ]; then
        echo "ERROR: no conda at ${CONDA_PREFIX_DIR}" >&2
        echo "       Set CONDA_PREFIX_DIR to your conda base, or run scripts/setup_env.sh first." >&2
        return 1
    fi
    # shellcheck disable=SC1090
    source "${hook}"
    conda activate "${CONDA_ENV_NAME}" || {
        echo "ERROR: conda env '${CONDA_ENV_NAME}' not found. Run scripts/setup_env.sh first." >&2
        return 1
    }
}
