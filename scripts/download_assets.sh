#!/usr/bin/env bash
# Fetches the two model assets that are too large to ship in this bundle.
# The Delta-Mem adapter and the LoCoMo dataset are already here -- only the
# Qwen base weights and the MiniLM retrieval encoder need downloading.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"
caimms_activate

echo "=== Delta-Mem adapter -> ${CAIMMS_ADAPTER_DIR} ==="
# On a fresh clone the trained adapter ships INSIDE the repo (trained_models/),
# but the eval loads it from the workspace models/ dir. Provision it there.
if [ -f "${CAIMMS_ADAPTER_DIR}/delta_mem_config.json" ]; then
    echo "  already present, skipping."
else
    mkdir -p "${CAIMMS_ADAPTER_DIR}"
    cp "${CAIMMS_ROOT}/trained_models/delta_mem_adapter.pt" \
       "${CAIMMS_ROOT}/trained_models/delta_mem_config.json" \
       "${CAIMMS_ADAPTER_DIR}/"
    echo "  copied from ${CAIMMS_ROOT}/trained_models/ ($(du -h "${CAIMMS_ADAPTER_DIR}/delta_mem_adapter.pt" | cut -f1))"
fi

echo
echo "=== Qwen3-4B-Instruct-2507 -> ${CAIMMS_MODEL_PATH} (~8GB) ==="
if [ -f "${CAIMMS_MODEL_PATH}/config.json" ]; then
    echo "  already present, skipping."
else
    mkdir -p "${CAIMMS_MODEL_PATH}"
    # local-dir keeps a plain directory rather than a symlinked blob cache,
    # because both vLLM and the eval load it with local_files_only=True.
    hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir "${CAIMMS_MODEL_PATH}" \
        || huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir "${CAIMMS_MODEL_PATH}"
fi

echo
echo "=== all-MiniLM-L6-v2 (IterRet retrieval encoder) ==="
# This one matters more than its size suggests. IterRet's
# build_default_embedding_backend() catches ANY load failure and silently falls
# back to KeywordOverlapEmbeddingBackend -- so if this model is missing at run
# time you get a completed run with quietly different retrieval and no error.
python - <<'PY'
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
v = m.encode("warmup")
print(f"  MiniLM cached and working (dim={len(v)})")
PY

echo
echo "Assets ready."
du -sh "${CAIMMS_MODEL_PATH}" "${CAIMMS_ADAPTER_DIR}" 2>/dev/null || true
