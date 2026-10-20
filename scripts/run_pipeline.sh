#!/usr/bin/env bash
# C-AIMMS (IterRet + Delta-Mem) on LoCoMo -- workstation port of
# run_full_pipeline.slurm, with no SLURM involved.
#
#   bash run_pipeline.sh --smoke     # sample 0 only, 152 questions, ~1h
#   bash run_pipeline.sh             # all 10 samples, 1540 questions, 30h+
#
# Run it under tmux. Nothing here requeues the job if your ssh session drops:
#     tmux new -s caimms
#     bash run_pipeline.sh --smoke
#     <ctrl-b d to detach, tmux attach -t caimms to come back>
#
# GPU layout matches the 2-GPU A100 script: vLLM serves Qwen3-4B on GPU 0, the
# eval loads its own copy plus the Delta-Mem adapter on GPU 1. Both fit inside
# 24GB with room to spare (~8GB of bf16 weights each), so unlike the 1-GPU
# smoke variant on the cluster there is no need to cap vLLM's memory hard.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"

SMOKE=0
[ "${1:-}" = "--smoke" ] && SMOKE=1

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"       # stands in for SLURM_JOB_ID; PID keeps parallel runs' logs apart
mkdir -p "${CAIMMS_OUTPUT_DIR}"
SERVER_LOG="${CAIMMS_OUTPUT_DIR}/server_${RUN_ID}.log"
RUN_LOG="${CAIMMS_OUTPUT_DIR}/run_${RUN_ID}.log"

# EVAL_MODULE: which python module runs once vLLM is up. Default = the LoCoMo
# eval. Other modules (deltamem.workmem.eval_longbench_iterret,
# deltamem.workmem.build_iterret_sft_data) reuse this script's GPU preflight,
# port guards and vLLM lifecycle; they must name their own WORKMEM_OUTPUT_FILE.
LOCOMO_MODULE="deltamem.workmem.eval_locomo_iterret_mock"
EVAL_MODULE="${EVAL_MODULE:-${LOCOMO_MODULE}}"

if [ "${EVAL_MODULE}" != "${LOCOMO_MODULE}" ]; then
    : "${WORKMEM_OUTPUT_FILE:?set WORKMEM_OUTPUT_FILE when EVAL_MODULE is not the LoCoMo eval}"
    export WORKMEM_OUTPUT_FILE
    MODE="${EVAL_MODULE##*.} -> $(basename "${WORKMEM_OUTPUT_FILE}")"
elif [ "${SMOKE}" = "1" ]; then
    export WORKMEM_MAX_SAMPLES=1
    # Scratch checkpoint, never the real one: rows a smoke test writes would
    # otherwise be picked up by the full run's resume logic and silently skipped.
    export WORKMEM_OUTPUT_FILE="${CAIMMS_OUTPUT_DIR}/smoke_results.jsonl"
    rm -f "${WORKMEM_OUTPUT_FILE}"
    MODE="SMOKE (1 sample / 152 questions)"
elif [ -n "${WORKMEM_MAX_SAMPLES:-}" ]; then
    # SUBSET: caller pinned WORKMEM_MAX_SAMPLES (e.g. =4 -> the ~584-question,
    # 4-conversation set). Give it its OWN checkpoint so it never collides with
    # the full run's workmem_iterret_full.jsonl (which resume would otherwise
    # treat as already-done). Not auto-cleared; archive it yourself between runs.
    # Honour a caller-preset WORKMEM_OUTPUT_FILE (so a mode/judge sweep can name
    # each run's file); otherwise default to a per-N checkpoint.
    export WORKMEM_OUTPUT_FILE="${WORKMEM_OUTPUT_FILE:-${CAIMMS_OUTPUT_DIR}/workmem_iterret_n${WORKMEM_MAX_SAMPLES}.jsonl}"
    MODE="SUBSET (${WORKMEM_MAX_SAMPLES} sample(s)) -> $(basename "${WORKMEM_OUTPUT_FILE}")"
else
    export WORKMEM_OUTPUT_FILE="${CAIMMS_OUTPUT_DIR}/workmem_iterret_full.jsonl"
    MODE="FULL (10 samples / 1540 questions)"
fi

caimms_activate

echo "=============================================="
echo "  C-AIMMS ${MODE}"
echo "  run  : ${RUN_ID} on $(hostname) at $(date)"
echo "  out  : ${WORKMEM_OUTPUT_FILE}"
echo "  logs : ${RUN_LOG}"
echo "=============================================="

# ── preflight ─────────────────────────────────────────────────────────────────
# Cheap checks up front. The alternative is discovering a missing file after
# vLLM has spent 90s loading weights, or worse, 20 hours in.
fail=0
for f in "${CAIMMS_MODEL_PATH}/config.json" "${CAIMMS_ADAPTER_DIR}/delta_mem_config.json" \
         "${CAIMMS_ADAPTER_DIR}/delta_mem_adapter.pt" "${CAIMMS_DATA_FILE}"; do
    [ -e "$f" ] || { echo "MISSING: $f"; fail=1; }
done
[ "$fail" = "0" ] || { echo "Run download_assets.sh first."; exit 1; }

# Which physical GPUs to use (nvidia-smi indices). On a box with more cards
# than you own, point these at the free ones: VLLM_GPU=2 EVAL_GPU=3 bash $0
# One card only: VLLM_GPU=0 EVAL_GPU=0 (see single-GPU mode below).
VLLM_GPU="${VLLM_GPU:-0}"
EVAL_GPU="${EVAL_GPU:-1}"
# Single-GPU mode: VLLM_GPU == EVAL_GPU puts the vLLM server and the delta-mem
# eval on one card. vLLM is then capped at VLLM_MEM_UTIL (default 0.52, ~12.7 GB of
# a 24 GB card: ~8 GB weights + start-up overhead + ~2.5 GB KV cache). 0.45 left
# only 1.06 GB of KV cache, below the 1.12 GB one 8192-token request needs, and
# vLLM refused to start. That leaves ~11.8 GB for the eval (~8 GB weights +
# working space). Fine for LongBench and for the
# retrieval setups of MemoryAgentBench; the truncated setups need a smaller
# window there (MAB_PROMPT_TOKENS=8192).
if [ "${VLLM_GPU}" = "${EVAL_GPU}" ]; then
    SINGLE_GPU=1
    GPU_IDS="${VLLM_GPU}"
    VLLM_MEM_UTIL="${VLLM_MEM_UTIL:-0.52}"
    NEED_MIB_DEFAULT=22000
else
    SINGLE_GPU=0
    GPU_IDS="${VLLM_GPU},${EVAL_GPU}"
    VLLM_MEM_UTIL="${VLLM_MEM_UTIL:-0.85}"
    NEED_MIB_DEFAULT=12000
fi
NGPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
for g in "${VLLM_GPU}" "${EVAL_GPU}"; do
    [ "${g}" -lt "${NGPU}" ] || { echo "ERROR: GPU ${g} does not exist (found ${NGPU})."; exit 1; }
done

# GPU availability preflight. This box is SHARED -- other users' jobs routinely
# occupy both cards, and without this check vLLM starts, spends ~30s loading,
# then dies with a CUDA OOM buried ~90 lines deep in its log behind a useless
# "Engine core initialization failed. See root cause above."
NEED_MIB="${CAIMMS_MIN_FREE_MIB:-${NEED_MIB_DEFAULT}}"
BUSY=0
while read -r idx free; do
    if [ "${free}" -lt "${NEED_MIB}" ]; then
        echo "GPU ${idx}: only ${free} MiB free (need >= ${NEED_MIB} MiB)"
        BUSY=1
    else
        echo "GPU ${idx}: ${free} MiB free -- OK"
    fi
done < <(nvidia-smi --id="${GPU_IDS}" --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ',')

if [ "${BUSY}" = "1" ]; then
    echo
    echo "ERROR: not enough free GPU memory. Who is using the cards:"
    nvidia-smi --id="${GPU_IDS}" --query-compute-apps=pid,process_name,used_memory --format=csv | sed 's/^/    /'
    echo
    echo "  Pick free cards with VLLM_GPU=<i> EVAL_GPU=<j>. These may belong to another user on this shared machine -- check before"
    echo "  killing anything. Wait for them to finish, or run on Mahamathi instead."
    echo "  To override this guard (e.g. you know a smaller footprint will fit):"
    echo "      CAIMMS_MIN_FREE_MIB=4000 bash $0 $*"
    exit 1
fi
echo "GPUs (vLLM on ${VLLM_GPU}, eval on ${EVAL_GPU}$([ "${SINGLE_GPU}" = "1" ] && echo ", SINGLE-GPU mode, vLLM capped at ${VLLM_MEM_UTIL}")):"
nvidia-smi --id="${GPU_IDS}" --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed 's/^/  /'

# No scheduler on this box means nobody is holding the GPUs for you. If someone
# else's process is resident, say so rather than OOMing 90 seconds from now.
OTHER="$(nvidia-smi --id="${GPU_IDS}" --query-compute-apps=pid,used_memory --format=csv,noheader | wc -l)"
if [ "${OTHER}" -gt 0 ]; then
    echo "  NOTE: ${OTHER} compute process(es) already on these GPUs:"
    nvidia-smi --id="${GPU_IDS}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/    /'
    echo "  Continuing in 10s -- ctrl-c to abort."
    sleep 10
fi

# IterRet swallows any MiniLM load failure and silently downgrades to keyword
# overlap, which changes retrieval without changing the exit code. Fail loudly.
python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode("warmup")
print("preflight: MiniLM retrieval encoder OK")
PY

# ── clean up our own stale servers ────────────────────────────────────────────
# -u "$USER": the cluster script's bare pkill was safe inside a private SLURM
# allocation, but this box is shared -- an unscoped pkill would kill someone
# else's vLLM.
#
# Scoped to THIS port: on a multi-GPU box two pipelines can run side by side
# (VLLM_GPU=0 EVAL_GPU=1 VLLM_PORT=8002 and VLLM_GPU=2 EVAL_GPU=3 VLLM_PORT=8003),
# and an unscoped pkill would kill the other run's server mid-eval.
OWN_VLLM_PATTERN="vllm.entrypoints.openai.api_server.*--port ${VLLM_PORT}( |\$)"
pkill -u "$USER" -f "${OWN_VLLM_PATTERN}" 2>/dev/null || true
sleep 3

# The bare pkill+sleep above is not enough on its own: if a stale server (ours,
# mid-shutdown, or another user's) is STILL holding VLLM_PORT, the fresh server
# below dies with "Address already in use" -- but the health check further down
# is a plain `curl /v1/models`, which the stale server ANSWERS, so the eval
# silently proceeds against the wrong server. That exact race invalidated a full
# 1540-question run (every question routed via fail_open_parse_failed, n_ev=6).
# So: actively wait for the port to be FREE before starting, and abort loudly if
# something else won't let go of it.
echo "Ensuring port ${VLLM_PORT} is free..."
# Anything LISTENING on the port that is not an HTTP server also blocks us: a
# vLLM engine process opens internal sockets on random free ports, and once took
# 8003 while a second pipeline wanted it -- curl then connected and hung forever
# (hence --max-time on every curl here). Fail fast with a clear message instead.
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${VLLM_PORT}\$" \
   && ! curl -sf --max-time 5 "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
    echo "ERROR: port ${VLLM_PORT} is held by a non-vLLM listener:"
    ss -ltnp 2>/dev/null | grep -E "[:.]${VLLM_PORT} " | sed 's/^/    /'
    echo "  Pick another port: VLLM_PORT=8xxx bash $0 $*"
    exit 1
fi
FREE_WAIT=0
while curl -sf --max-time 5 "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
    pkill -9 -u "$USER" -f "${OWN_VLLM_PATTERN}" 2>/dev/null || true   # our own stragglers on this port, harder
    sleep 3; FREE_WAIT=$((FREE_WAIT + 3))
    if [ ${FREE_WAIT} -ge 30 ]; then
        echo "ERROR: port ${VLLM_PORT} is still serving after ${FREE_WAIT}s -- a vLLM"
        echo "  server we cannot kill (likely another user's) is holding it. Point"
        echo "  this run at a different port and retry:  VLLM_PORT=8001 bash $0 $*"
        echo "  Current occupant /v1/models says:"
        curl -s --max-time 5 "http://localhost:${VLLM_PORT}/v1/models" | head -c 300; echo
        exit 1
    fi
    echo "      ...still occupied after ${FREE_WAIT}s, retrying kill"
done
echo "      port ${VLLM_PORT} free."

SERVER_PID=""
# vLLM runs in its OWN process group (setsid below), so stopping the group also
# stops its "VLLM::EngineCore" worker -- killing only the API-server pid used to
# leave that worker orphaned, still holding the GPU.
cleanup() {
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Stopping vLLM (process group ${SERVER_PID})..."
        kill -TERM -- -"${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
        for _ in $(seq 1 20); do kill -0 -- -"${SERVER_PID}" 2>/dev/null || break; sleep 1; done
        kill -KILL -- -"${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

cd "${CAIMMS_ROOT}/delta-Mem"

# ── 1. vLLM on GPU 0 ──────────────────────────────────────────────────────────
echo "[1/3] Starting vLLM on GPU ${VLLM_GPU} (port ${VLLM_PORT})..."
CUDA_VISIBLE_DEVICES="${VLLM_GPU}" setsid python3 -m vllm.entrypoints.openai.api_server \
    --model "${CAIMMS_MODEL_PATH}" \
    --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
    --port "${VLLM_PORT}" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization "${VLLM_MEM_UTIL}" \
    --enforce-eager \
    --disable-log-requests \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "      pid ${SERVER_PID}, log ${SERVER_LOG}"

# ── 2. wait ───────────────────────────────────────────────────────────────────
MAX_WAIT=600; INTERVAL=10; ELAPSED=0
echo "[2/3] Waiting for server (up to ${MAX_WAIT}s)..."
until curl -sf --max-time 5 "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: vLLM died during startup. Tail of ${SERVER_LOG}:"
        grep -nE "OutOfMemory|CUDA|Error|Traceback|Address already in use" "${SERVER_LOG}" | tail -15; echo "--- last 25 lines ---"; tail -25 "${SERVER_LOG}"
        exit 1
    fi
    sleep ${INTERVAL}; ELAPSED=$((ELAPSED + INTERVAL))
    if [ ${ELAPSED} -ge ${MAX_WAIT} ]; then
        echo "ERROR: server did not come up within ${MAX_WAIT}s."
        grep -nE "OutOfMemory|CUDA|Error|Traceback|Address already in use" "${SERVER_LOG}" | tail -15; echo "--- last 25 lines ---"; tail -25 "${SERVER_LOG}"
        exit 1
    fi
    echo "      ...${ELAPSED}s"
done

# Confirm the server that answered is the one WE started and serving the model
# we expect -- a stale/foreign server on this port would also answer /v1/models
# but may serve a different model or return unusable output. Belt-and-suspenders
# on top of the port-free wait above.
if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ERROR: our vLLM (pid ${SERVER_PID}) is not alive but the port answers --"
    echo "  a different server is on ${VLLM_PORT}. Refusing to run against it."
    grep -nE "Address already in use|Error|Traceback" "${SERVER_LOG}" | tail -10
    exit 1
fi
MODELS_JSON="$(curl -sf --max-time 5 "http://localhost:${VLLM_PORT}/v1/models" || true)"
if ! printf '%s' "${MODELS_JSON}" | grep -q "Qwen/Qwen3-4B-Instruct-2507"; then
    echo "ERROR: server on ${VLLM_PORT} is not serving Qwen/Qwen3-4B-Instruct-2507:"
    printf '%s\n' "${MODELS_JSON}" | head -c 400; echo
    exit 1
fi
echo "      server online (pid ${SERVER_PID}, model verified)."

# ── 3. eval on GPU 1 ──────────────────────────────────────────────────────────
echo "[3/3] Starting eval on GPU ${EVAL_GPU}..."
set +e
CUDA_VISIBLE_DEVICES="${EVAL_GPU}" python3 -u -m "${EVAL_MODULE}" 2>&1 | tee -a "${RUN_LOG}"
EVAL_EXIT=${PIPESTATUS[0]}
set -e

ROWS="$(wc -l < "${WORKMEM_OUTPUT_FILE}" 2>/dev/null || echo 0)"
if [ "${EVAL_MODULE}" = "${LOCOMO_MODULE}" ]; then
    EXPECT=" (expect $([ "${SMOKE}" = "1" ] && echo 152 || echo 1540))"
else
    EXPECT=""
fi
echo "=============================================="
echo "  Done at $(date) | exit ${EVAL_EXIT}"
echo "  Rows written: ${ROWS}${EXPECT}"
echo "  Results: ${WORKMEM_OUTPUT_FILE}"
echo "=============================================="
exit "${EVAL_EXIT}"
