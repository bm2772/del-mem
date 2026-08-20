#!/bin/bash
# Generic checkpoint-aware resubmission loop, parameterized by output file --
# unlike guardian.sh (hardcoded to the main production checkpoint path and
# threshold), this works for any WORKMEM_OUTPUT_FILE-scoped run. Built for the
# two isolated ranking-attribution arms (idf_only / idf_plus_fusion), which
# guardian.sh can't safely supervise since it would poll the wrong file.
#
# Usage: ./guardian_arm.sh <output_file_path> [extra_sbatch_export_vars]
#   ./guardian_arm.sh /home/kbasu/arnavbhatt/workmem_test/outputs/workmem_idf_only.jsonl DISABLE_CONTENT_EMBEDDER_FUSION=1
#   ./guardian_arm.sh /home/kbasu/arnavbhatt/workmem_test/outputs/workmem_idf_plus_fusion.jsonl
#
# Same logic as guardian.sh, same 1540 threshold (1986 total LoCoMo questions
# minus 446 adversarial, which the eval script excludes entirely and never
# writes a row for -- this file can never reach 1986 lines).

SLURM_SCRIPT="/home/kbasu/arnavbhatt/workmem_test/run_full_pipeline.slurm"
OUTPUT_FILE="$1"
EXTRA_EXPORT="$2"

if [ -z "${OUTPUT_FILE}" ]; then
    echo "usage: $0 <output_file> [extra_export_vars]"
    exit 1
fi

EXPORT_VARS="ALL,WORKMEM_OUTPUT_FILE=${OUTPUT_FILE}"
if [ -n "${EXTRA_EXPORT}" ]; then
    EXPORT_VARS="${EXPORT_VARS},${EXTRA_EXPORT}"
fi

TAG="guardian_arm:$(basename "${OUTPUT_FILE}")"
echo "[${TAG}] Starting. export=${EXPORT_VARS}"
while true; do
    N_DONE=$([ -f "${OUTPUT_FILE}" ] && wc -l < "${OUTPUT_FILE}" || echo 0)
    echo "[${TAG}] Submitting (${N_DONE} questions done so far)..."
    JOB_ID=$(sbatch --parsable --export="${EXPORT_VARS}" "${SLURM_SCRIPT}")
    if [ -z "${JOB_ID}" ]; then
        echo "[${TAG}] sbatch failed. Retrying in 30s..."
        sleep 30
        continue
    fi
    echo "[${TAG}] Job ${JOB_ID} submitted."
    while squeue -j "${JOB_ID}" 2>/dev/null | grep -q "${JOB_ID}"; do
        sleep 30
    done
    N_DONE=$([ -f "${OUTPUT_FILE}" ] && wc -l < "${OUTPUT_FILE}" || echo 0)
    echo "[${TAG}] Job ${JOB_ID} finished. ${N_DONE} questions done."
    if [ "${N_DONE}" -ge 1540 ]; then
        echo "[${TAG}] Complete. Stopping."
        exit 0
    fi
    echo "[${TAG}] Resubmitting in 5s..."
    sleep 5
done
