#!/bin/bash
SLURM_SCRIPT="/home/kbasu/arnavbhatt/workmem_test/run_full_pipeline.slurm"
OUTPUT_FILE="/home/kbasu/arnavbhatt/workmem_test/outputs/workmem_iterret_full.jsonl"

echo "[guardian] Starting."
while true; do
    N_DONE=$([ -f "${OUTPUT_FILE}" ] && wc -l < "${OUTPUT_FILE}" || echo 0)
    echo "[guardian] Submitting (${N_DONE} questions done so far)..."
    JOB_ID=$(sbatch --parsable "${SLURM_SCRIPT}")
    if [ -z "${JOB_ID}" ]; then
        echo "[guardian] sbatch failed. Retrying in 30s..."
        sleep 30
        continue
    fi
    echo "[guardian] Job ${JOB_ID} submitted."
    while squeue -j "${JOB_ID}" 2>/dev/null | grep -q "${JOB_ID}"; do
        sleep 30
    done
    N_DONE=$([ -f "${OUTPUT_FILE}" ] && wc -l < "${OUTPUT_FILE}" || echo 0)
    echo "[guardian] Job ${JOB_ID} finished. ${N_DONE} questions done."
    # 1540 = 1986 total LoCoMo questions - 446 adversarial (category 5).
    # Adversarial questions are excluded from the eval entirely and no longer
    # write a row to the output file, so this file can never reach 1986 lines.
    if [ "${N_DONE}" -ge 1540 ]; then
        echo "[guardian] Complete. Stopping."
        exit 0
    fi
    echo "[guardian] Resubmitting in 5s..."
    sleep 5
done
