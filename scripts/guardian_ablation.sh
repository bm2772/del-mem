#!/bin/bash
SLURM_SCRIPT="/home/kbasu/arnavbhatt/workmem_test/run_ablation.slurm"
OUTPUT_FILE="/home/kbasu/arnavbhatt/workmem_test/outputs/workmem_ablation_direct.jsonl"

echo "[guardian_ablation] Starting."
while true; do
    N_DONE=$([ -f "${OUTPUT_FILE}" ] && wc -l < "${OUTPUT_FILE}" || echo 0)
    echo "[guardian_ablation] Submitting (${N_DONE} questions done)..."
    JOB_ID=$(sbatch --parsable "${SLURM_SCRIPT}")
    if [ -z "${JOB_ID}" ]; then
        echo "[guardian_ablation] sbatch failed. Retrying in 30s..."
        sleep 30; continue
    fi
    echo "[guardian_ablation] Job ${JOB_ID} submitted."
    while squeue -j "${JOB_ID}" 2>/dev/null | grep -q "${JOB_ID}"; do
        sleep 30
    done
    N_DONE=$([ -f "${OUTPUT_FILE}" ] && wc -l < "${OUTPUT_FILE}" || echo 0)
    echo "[guardian_ablation] Job ${JOB_ID} finished. ${N_DONE} questions done."
    if [ "${N_DONE}" -ge 1986 ]; then
        echo "[guardian_ablation] Complete."
        exit 0
    fi
    sleep 5
done
