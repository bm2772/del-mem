#!/bin/bash
while true; do
  # Explicitly force the a100 partition
  JOB_ID=$(sbatch --parsable -p a100 ~/workmem_test/run_full_pipeline.slurm)
  
  # Safety check: if sbatch fails, don't crash
  if [ -z "$JOB_ID" ]; then
    echo "Error: sbatch failed! Retrying in 10 seconds..."
    sleep 10
    continue
  fi

  echo "Successfully submitted job $JOB_ID to a100. Guarding it..."
  
  # Wait while the job exists in the queue
  while squeue -j "$JOB_ID" 2>/dev/null | grep -q "$JOB_ID"; do
    sleep 30
  done
  
  # Check if it finished successfully
  STATE=$(sacct -j "$JOB_ID" --format=State -n | head -n 1 | tr -d ' ')
  if [[ "$STATE" == *"COMPLETED"* ]]; then
    echo "Benchmark successfully completed!"
    break
  fi
  
  echo "SLURM assassinated job $JOB_ID (State: $STATE). Resubmitting..."
  sleep 5
done
