#!/bin/bash

# Pull specific files from main branch, ignoring submit_job.sh
git fetch origin
git checkout origin/main -- run_BALS_trainer.job BALS_trainer.py

# Convert line endings to Unix format
sed -i 's/\r$//' run_BALS_trainer.job

# Submit the job
sbatch run_BALS_trainer.job