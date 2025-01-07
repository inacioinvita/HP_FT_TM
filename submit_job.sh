#!/bin/bash

# 4) Pull the latest changes from the repo
# git fetch origin
# git checkout origin/main -- run_BALS_trainer.job BALS_trainer.py
git pull origin main

# Convert line endings to Unix format
sed -i 's/\r$//' run_BALS_trainer.job

# Submit the job
sbatch run_BALS_trainer.job