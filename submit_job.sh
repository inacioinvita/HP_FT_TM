#!/bin/bash

# Convert line endings to Unix format
sed -i 's/\r$//' run_BALS_trainer.job

# Submit the job
sbatch run_BALS_trainer.job