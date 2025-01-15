#!/bin/bash

# Pull specific files from main branch, ignoring submit_job.sh
git fetch origin
git checkout origin/main -- inference_eval.py run_llama_factory.job run_sage_train_inf_eval.job trainer_cp2_pt.py run_BALS_trainer.job BALS_trainer.py run_factory.job run_llama_factory.py first_attempt_llama_factory.sh

# Convert line endings to Unix format
sed -i 's/\r$//' inference_eval.py run_llama_factory.job run_BALS_trainer.job run_sage_train_inf_eval.job run_factory.job run_llama_factory.py first_attempt_llama_factory.sh

# Submit the job
sbatch run_llama_factory.job
