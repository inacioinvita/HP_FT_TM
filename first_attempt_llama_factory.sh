#!/bin/bash

# First, ensure conda is properly initialized
eval "$(conda shell.bash hook)"

# Create the environment if it doesn't exist
conda create -n llama-env python=3.11 -y

# Then activate environment
conda activate llama-env

#
# first_attempt_llama_factory.sh
# 
# Example script that:
#   1. Clones LLaMA Factory
#   2. Installs pinned dependencies (from requirements.txt)
#   3. Installs optional extras for PyTorch, bitsandbytes, etc.
#   4. Verifies GPU is detected by PyTorch
#   5. Demonstrates a minimal training run

set -e  # Exit if any command fails

# --- 1. Clone LLaMA Factory ---
cd ~ || exit 1  # Or choose your preferred directory
rm -rf LLaMA-Factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# --- 2. Install pinned dependencies from the recently updated requirements.txt ---
#     This ensures we match the minimum/maximum versions that LLaMA Factory devs recommend.
pip install -r requirements.txt

# --- 3. (Optional) Install extras to get GPU-based training, bitsandbytes, etc. ---
#     You can add or remove extras as needed: e.g. [torch,metrics,deepspeed,bitsandbytes,liger-kernel].
#     This is helpful if you want quantization or advanced GPU usage.
pip install -e .[torch,bitsandbytes,liger-kernel]

# --- 4. Quick GPU check ---
echo "Verifying that PyTorch sees the GPU..."
python -c "import torch; print('CUDA available?', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count());"

# --- 5. Minimal Fine-Tuning Example ---
#     Create a small JSON config to do a short LoRA fine-tuning on some small dataset.

cat <<EOF > train_llama3.json
{
  "stage": "sft",
  "do_train": true,
  "model_name_or_path": "unsloth/llama-3-8b-Instruct-bnb-4bit", 
  "dataset": "identity,alpaca_en_demo",
  "template": "llama3",
  "finetuning_type": "lora",
  "lora_target": "all",
  "output_dir": "llama3_lora",
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "lr_scheduler_type": "cosine",
  "logging_steps": 10,
  "warmup_ratio": 0.1,
  "save_steps": 100,
  "learning_rate": 5e-5,
  "num_train_epochs": 1.0,
  "max_samples": 50,
  "max_grad_norm": 1.0,
  "loraplus_lr_ratio": 16.0,
  "fp16": true,
  "use_liger_kernel": true
}
EOF

echo
echo "Starting a minimal fine-tuning run (this may take 5-10 minutes or more depending on your GPU)..."
llamafactory-cli train train_llama3.json

echo
echo "Done! You have now installed pinned dependencies and run a minimal training example."