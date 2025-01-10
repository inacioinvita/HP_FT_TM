#!/bin/bash

# First, ensure conda is properly initialized
eval "$(conda shell.bash hook)"

# Remove existing environment if it exists
conda deactivate
conda env remove -n llama-env -y
conda clean -a -y

# Create the environment with specific CUDA support
conda create -n llama-env python=3.11 -y
conda activate llama-env

# Install CUDA-enabled PyTorch first
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install bitsandbytes with CUDA support
pip install bitsandbytes==0.43.1 --prefer-binary

# Install other dependencies
pip install -r requirements.txt
pip install -e .[torch,bitsandbytes]

# Quick GPU check
echo "Verifying that PyTorch sees the GPU..."
python -c "import torch; print('CUDA available?', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count());"

# Create a small JSON config for fine-tuning
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
echo "Starting a minimal fine-tuning run..."
llamafactory-cli train train_llama3.json

