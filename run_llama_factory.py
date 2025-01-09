#!/usr/bin/env python3

import os
import subprocess
import sys

def main():
    # --- CUDA AND ENVIRONMENT SETUP ---
    # These environment variables should match run_BALS_trainer.job
    os.environ["CUDA_HOME"] = "/usr/local/cuda-12.5"
    os.environ["LD_LIBRARY_PATH"] = f"{os.environ['CUDA_HOME']}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:{os.environ.get('PATH', '')}"

    # Use SLURM-provided GPU devices
    if "SLURM_STEP_GPUS" in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_STEP_GPUS"]

    workdir = os.getcwd()

    # --- CLONE THE REPO ---
    llama_dir = os.path.join(workdir, "LLaMA-Factory")
    if not os.path.isdir(llama_dir):
        subprocess.run(["git", "clone", "https://github.com/hiyouga/LLaMA-Factory.git"], check=True)

    os.chdir(llama_dir)

    # --- INSTALL REQUIREMENTS ---
    # Install PyTorch first (similar to run_BALS_trainer.job)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir",
                   "torch", "torchvision", "torchaudio", 
                   "--index-url", "https://download.pytorch.org/whl/cu118"], check=True)

    # Install LLaMA Factory requirements
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-e", ".[torch,bitsandbytes]"], check=True)

    # --- LOGIN TO HUGGINGFACE HUB ---
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
    if not hf_token:
        raise ValueError(
            "HUGGINGFACEHUB_API_TOKEN environment variable not set.\n"
            "Please add 'export HUGGINGFACEHUB_API_TOKEN=your_token_here' to run_factory.job"
        )

    from huggingface_hub import login
    login(token=hf_token)

    # --- RUN LLAMA FACTORY CLI ---
    # Using a specific port that matches your cluster's setup
    os.environ["GRADIO_SERVER_PORT"] = "6927"  # Using same port as in search results
    
    # Run the CLI
    subprocess.run(["llamafactory-cli", "webui"], check=True)

if __name__ == "__main__":
    main()
