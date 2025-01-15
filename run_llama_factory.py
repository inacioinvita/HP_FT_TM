#!/usr/bin/env python3

import os
import subprocess
import sys

def main():
    workdir = os.getcwd()

    # --- CLONE THE REPO ---
    llama_dir = os.path.join(workdir, "LLaMA-Factory")
    if not os.path.isdir(llama_dir):
        subprocess.run(["git", "clone", "https://github.com/hiyouga/LLaMA-Factory.git"], check=True)

    os.chdir(llama_dir)

    # --- INSTALL REQUIREMENTS ---
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[torch,bitsandbytes]"], check=True)

    # --- LOGIN TO HUGGINGFACE HUB ---
    from huggingface_hub import login
    login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"])

    # --- RUN LLAMA FACTORY CLI ---
    # Enable Gradio share feature (like in Colab)
    os.environ["GRADIO_SHARE"] = "1"
    
    print("Starting LLaMA Factory Web UI...")
    print("Waiting for public URL...")
    
    subprocess.run(["llamafactory-cli", "webui"], check=True)

if __name__ == "__main__":
    main()
