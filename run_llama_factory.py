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
    # Use local port instead of public URL
    os.environ["GRADIO_SERVER_PORT"] = "7860"
    os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
    
    print("Starting LLaMA Factory Web UI...")
    print("\nTo access the UI, open a new terminal and run:")
    print(f"ssh -L 7860:localhost:7860 {os.environ.get('USER', 'your_username')}@cluster_address")
    print("\nThen open http://localhost:7860 in your browser\n")
    
    subprocess.run(["llamafactory-cli", "webui"], check=True)

if __name__ == "__main__":
    main()
