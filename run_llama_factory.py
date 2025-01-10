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
    # Set specific port and host for cluster environment
    os.environ["GRADIO_SERVER_PORT"] = "6006"  # Choose a specific port
    os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"  # Listen on all interfaces
    
    print("Starting LLaMA Factory Web UI...")
    print("You may need to set up port forwarding to access the UI:")
    print("ssh -L 6006:localhost:6006 your_username@cluster_address")
    
    subprocess.run(["llamafactory-cli", "webui", "--listen", "--share"], check=True)

if __name__ == "__main__":
    main()
