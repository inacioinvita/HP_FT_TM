#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Investigate CUDA + BitsAndBytes Compatibility on an HPC Cluster
#
# Usage:
#   1) chmod +x investigate_cuda_compatibility.sh
#   2) ./investigate_cuda_compatibility.sh
#
# This script:
#   1. Logs HPC info (driver version, env vars, etc.)
#   2. Creates a fresh conda environment
#   3. Installs PyTorch with CUDA
#   4. Verifies GPU detection with PyTorch
#   5. Installs BitsAndBytes from GitHub and verifies CUDA detection
###############################################################################


########################
# 1) Cluster Diagnostics
########################

echo "=== Step 1: Cluster Diagnostics ==="

# 1a. Nvidia driver + GPU info
echo "NVIDIA GPU + Driver Info:"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi || echo "nvidia-smi failed, GPU may not be accessible."
else
    echo "nvidia-smi not found"
fi

# 1b. Show environment variables
echo
echo "Environment Variables (CUDA-related):"
echo "CUDA_HOME=${CUDA_HOME:-UNSET}"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "PATH=$PATH"
echo

# 1c. Show HPC relevant modules if "module" is available
if command -v module &>/dev/null; then
    echo "Modules loaded:"
    module list || echo "Could not list modules."
else
    echo "No 'module' command found, skipping loaded modules check."
fi

# 1d. Basic system info
echo
echo "System Info:"
uname -a
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo


#####################################
# 2) Create Fresh Conda Environment
#####################################

echo "=== Step 2: Creating Fresh Conda Environment ==="

# 2a. Initialize conda (adjust path to conda if needed)
if ! command -v conda &>/dev/null; then
  echo "Conda not in PATH, please adjust the path to conda or load the condamodule."
  exit 1
fi

eval "$(conda shell.bash hook)"

# 2b. Create + activate new environment
ENV_NAME="cuda_test_env"
echo "Removing environment '${ENV_NAME}' if it exists..."
conda deactivate || true
conda env remove -n "${ENV_NAME}" --yes || true
echo "Creating new conda environment: ${ENV_NAME}"
conda create -n "${ENV_NAME}" python=3.8 -y
conda activate "${ENV_NAME}"

echo "Conda environment ready:"
conda info --envs
echo


#########################
# 3) PyTorch Installation
#########################

echo "=== Step 3: PyTorch with CUDA Installation ==="

# By default, we attempt to match your system’s CUDA version. 
# For HPC clusters, using a specific known-good version (e.g., 11.8, 12.1) is common.
# We'll first try pip (which provides official wheels for certain CUDA versions).
# If that fails to detect GPUs, we try conda.

# 3a. Attempt pip-based PyTorch installation with CUDA 12.1 wheels
echo "Installing PyTorch from PyPI (CUDA 12.1 wheels)..."
python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# 3b. Verify GPU availability
echo "Verifying PyTorch GPU detection..."
if python -c "
import torch
print('PyTorch version:', torch.__version__)
print('PyTorch CUDA available:', torch.cuda.is_available())
print('CUDA version (torch.version.cuda):', torch.version.cuda)
if not torch.cuda.is_available():
    raise RuntimeError('GPU not detected by PyTorch.')
"; then
  echo "PyTorch pip install succeeded with CUDA!"
else
  echo "PyTorch pip install failed to detect CUDA. Trying conda-based install with CUDA 11.8..."
  # Clean up pip-based PyTorch
  python -m pip uninstall -y torch

  # Attempt conda-based install with CUDA 11.8
  conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

  # Final check:
  python -c "
import torch
print('Conda PyTorch version:', torch.__version__)
print('Conda PyTorch CUDA available:', torch.cuda.is_available())
print('Conda CUDA version (torch.version.cuda):', torch.version.cuda)
assert torch.cuda.is_available(), 'GPU not detected even after conda install.'
  "
fi

echo
echo "PyTorch installation + verification complete."
echo


########################################
# 4) BitsAndBytes Installation & Check
########################################

echo "=== Step 4: BitsAndBytes Installation & Verification ==="

# 4a. Install BitsAndBytes from GitHub's main or multi-backend branch
echo "Installing BitsAndBytes from GitHub..."
python -m pip install --no-cache-dir --upgrade \
  "bitsandbytes@https://github.com/bitsandbytes-foundation/bitsandbytes/archive/refs/heads/main.zip"

# 4b. Verify that BitsAndBytes compiled with CUDA
echo "Verifying BitsAndBytes GPU support..."
python -c "
import bitsandbytes as bnb
import torch

print('BitsAndBytes version:', getattr(bnb, '__version__', 'unknown'))
try:
    compiled_with_cuda = bnb.COMPILED_WITH_CUDA
    print('BNB compiled with CUDA:', compiled_with_cuda)
except AttributeError:
    print('BNB is missing COMPILED_WITH_CUDA attribute; no CUDA support found.')

# If PyTorch has a GPU, but bitsandbytes isn't compiled with CUDA, raise error
if torch.cuda.is_available():
    assert hasattr(bnb, 'COMPILED_WITH_CUDA') and bnb.COMPILED_WITH_CUDA, \
        'BitsAndBytes is not CUDA-enabled despite PyTorch seeing a GPU.'
"

echo
echo "BitsAndBytes installation check complete."
echo


##################################################################
# 5) Summary and Next Steps
##################################################################

echo "=== Summary ==="
echo "If you see any assertion failures or exception traces above,"
echo "please review them for potential driver mismatch, library conflicts,"
echo "or environment variable issues."
echo
echo "If PyTorch recognized the GPU and BitsAndBytes reported 'BNB compiled with CUDA: True',"
echo "your environment is likely ready for 4-bit quantization runs!"
echo
echo "You can now install additional libraries or proceed with your training script."
echo "================================" 