# =======================================
# Author: Hung Tran-Nam
# Email: namhung34.info@gmail.com
# Repo: https://github.com/hungtrannam/probclust
# =======================================
# File: builtEnv.sh
# Description: Script to set up a Python virtual environment with required packages


#!/bin/bash
set -e

echo "🟢 Updating system..."
sudo apt update -y && sudo apt upgrade -y
sudo apt install -y build-essential

ENV_NAME="proclustEnv"
conda deactivate 2>/dev/null || true
conda env remove -n "$ENV_NAME" -y || true
echo "✅ Creating clean environment '$ENV_NAME' with Python 3.11..."
conda create -n "$ENV_NAME" python=3.11 -y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 1. PyTorch CUDA 12.1 (không ép 11.8)
echo "🔥 Installing PyTorch CUDA 12.1..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# # 2. TensorFlow GPU build CUDA 12
# echo "📦 Installing TensorFlow GPU..."
# pip install tensorflow==2.17.0
# pip install \
#     reformer_pytorch optuna-dashboard wandb captum\
#     einops optunahub cmaes \
#     gpustat gputil 

# 3. Các gói còn lại (tránh conda solver)
echo "📦 Installing other packages..."
conda install -y -c conda-forge \
    numpy\
    matplotlib pandas \
    tqdm seaborn optuna\
    argparse torch torchvision torchmetrics \
    scikit-learn einops optunahub cmaes\
    gpustat statsmodels gputil \
    shap ipykernel\
    opencv-python\
    scikit-image\
    pyvis

############## Save environment ##############
echo "💾 Exporting environment to environment.yml..."
conda env export --no-builds > environment.yml

############## Jupyter kernel setup ##############
echo "📝 Setting up Jupyter kernel..."
python -m ipykernel install --user --name=$ENV_NAME --display-name "Python ($ENV_NAME)"

# 5. Alias
grep -q "alias activate_env=" ~/.bashrc || \
  echo "alias activate_env='conda activate $ENV_NAME'" >> ~/.bashrc

echo "🎉 Done. Run: source ~/.bashrc && activate_env"