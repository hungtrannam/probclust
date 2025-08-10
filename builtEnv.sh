# =======================================
# Author: Hung Tran-Nam
# Email: namhung34.info@gmail.com
# Repo: https://github.com/hungtrannam/probclust
# =======================================
# File: builtEnv.sh
# Description: Script to set up a Python virtual environment with required packages

#!/bin/bash

set -e  # Exit script on any error

echo "🟢 Updating the system..."
sudo apt update -y && sudo apt upgrade -y

############## System Dependencies ##############
echo "Installing system dependencies..."

############## Create virtual environment ##############
if [ ! -d ".venv" ]; then
    echo "Creating a virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists."
fi

############## Activate virtual environment ##############
echo "Activating the virtual environment..."
source .venv/bin/activate

############## Upgrade pip ##############
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

############## Install packages ##############
echo "Installing required packages..."
pip install --no-cache-dir \
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
echo "Exporting installed packages to requirements.txt..."
pip freeze > requirements.txt

############## Jupyter kernel setup ##############
echo "Setting up Jupyter kernel..."
python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"

############## Bash alias utility ##############
echo "Adding alias for quick env activation..."
if ! grep -q "alias activate_env=" ~/.bashrc; then
    echo "alias activate_env='source $(pwd)/.venv/bin/activate'" >> ~/.bashrc
fi

echo "Setup complete. Run: source ~/.bashrc"
