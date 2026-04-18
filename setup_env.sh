#!/bin/bash

set -e

echo ">>> Creating virtual environment with Python 3.10..."
uv venv venv --python=3.10

echo ">>> Activating virtual environment..."
source venv/bin/activate

echo ">>> Installing PyTorch 2.8.0 with CUDA 12.8..."
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

echo ">>> Installing PyTorch Geometric..."
uv pip install torch_geometric

echo ">>> Installing PyG extensions..."
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

echo ">>> Installing torch-spatiotemporal and matplotlib..."
uv pip install torch-spatiotemporal matplotlib

echo ">>> Done! Environment ready."
