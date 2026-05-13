#!/bin/bash
#SBATCH --job-name=molcraft-env-setup
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=setup_env_%j.log
#SBATCH --error=setup_env_%j.log

set -e

echo "=== molcraft-water env setup ==="
echo "Node: $(hostname)"
echo "Time: $(date)"

# Remove old env if it exists
if mamba env list | grep -q "^molcraft-water "; then
    echo "Removing existing molcraft-water env..."
    mamba env remove -n molcraft-water -y
fi

# Remove any ~/.local torch that would shadow the conda-env install
echo "Removing any user-level torch from ~/.local..."
pip uninstall torch -y 2>/dev/null || true

# Create the environment
echo "Creating environment from YAML..."
mamba env create -f MolJO/environment_water_guidance.yml

# Verify
echo "=== Verifying install ==="
conda run -n molcraft-water python -c "
import torch, torch_cluster
print('torch:', torch.__version__)
print('torch path:', torch.__file__)
x = torch.randn(10, 3).cuda()
b = torch.zeros(10, dtype=torch.long).cuda()
ei = torch_cluster.knn_graph(x, k=3, batch=b)
print('knn CUDA ok, edge_index shape:', ei.shape)
"

echo "=== Done: $(date) ==="
