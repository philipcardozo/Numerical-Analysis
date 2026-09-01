#!/bin/bash
#SBATCH --job-name=operator-nb
#SBATCH --output=slurm_logs/notebook_%j.out
#SBATCH --error=slurm_logs/notebook_%j.err
#SBATCH --partition=zubmain
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

source ~/.bashrc
conda activate jax-gpu

cd /local/home/larsaipz/code/2025-ComputationalMathAndAICourse/code/07-sciml

echo "=========================================="
echo "Running operator-learning.ipynb"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=========================================="

python3 -m jupyter nbconvert --to notebook --execute operator-learning.ipynb \
    --ExecutePreprocessor.timeout=7200 \
    --output operator-learning-executed.ipynb

echo ""
echo "=========================================="
echo "Generated files:"
echo "=========================================="
echo "--- Figures ---"
ls -la figures/*.png 2>/dev/null | head -20
echo ""
echo "--- CSVs ---"
ls -la figures/*.csv 2>/dev/null
echo ""
echo "--- Models ---"
ls -la saved_models/ 2>/dev/null

echo ""
echo "Done!"
