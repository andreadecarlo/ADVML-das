#!/bin/bash
#SBATCH --job-name=mult_dataset
#SBATCH --partition=edu-short
##SBATCH --time=02:00:00
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=.out/mult_dataset_%j.out

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p datasets

# Load any necessary modules (adjust based on your cluster setup)
# module load python/3.9  # Uncomment and adjust if needed

# Activate virtual environment if using one (adjust path as needed)
# source venv/bin/activate  # Uncomment if using venv
# conda activate your_env  # Uncomment if using conda

# Check if uv is available and use it, otherwise use python directly
if command -v uv &> /dev/null; then
    echo "Using uv to run script..."
    uv run python scripts/generate_multiplication_dataset.py
else
    echo "Using python directly..."
    # Set Python path
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    python scripts/generate_multiplication_dataset.py
fi

echo ""
echo "Job completed at: $(date)"
echo "Check output files in datasets/ directory"
