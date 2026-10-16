#!/bin/bash
#SBATCH --job-name=cwpwm_pipeline
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/share/nesylearning/%u/logs/cwpwm_pipeline.%j.out
# The development-stage pipeline on one GPU: the CPU/CUDA agreement test
# (Requirement 5.5, meant to run once on the GPU host), the causal
# analysis, the six retrained ablation arms at the five frozen seeds, and
# the five-seed replication with determinism verification.
#
# Run setup_xfer.sbatch first. Submit with:
#     sbatch $R/scripts/slurm/run_pipeline.sbatch
# Keep one GPU type for every run you intend to compare: determinism is
# verified on a fixed hardware/software stack, and different GPU models can
# change results in the last few digits. Override with e.g.
#     sbatch --gres=gpu:h100:1 $R/scripts/slurm/run_pipeline.sbatch
set -euo pipefail
source /share/nesylearning/$USER/hazel_cwpwm_env.sh
cd $R
[ -d data/raw/actg175_v3_2 ] || { echo "data/raw/actg175_v3_2 missing; run setup_xfer.sbatch" >&2; exit 1; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

mkdir -p results/hazel
echo "commit $(git rev-parse HEAD), uncommitted files: $(git status --porcelain | wc -l)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee results/hazel/gpu.txt
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())"

echo "== CPU/CUDA agreement test"
CWPWM_DEVICE_AGREEMENT_DIR=results/hazel/device_agreement \
    $PY -m pytest -q -p no:cacheprovider tests/neural/test_determinism.py

echo "== causal analysis"
$PY scripts/run_causal_analysis.py --real-data-path data/raw/actg175_v3_2 --master-seed 20240101

echo "== ablations"
time $PY scripts/run_ablations.py --target-trial-config configs/target_trials/main.yaml \
    --master-seed 20240101 --device cuda

echo "== replication"
time $PY scripts/run_replication.py --device cuda --verify-determinism

echo "CWPWM PIPELINE DONE; copy results/ off scratch"
