#!/bin/bash
#SBATCH --job-name=cwpwm_ablate
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:55:00
#SBATCH --output=/share/nesylearning/%u/logs/cwpwm_ablate.%j.out
# Job 1 of 2 (runs in parallel with run_replication.sbatch): the CPU/CUDA
# agreement test (Requirement 5.5, meant to run once on the GPU host), the
# causal analysis, and the six retrained ablation arms at the five frozen
# seeds. Uses the 2-hour short_gpu QOS on idle partner A100s.
#
# Keep one GPU type (A100) for every run you intend to compare: determinism
# is verified on a fixed hardware/software stack.
set -euo pipefail
source /share/nesylearning/$USER/hazel_cwpwm_env.sh
cd $R
[ -d data/raw/actg175_v3_2 ] || { echo "data/raw/actg175_v3_2 missing; run setup_xfer.sbatch" >&2; exit 1; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

mkdir -p results/hazel
echo "commit $(git rev-parse HEAD), uncommitted files: $(git status --porcelain | wc -l)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee results/hazel/gpu_ablate.txt
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())"

echo "== CPU/CUDA agreement test"
CWPWM_DEVICE_AGREEMENT_DIR=results/hazel/device_agreement \
    $PY -m pytest -q -p no:cacheprovider tests/neural/test_determinism.py

echo "== causal analysis"
time $PY scripts/run_causal_analysis.py --real-data-path data/raw/actg175_v3_2 --master-seed 20240101

echo "== ablations"
time $PY scripts/run_ablations.py --target-trial-config configs/target_trials/main.yaml \
    --master-seed 20240101 --device cuda

echo "CWPWM ABLATE DONE"
