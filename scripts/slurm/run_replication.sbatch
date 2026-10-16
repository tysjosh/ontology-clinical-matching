#!/bin/bash
#SBATCH --job-name=cwpwm_repl
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:55:00
#SBATCH --output=/share/nesylearning/%u/logs/cwpwm_repl.%j.out
# Job 2 of 2 (runs in parallel with run_causal_ablations.sbatch): the
# five-seed replication with determinism verification (each seed run twice,
# hashes compared). Uses the 2-hour short_gpu QOS on idle partner A100s.
# It writes only results/replication/ and results/_determinism_rerun/, so it
# does not touch job 1's outputs.
set -euo pipefail
source /share/nesylearning/$USER/hazel_cwpwm_env.sh
cd $R
[ -d data/raw/actg175_v3_2 ] || { echo "data/raw/actg175_v3_2 missing; run setup_xfer.sbatch" >&2; exit 1; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

mkdir -p results/hazel
echo "commit $(git rev-parse HEAD), uncommitted files: $(git status --porcelain | wc -l)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee results/hazel/gpu_repl.txt

echo "== replication"
time $PY scripts/run_replication.py --device cuda --verify-determinism

echo "CWPWM REPL DONE"
