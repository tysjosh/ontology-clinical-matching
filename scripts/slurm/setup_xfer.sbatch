#!/bin/bash
#SBATCH --job-name=cwpwm_setup
#SBATCH --partition=xfer
#SBATCH --qos=xfer
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/share/nesylearning/%u/logs/cwpwm_setup.%j.out
# Downloads only (xfer partition): Python 3.11 via uv, the clone, the venv
# with the pinned dev/neural/boosted extras, and the protected ACTG175 data
# from the private HF dataset repo. Safe to rerun: each step skips work
# already done, and an existing clone is fast-forwarded.
set -euo pipefail
source /share/nesylearning/$USER/hazel_cwpwm_env.sh
cd $S
[ -x $UV ] || curl -LsSf https://astral.sh/uv/0.8.17/install.sh | env UV_INSTALL_DIR=$S/uv/bin UV_NO_MODIFY_PATH=1 sh
$UV python install 3.11
if [ -d $R ]; then
    git -C $R pull -q --ff-only
else
    GIT_TERMINAL_PROMPT=0 git clone -q --branch neural-pipeline/solvers-and-adjustment-set \
        https://github.com/tysjosh/ontology-clinical-matching.git $R
fi
cd $R
git log --oneline -1
[ -x $PY ] || $UV venv --python 3.11 --seed .venv -q
$PY -m pip install -q --upgrade pip
$PY -m pip install -q -e ".[dev]" huggingface_hub
$PY -c "import torch, xgboost, sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'xgboost', xgboost.__version__)"

# Protected data (docs/data_access.md): owner-only permissions after download.
$PY -c "from huggingface_hub import snapshot_download; snapshot_download('olukotunjosh/actg175-protected-data', repo_type='dataset', local_dir='.', allow_patterns=['data/raw/*'])"
chmod -R go-rwx data/raw
ls data/raw/actg175_v3_2
echo "CWPWM SETUP DONE"
