# Sourced by every Hazel job for this project. Everything lives in scratch:
# home has a 15 GB / 10K-file quota, and the torch environment alone is
# about 10K files. Scratch files unused for 30 days are deleted, so copy
# results/ off when a run finishes.
export S=/share/nesylearning/$USER
export R=$S/ontology-clinical-matching
export UV=$S/uv/bin/uv
export UV_PYTHON_INSTALL_DIR=$S/uv/python UV_CACHE_DIR=$S/uv/cache PIP_CACHE_DIR=$S/pip_cache
export HF_HUB_CACHE=$S/hf_cache            # the HF token stays in ~/.cache/huggingface
export PYTHONHASHSEED=0
# Deterministic cuBLAS kernels; without it GRU backward is nondeterministic on CUDA.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PY=$R/.venv/bin/python
mkdir -p $S/logs
