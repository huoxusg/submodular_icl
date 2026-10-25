# 复制为 scripts/local_env.sh（已被 .gitignore 忽略）后再 `source` 它。
# Copy to scripts/local_env.sh (git-ignored) and `source` it before running.
export SUBICL_LLM_DIR="/path/to/huggingface/llm/models/"
export SUBICL_MODEL_CACHE_DIR="/path/to/retriever/model/cache"
# 仅 datasets/preprocess.ipynb 需要；或改用 `huggingface-cli login`
export HF_TOKEN=""
