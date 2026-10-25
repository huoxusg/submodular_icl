# Submodular In-Context Learning (ICL)

## Repository Structure

```
datasets/     raw train/test JSON + preprocessing notebook
retrieval/    stage 1: embeddings and submodular exemplar selection
icl/          stage 2: LoRA fine-tuning, ICL evaluation, metrics, statistics
scripts/      standalone helper scripts
results/      committed experiment artifacts (stats/, tables/)
```

Generated locally and never committed: `npy_files/`, `peft_models/`, `ragged_datasets/`, `results/logs/`.

## Local Configuration

Paths and secrets are read from environment variables, so no machine-specific value lives in the code:

```bash
export SUBICL_LLM_DIR=/path/to/huggingface/llm/models/     # icl/main.py: LLM_DIR
export SUBICL_MODEL_CACHE_DIR=/path/to/peft_models        # retrieval/constants.py: cache_dir
export HF_TOKEN=...                                        # datasets/preprocess.ipynb only
```

Or `cp scripts/local_env.example.sh scripts/local_env.sh && source scripts/local_env.sh` (the copy is git-ignored).

## Publishing the Public Snapshot

This working copy holds both the private project and its public subset:

| Remote | Branch | Content |
|---|---|---|
| `gitlab` | `private` | everything (code, data, logs, plotting, experiments) |
| `origin` (GitHub) | `main` | code + README only |

```bash
git push gitlab private:main     # daily development
bash scripts/publish_public.sh   # export the public snapshot to GitHub
```

`publish_public.sh` builds a commit from the whitelist (`README.md`, `.gitignore`, `datasets/preprocess.ipynb`, `icl/`, `retrieval/`, `scripts/`) plus any overrides in `scripts/public_overrides/`; datasets, `results/`, logs and plotting never leave the private repo.



## First step - retrieval

`python -m retrieval.main`

### Preparation

Generating the embedding vectors of queries in train / test datasets, using different retriever models.

### 1.1 Exemplar Summary


### 1.2 Exemplar Selection


## Second step - ICL

the process of in-context learning after generating demonstration examples

addition fine-tuning procedure between pre-trained model and evaluation

## Statistical Comparison

### 10-fold paired t-test on existing results

Use this mode to compare two methods with paired folds from existing `*_results.json` files:

```bash
python -m icl.main \
	--stats_mode cv_ttest \
	--file_dir ragged_datasets \
	--pattern1 random_similar_100_3 \
	--pattern2 dual_reg_100_30_3 \
	--n_folds 10 \
	--seed 42 \
	--alpha 0.05 \
	--stats_output_dir results/stats \
	--stats_output_prefix random_vs_dual_10fold
```

Output includes per-metric `Mean Diff`, `Std Diff`, `t`, `p`, and `95% CI`.
Structured outputs are saved to both JSON and CSV in `--stats_output_dir`.

Method note: this is posterior fold-splitting on fixed prediction files, not full retrain/reinference cross-validation.

### Legacy bootstrap mode (deprecated)

```bash
python -m icl.main \
	--stats_mode bootstrap \
	--file_dir ragged_datasets \
	--pattern1 random_similar_100_3 \
	--pattern2 dual_reg_100_30_3 \
	--n_bootstraps 1000
```