# K-Token Merging

Open-source release of the method described in the paper **"Compressing Sequences in the Latent Embedding Space: $K$-Token Merging for Large Language Models."**

This repository is built from the original `Token Compression` research folder, but reorganized for public use. The focus here is the core method from the paper: a lightweight encoder that merges each contiguous block of `K` token embeddings into a single latent embedding, followed by a LoRA-adapted LLM that reads the compressed prefix while still generating in the original vocabulary.

The paper reports that **K-Token Merging reaches up to 75% input-length reduction with minimal degradation**, and the teaser figure highlights the `K=4` case on Textualized Tree with only a `1.59%` accuracy drop while reducing input length by `75%`.

![K-Token Merging teaser](assets/figures/teaser.png)

The model structure follows the paper directly: compression happens only on the input side during prefill, while generation stays in the original token space.

![K-Token Merging model overview](assets/figures/model_overview.png)

## What is included

- Shared method code for the average-initialized encoder, prompt compression, LoRA loading, and evaluation utilities.
- Reorganized benchmark runners for the three tasks used in the paper:
  - Textualized Tree
  - Amazon Reviews
  - CommitPackFT
- A cleaned embedding-extraction utility.
- The original paper source and figures under [`paper/`](paper).

## What is intentionally excluded

This release does **not** include:

- LLMLingua2 code
- LoseLess Token Compression / LTSC baseline code
- SelectiveContext baseline code
- pretrained checkpoints
- training runs, TensorBoard logs, optimizer dumps, or other local experiment artifacts

The goal is to open-source **K-Token Merging itself**, not the full internal comparison workspace.

## Project layout

```text
K-Token-Merging/
├── assets/figures/              # README-ready figures exported from the paper
├── paper/                       # original paper source copied from the research folder
├── scripts/
│   ├── amazon_reviews/
│   ├── commitpackft/
│   ├── textualized_tree/
│   └── utils/
└── src/k_token_merging/         # shared method implementation
```

This structure mirrors the paper:

- `src/k_token_merging/` contains the reusable method components from the **Method** and **Implementation Details** sections.
- `scripts/textualized_tree`, `scripts/amazon_reviews`, and `scripts/commitpackft` map to the three datasets in the **Datasets & Tasks** section.
- `paper/` preserves the original manuscript and figure sources used to describe the method.

## Installation

```bash
pip install -e .
```

Or install the direct dependencies:

```bash
pip install -r requirements.txt
```

## Preparing embeddings

The original code uses a cached embedding table from the base model. This release keeps that workflow as a utility script:

```bash
python scripts/utils/extract_embeddings.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --output-file artifacts/qwen2.5_0.5b_embeddings_id_full.pkl
```

## Running the benchmarks

### 1. Textualized Tree

Generate synthetic tree files:

```bash
python scripts/textualized_tree/generate_data.py \
  --max-depth 4 \
  --max-nodes 30 \
  --min-children 1 \
  --max-children 3 \
  --num-trees 1000 \
  --save-dir data/tree_data_large
```

Train:

```bash
python scripts/textualized_tree/run.py train \
  --tree-data-root data \
  --embedding-file artifacts/qwen2.5_0.5b_embeddings_id_full.pkl \
  --merge-factor 4 \
  --output-dir outputs/textualized_tree
```

Evaluate:

```bash
python scripts/textualized_tree/run.py evaluate \
  --tree-data-root data \
  --stage x3large \
  --embedding-file artifacts/qwen2.5_0.5b_embeddings_id_full.pkl \
  --merge-factor 4 \
  --checkpoint-dir outputs/textualized_tree/x3large/step_last
```

### 2. Amazon Reviews

Train:

```bash
python scripts/amazon_reviews/run.py train \
  --embedding-file artifacts/qwen2.5_0.5b_embeddings_id_full.pkl \
  --merge-factor 4 \
  --output-dir outputs/amazon_reviews
```

Evaluate:

```bash
python scripts/amazon_reviews/run.py evaluate \
  --embedding-file artifacts/qwen2.5_0.5b_embeddings_id_full.pkl \
  --merge-factor 4 \
  --checkpoint-dir outputs/amazon_reviews/Amazon_Fashion/step_last
```

### 3. CommitPackFT

Train:

```bash
python scripts/commitpackft/run.py train \
  --embedding-file artifacts/qwen2.5_0.5b_embeddings_id_full.pkl \
  --merge-factor 2 \
  --output-dir outputs/commitpackft
```

Evaluate:

```bash
python scripts/commitpackft/run.py evaluate \
  --embedding-file artifacts/qwen2.5_0.5b_embeddings_id_full.pkl \
  --merge-factor 2 \
  --checkpoint-dir outputs/commitpackft/python/step_last
```

## Method summary

Following the paper:

1. Split the input prompt into contiguous `K`-token blocks.
2. Map each block of token ids to base-model embeddings.
3. Merge each block with a lightweight three-layer MLP plus residual mean pooling.
4. Feed the compressed prefix into a LoRA-adapted LLM.
5. Compute training loss only on the uncompressed answer tokens.

This preserves standard generation while reducing the effective input length during prefill.

## Notes

- The released runners are a cleaned open-source extraction of the original research code, not a byte-for-byte export of the internal workspace.
- The original paper material remains available in [`paper/`](paper), including the LaTeX source in [paper/acl_latex.tex](paper/acl_latex.tex).
- The README figures were exported from the original paper figures under `paper/Figure/`.
