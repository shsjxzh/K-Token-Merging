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

## Project layout

```text
K-Token-Merging/
├── assets/figures/              # README-ready figures exported from the paper
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

We need a cached embedding table from the base model. It can be produced by the following utility script:

```bash
python scripts/utils/extract_embeddings.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --output-file artifacts/qwen2.5_0.5b_embeddings_id_full.pkl
```

## Running the benchmarks

### 1. Textualized Tree

Generate one dataset manually:

```bash
python scripts/textualized_tree/generate_data.py \
  --max-depth 4 \
  --max-nodes 30 \
  --min-children 1 \
  --max-children 3 \
  --num-trees 1000 \
  --save-dir data/tree_data_large \
  --stage-name large
```

Generate the full curriculum used by the tree benchmark:

```bash
python scripts/textualized_tree/generate_curriculum_datasets.py \
  --output-root data \
  --stages small xsmall medium xmedium large x3large \
  --num-trees 1000 \
  --write-summary
```

This creates the stage directories expected by the runner:

```text
data/
├── tree_data_small/
├── tree_data_xsmall/
├── tree_data_medium/
├── tree_data_xmedium/
├── tree_data_large/
└── tree_data_x3large/
```

Each stage directory contains:

- `tree_*.json`
- `train_file_<stage>.csv`
- `test_file_<stage>.csv`

The built-in curriculum stage settings are taken from the original research configs:

| Stage | Max Depth | Max Nodes | Min Children | Max Children |
| --- | ---: | ---: | ---: | ---: |
| `small` | 2 | 3 | 0 | 2 |
| `xsmall` | 3 | 5 | 0 | 2 |
| `medium` | 3 | 10 | 0 | 3 |
| `xmedium` | 4 | 15 | 1 | 3 |
| `large` | 4 | 30 | 1 | 3 |
| `x3large` | 4 | 150 | 3 | 5 |

Train across the full curriculum:

```bash
python scripts/textualized_tree/run.py train \
  --tree-data-root data \
  --stages small xsmall medium xmedium large x3large \
  --embedding-file artifacts/qwen2.5_0.5b_embeddings_id_full.pkl \
  --gpu-ids 0 1 2 3 \
  --merge-factor 4 \
  --grad-accum-steps 4 \
  --output-dir outputs/textualized_tree
```

This follows the original project’s multi-GPU design: pass a GPU id list with `--gpu-ids`, and the runner will launch one worker per GPU with `mp.spawn` and `DistributedDataParallel`. `--grad-accum-steps` controls gradient accumulation inside each worker.

Evaluate one stage:

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
  --gpu-ids 0 1 2 3 \
  --merge-factor 4 \
  --grad-accum-steps 4 \
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
  --gpu-ids 0 1 2 3 \
  --merge-factor 2 \
  --grad-accum-steps 4 \
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
