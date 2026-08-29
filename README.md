<div align="center">

# RMB

### Reward Model Boosting Mitigates Reward Hacking

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-1F9D8A)](https://huggingface.co/docs/peft/)
[![XGBoost](https://img.shields.io/badge/XGBoost-pairwise-F2B134)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-Apache--2.0-6B7280)](LICENSE)

**Diverse LoRA reward models, trained jointly. A lightweight boosted ranker, trained without stacking leakage.**

<img src="docs/assets/rmb-pipeline.svg" width="100%" alt="Animated RMB pipeline: preference pairs feed diverse LoRA reward models and a pairwise XGBoost aggregator">

</div>

RMB builds a more reliable reward signal for RLHF in two stages:

1. **Diverse reward modeling.** A frozen language-model backbone hosts several LoRA adapters and independent MLP value heads. They learn the Bradley-Terry preference objective jointly, while HSIC discourages dependence between their **preference margins**.
2. **Reward model boosting.** The frozen reward scores become features for a pairwise XGBoost ranker. Each tree corrects residual ranking errors left by the preceding trees.

This repository contains reward-model training, boosted aggregation, evaluation, Best-of-N, and PPO workflows.

## Contents

- [What changed](#what-changed)
- [Method](#method)
- [Quick start on Vulcan](#quick-start-on-vulcan)
- [Training](#training)
- [Outputs](#outputs)
- [Repository map](#repository-map)
- [Troubleshooting](#troubleshooting)

## What changed

The RMB training path has been aligned with the manuscript and the corrected implementation in `lora-boosting`.

| Area | Correct behavior |
|---|---|
| HSIC target | Compute dependence on `r(chosen) - r(rejected)`, never on raw reward levels |
| HSIC estimator | Normalize kernels, use median bandwidths, and average every adapter pair |
| Joint LoRA training | Keep all adapters trainable after PEFT switches the active adapter |
| Gradient checkpointing | Bind backward recomputation to the adapter used by the original forward |
| Validation | Select Stage-1 checkpoints on Unified-Feedback validation by default |
| Stacking split | Train Stage 2 on `40K-heldout`, disjoint from the Stage-1 `40K` stride |
| Distributed extraction | Gather one complete chosen/rejected pair before flattening XGBoost rows |
| Padding | Read the last attended token correctly for either left or right padding |
| Checkpoints | Load both nested PEFT and legacy flat adapter layouts; do not duplicate the frozen backbone |
| XGBoost | Use grouped `rank:pairwise`, pair accuracy early stopping, and save the true best iteration |

The rolling, detached margin buffer stabilizes the HSIC kernel estimate when memory constraints require very small micro-batches. Gradients still flow only through the current micro-batch.

## Method

For preference tuple `(x, y_w, y_l)`, reward model `i` produces the margin

```text
m_i = r_i(x, y_w) - r_i(x, y_l)
```

Stage 1 minimizes

```text
L_stage1 = mean_i[-log sigmoid(m_i)]
           + lambda_HSIC * mean_{i<j}[normalized_HSIC(m_i, m_j)]
```

Minimizing the positive HSIC term reduces statistical dependence while the Bradley-Terry term preserves preference accuracy. Three adapters, `lambda_HSIC=0.1`, LoRA rank 32, LoRA alpha 64, and a two-layer-style 1024-wide reward head match the main paper setup.

Stage 2 learns an additive tree ensemble over the reward vector:

```text
R(x, y) = [r_1(x, y), ..., r_N(x, y)]
F_K(R)  = sum_{k=1..K} f_k(R)
```

The implementation defaults to the paper's `max_depth=5`, `num_round=256`, and L2 coefficient `0.1`. Every XGBoost group is exactly `[chosen, rejected]`.

### Data contract

A pairwise dataset must expose:

```text
input_ids_chosen        attention_mask_chosen
input_ids_rejected      attention_mask_rejected
```

Unified-Feedback modes used by the reproducible path:

| Mode | Purpose | Post-filter indices |
|---|---|---|
| `40K` | Stage-1 LoRA/HSIC training | `0, 20, 40, ...` |
| `40K-heldout` | Stage-2 XGBoost fitting | `1, 21, 41, ...` |
| validation split | checkpoint selection / early stopping | dataset-provided validation |

Do not train Stage 2 on the same rows used by Stage 1. That turns stacking into in-sample fitting and inflates the apparent value of the booster.

## Quick start on Vulcan

All Python work runs through Slurm. Checkpoints, Hugging Face caches, environment files, and logs stay on `$SCRATCH`.

### 1. Build the environment

```bash
cd /home/jiabin/projects/aip-lilimou/jiabin/RMB/RMB
sbatch scripts/setup_env.slurm
```

The setup job loads the verified `StdEnv/2023`, `python/3.11.5`, and `arrow/18.1.0` modules, then installs `requirements-hpc.txt` from the Alliance wheelhouse with `--no-index`.

The historical `requirements.txt` is retained for the original broad environment. New Vulcan runs should use the smaller HPC file.

For gated Hugging Face models, export a token before submission:

```bash
export HF_TOKEN=...
```

`HF_HOME` is a cache directory, not a token. The Slurm scripts set it below `$SCRATCH/rmb`.

### 2. Train diverse reward models

```bash
sbatch scripts/train_hsic_rms.sh
```

Useful overrides are regular environment variables:

```bash
NUM_TRAIN_EPOCHS=3 EVAL_SIZE=2000 \
  sbatch --export=ALL scripts/train_hsic_rms.sh
```

With the defaults, one L40S trains three Gemma-2B LoRA adapters jointly. The effective batch size is:

```text
micro_batch_size * gradient_accumulation_steps * number_of_processes
= 2 * 8 * 1 = 16
```

### 3. Fit the boosted aggregator

Choose a `best_step_*` directory produced by Stage 1:

```bash
export ADAPTER_CHECKPOINT="$SCRATCH/rmb/checkpoints/gemma-2b-it_rmb_hsic_multilora3/logspaper_3adps/best_step_<STEP>_acc=<ACC>"
sbatch --export=ALL scripts/rmb_boosting.sh
```

The default booster is written to:

```text
$SCRATCH/rmb/results/rmb_booster.json
```

### Resource defaults

| Job | GPU | CPU | Memory | Wall time |
|---|---:|---:|---:|---:|
| Environment setup | none | 4 | 16 GB | 1 hour |
| Stage 1: LoRA + HSIC | 1 x L40S | 8 | 64 GB | 24 hours |
| Stage 2: XGBoost | 1 x L40S | 8 | 64 GB | 12 hours |

The scripts use Alliance auto-routing and intentionally do not set a partition.

## Training

### Stage-1 controls

| Variable | Default | Meaning |
|---|---:|---|
| `BASE_MODEL` | `google/gemma-2b-it` | Frozen backbone |
| `DATASET_MODE` | `40K` | Stage-1 subset |
| `NUM_ADAPTERS` | `3` | LoRA/value-head learners |
| `HSIC_LAMBDA` | `0.1` | Dependence penalty |
| `LORA_R` / `LORA_ALPHA` | `32` / `64` | LoRA capacity and scaling |
| `LEARNING_RATE` | `1e-5` | AdamW learning rate |
| `MICRO_BATCH_SIZE` | `2` | Per-device micro-batch |
| `GRADIENT_ACCUMULATION_STEPS` | `8` | Effective-batch multiplier |
| `SEED` | `32` | Dataset, model, and trainer seed |

The first optimizer step includes a fail-fast gradient check. Training stops immediately if any adapter has missing, zero, or non-finite LoRA-B gradients.

### Stage-2 controls

| Variable | Default | Meaning |
|---|---:|---|
| `ADAPTER_CHECKPOINT` | required | Stage-1 `best_step_*` directory |
| `DATASET_MODE` | `40K-heldout` | Leakage-free booster rows |
| `NUM_ROUND` | `256` | Maximum trees |
| `EARLY_STOPPING` | `30` | Pair-accuracy patience |
| `XGB_MAX_DEPTH` | `5` | Tree depth |
| `XGB_REG_LAMBDA` | `0.1` | Leaf L2 regularization |
| `BATCH_SIZE` | `8` | Reward feature extraction batch |

For a cheap pipeline check, pass `--max_train_samples` and `--max_eval_samples` directly to `run_booster_rmb.py` inside a short Slurm job.

## Outputs

A Stage-1 best checkpoint is self-contained except for the frozen base model:

```text
best_step_<STEP>_acc=<ACC>/
|-- adapter_0/
|   |-- adapter_0/adapter_config.json
|   |-- adapter_0/adapter_model.safetensors
|   `-- v_head.bin
|-- adapter_1/
|-- adapter_2/
|-- tokenizer.json
`-- tokenizer_config.json
```

Stage 2 accepts this nested PEFT layout and the older flat `adapter_N/` layout.

The XGBoost model stores:

- `feature_mode=response`
- `best_iteration`
- `best_pair_accuracy`

These attributes make it possible to reject incompatible feature pipelines during downstream inference.

## Evaluation and RLHF

Reward-model evaluation entry points live in `rm_eval/` and `scripts/eval_*.sh`. Downstream workflows are under `rlhf/`:

- `rlhf/bon/`: Best-of-N generation, scoring, selection, and collection.
- `rlhf/ppo/`: PPO with baseline, ensemble, GRM, or RMB rewards.
- `rlhf/data_generation/`: preference and gold-score preparation.

Keep model selection and final OOD reporting separate. Unified-Feedback validation is the default selection set; HHH, MT-Bench, RewardBench, and other OOD suites should remain untouched until final evaluation.

## Repository map

```text
RMB/
|-- reward_models/
|   |-- hsic_rms_train.py       # Stage 1: joint LoRA + margin HSIC
|   |-- run_booster_rmb.py      # Stage 2: grouped pairwise XGBoost
|   |-- grm_utils.py            # multi-value-head wrapper
|   |-- load_datasets.py        # train / held-out split builders
|   `-- reward_trainer.py       # pair collator and base RM losses
|-- rm_eval/                    # reward-model evaluation
|-- rlhf/                       # BoN, PPO, and data generation
|-- scripts/
|   |-- setup_env.slurm
|   |-- train_hsic_rms.sh
|   `-- rmb_boosting.sh
|-- requirements-hpc.txt
`-- README.md
```

## Troubleshooting

| Symptom | Check |
|---|---|
| Only the last adapter changes | The first-step gradient check should report finite norms for every adapter |
| HSIC is always zero | Use at least two adapters and enough directional margins; tiny batches rely on the rolling buffer |
| Reward changes with padding side | Confirm the updated `last_non_pad_indices` path is used |
| XGBoost group-size error | Keep rows interleaved as `[chosen, rejected]`; never gather flattened rows across ranks |
| Booster looks unrealistically strong | Confirm Stage 1 uses `40K` and Stage 2 uses `40K-heldout` |
| Checkpoint cannot load | Point to the outer `best_step_*` directory, not one inner adapter payload |
| Job writes into project/home | Keep `RMB_OUTPUT_ROOT`, `RMB_RESULT_ROOT`, `HF_HOME`, and the venv on `$SCRATCH` |

## Acknowledgments

RMB builds on [Generalizable Reward Modeling](https://github.com/YangRui2015/Generalizable-Reward-Model), [Transformers](https://github.com/huggingface/transformers), [TRL](https://github.com/huggingface/trl), [PEFT](https://github.com/huggingface/peft), [XGBoost](https://github.com/dmlc/xgboost), and ideas from [RLHFlow](https://github.com/RLHFlow/RLHF-Reward-Modeling).

The manuscript is currently under double-blind review. Citation metadata will be added after review.
