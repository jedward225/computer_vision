# Computer Vision Final Project

Generative models for dense prediction on KiTS23 kidney, tumor, and cyst segmentation.

This repository is organized for the 2026 Spring Introduction to Computer Vision final project. The main task is Task 2: semantic segmentation with generative models. The planned core model is a conditional diffusion segmentation model, compared against discriminative U-Net-style baselines and a conditional VAE baseline.

## Project Scope

- Dataset: KiTS23 3D CT volumes, converted into axial 2D or 2.5D slices.
- Main task: multi-class segmentation of background, kidney, tumor, and cyst.
- Additional task view: binary foreground segmentation.
- Main model: conditional diffusion segmentation with DDPM training and DDIM sampling.
- Baselines: U-Net, SegResNet-style residual segmentation, and conditional VAE segmentation.
- Analysis: sampling-step ablation, 2D vs 2.5D input, uncertainty estimation, and failure cases.

## Repository Layout

```text
src/
  configs/      Experiment YAML files.
  data/         KiTS23 preprocessing, splits, and dataset loaders.
  models/       U-Net, SegResNet, cVAE, and diffusion U-Net definitions.
  diffusion/    Noise schedules, DDPM training utilities, DDIM/DDPM samplers.
  engine/       Training, evaluation, inference, and figure-generation entrypoints.
  utils/        Metrics, visualization, logging, seeds, and experiment helpers.
scripts/        Shell wrappers for common runs.
docs/           Method notes and experiment plan.
results/        Figures and result tables used in the report.
checkpoints/    Local model weights, not committed to Git.
logs/           Local TensorBoard/W&B/plain logs, not committed to Git.
reference/      External reference code, not committed to Git.
```

## Data

The KiTS23 dataset is not committed. A typical local layout is:

```text
data/
  kits23/
    kits23_repo/
      dataset/
      case_00000/
        imaging.nii.gz
        segmentation.nii.gz
      case_00001/
        imaging.nii.gz
        segmentation.nii.gz
```

Download with the official package:

```bash
bash scripts/download_kits23.sh --num-cases 100
```

The final experiments use the downloaded KiTS23 cases with train/val/test split by patient ID.

## Planned Commands

```bash
# Environment
bash scripts/setup_env.sh
conda activate cv-final

# Preprocess 100 KiTS23 volumes into 2D/2.5D slices
bash scripts/download_kits23.sh --num-cases 100
bash scripts/prepare_data.sh --max-cases 100

# Train baselines
bash scripts/train_unet.sh
bash scripts/train_segresnet.sh

# Train generative models
bash scripts/train_cvae.sh
bash scripts/train_diffusion.sh  # segmentation-oriented x0 diffusion
bash scripts/train_diffusion_25d.sh

# Evaluate and generate report figures
bash scripts/evaluate_all.sh
bash scripts/infer_diffusion.sh
bash scripts/make_figures.sh
```

The scripts run preprocessing, training, evaluation, and report-figure generation once the environment and data are available.

For tmux-friendly commands, see `scripts/tmux_commands.md`.

## Report

The LaTeX report template is in `report.tex`. It currently contains the final-project section structure and placeholder tables for the planned experiments.
