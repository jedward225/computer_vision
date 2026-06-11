# tmux Commands

Run from `/home/jiajun/CV`.

## Environment

```bash
tmux new -s cv-final
cd /home/jiajun/CV
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cv-final
nvidia-smi
```

## Tonight Rescue Run

The old processed data used the wrong physical slice axis. The current code writes true axial slices to:

```text
data/processed/kits23_axial_slices
```

Prepare data first. Use 100 cases if it finishes in a reasonable time; fallback to 50 only if needed.

```bash
mkdir -p logs/run
bash scripts/prepare_data.sh --max-cases 100 2>&1 | tee logs/run/prepare_axial_100.log
```

Check the new split:

```bash
cat data/processed/kits23_axial_slices/metadata.json
```

Start the main parallel run. This launches six jobs:

- GPU 0: U-Net 2D
- GPU 1: SegResNet 2D
- GPU 2: cVAE 2D
- GPU 3: x0 diffusion 2D
- GPU 4: U-Net 2.5D optional
- GPU 5: SegResNet 2.5D optional

```bash
bash scripts/train_axial_parallel.sh
```

Watch logs:

```bash
tail -f logs/run/diffusion_axial_2d_x0.out
tail -f logs/run/segresnet_axial_2d.out
```

## Evaluation

Main four-model table:

```bash
bash scripts/evaluate_all.sh
```

Extended table if 2.5D also finishes:

```bash
bash scripts/evaluate_axial_extended.sh
```

Figures:

```bash
bash scripts/make_figures.sh
bash scripts/infer_diffusion.sh
```

## Second Run If Needed

If only one extra training chance remains, do not rerun everything. Use it for x0 diffusion only:

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/train_diffusion_x0.sh 2>&1 | tee logs/run/diffusion_axial_2d_x0_rerun.out
```

If 2D diffusion is acceptable but 2.5D baselines look much better, use the story in the report: 3D context matters more than extra diffusion steps.
