# tmux Commands

Run these commands from the repository root: `/home/jiajun/CV`.

## Environment

```bash
tmux new -s cv-final
cd /home/jiajun/CV
bash scripts/setup_env.sh
conda activate cv-final
```

If `conda activate` is not available inside tmux:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cv-final
```

Check GPU visibility:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## KiTS23 Download

```bash
bash scripts/download_kits23.sh --num-cases 100
```

The expected layout after download is:

```text
data/kits23/dataset/case_00000/imaging.nii.gz
data/kits23/dataset/case_00000/segmentation.nii.gz
```

## First Pipeline Run

```bash
bash scripts/prepare_data.sh
bash scripts/train_unet.sh
bash scripts/evaluate_all.sh
```

## Full Runs

```bash
bash scripts/train_unet.sh
bash scripts/train_segresnet.sh
bash scripts/train_cvae.sh
bash scripts/train_diffusion.sh
bash scripts/train_diffusion_25d.sh
bash scripts/evaluate_all.sh
bash scripts/infer_diffusion.sh
bash scripts/make_figures.sh
```
