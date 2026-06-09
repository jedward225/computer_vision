# Experiment Plan

## Main Comparison

| Model | Type | Input | Output | Metrics |
| --- | --- | --- | --- | --- |
| U-Net | Discriminative | 2D slice | Multi-class mask | Dice, IoU |
| SegResNet | Discriminative | 2D slice | Multi-class mask | Dice, IoU |
| cVAE-Seg | Generative | 2D slice | Multi-class mask | Dice, IoU |
| Conditional Diffusion | Generative | 2D slice + noisy mask | Multi-class mask | Dice, IoU, time |

## Ablations

- 2D vs 2.5D input.
- Binary foreground segmentation vs multi-class segmentation.
- DDIM sampling steps: 5, 10, 25, 50, 100.
- Diffusion loss design: noise MSE only vs noise MSE + DiceCE auxiliary loss.
- Optional data-scale study: 25, 50, 100 patients.

## Visualizations

- Training loss and validation Dice curves.
- At least 10 qualitative test cases: image, ground truth, prediction, overlay.
- Denoising trajectory for 3 cases.
- Uncertainty maps from repeated diffusion samples.
- Failure cases: small tumor miss, cyst/tumor confusion, boundary leakage.

## Risk Control

The minimum complete submission is U-Net baseline, diffusion segmentation, Dice/IoU evaluation, denoising visualization, and discriminative-vs-generative discussion. cVAE, SegResNet, 2.5D, and uncertainty maps are extensions.

