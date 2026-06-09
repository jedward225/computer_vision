# KiTS23 Task 2 Experiment Analysis

## Setup

The formal Task 2 run uses KiTS23 only. We selected 100 KiTS23 training cases with smaller NIfTI volumes to fit the 50 GB AutoDL data disk while keeping the assignment-recommended 50-100 patient scale. The dataset was split by patient ID into 70 training cases, 15 validation cases, and 15 test cases.

Preprocessing follows the assignment requirements: each 3D CT volume is converted into axial 2D slices, clipped with a CT HU window of center 40 and width 400, normalized to [0, 1], resized to 256 x 256, and paired with the corresponding segmentation mask. Labels are simplified to binary foreground segmentation, where foreground is kidney, tumor, or cyst. This binary setting is explicitly allowed by the assignment.

The processed split sizes are:

| Split | Cases | Slices |
| --- | ---: | ---: |
| Train | 70 | 17,047 |
| Val | 15 | 3,521 |
| Test | 15 | 3,755 |

Training was run on an NVIDIA RTX PRO 6000 Blackwell Server Edition GPU. The discriminative baseline is a 2D U-Net. The generative model is a conditional diffusion segmentation model that denoises a noisy mask conditioned on the CT image.

## Quantitative Results

| Model | Type | Sampling steps | Test Dice | Test IoU | ms / image |
| --- | --- | ---: | ---: | ---: | ---: |
| U-Net | Discriminative | 1 | 0.8877 | 0.7981 | 0.87 |
| Conditional diffusion | Generative | 5 | 0.1682 | 0.0918 | 16.88 |
| Conditional diffusion | Generative | 10 | 0.1683 | 0.0919 | 33.63 |
| Conditional diffusion | Generative | 25 | 0.1689 | 0.0922 | 84.08 |

The U-Net baseline is clearly stronger on this binary KiTS23 setup. It reached a best validation Dice of 0.9307 and a test Dice of 0.8877. The diffusion model reached a best validation Dice of 0.1978 and a test Dice of about 0.168.

Increasing DDIM sampling steps from 5 to 25 did not materially improve Dice or IoU, but it increased inference time nearly linearly. This is useful for the required sampling-step comparison: more denoising steps made inference slower without improving segmentation quality under the current training setup.

## Interpretation

The current conditional diffusion implementation successfully follows the generative dense-prediction formulation: the mask is treated as the generated variable, Gaussian noise is added during training, and inference starts from a random mask and iteratively denoises it conditioned on the CT image. However, the result quality is poor compared with U-Net.

The main reason is that this segmentation task is highly foreground sparse, while the current mask diffusion model starts inference from pure random mask noise. The model learns low noise MSE, but this objective alone does not reliably produce spatially coherent foreground masks after iterative sampling. The auxiliary DiceCE term helps align training with segmentation quality, but with the current architecture and schedule it is not enough to match the discriminative baseline.

The result supports the expected generative-vs-discriminative tradeoff. U-Net directly learns the image-to-mask mapping and is much faster and more accurate. Diffusion gives an explicit generation process and supports denoising trajectory visualization, but it has higher training and inference cost and is harder to optimize for dense segmentation quality.

## Report Usage

Use the U-Net result as the strong discriminative baseline. Use the diffusion result to satisfy the generative modeling requirement, the denoising-process visualization requirement, and the sampling-step speed comparison. In the report, the diffusion result should be discussed honestly as a negative or limited result rather than presented as competitive with U-Net.

Relevant artifacts:

| Artifact | Path |
| --- | --- |
| Evaluation CSV | `results/tables/binary_100case_evaluation.csv` |
| Formal training curves | `results/figures/formal/` |
| U-Net qualitative panels | `results/figures/qualitative/unet_binary_100case/` |
| Diffusion qualitative panels | `results/figures/qualitative/diffusion_binary_100case/` |
| Diffusion denoising trajectories | `results/figures/qualitative/diffusion_binary_100case/*_denoising.png` |

