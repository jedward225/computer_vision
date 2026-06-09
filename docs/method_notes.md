# Method Notes

## Conditional Diffusion Segmentation

The segmentation mask is treated as the generated variable. During training, a clean one-hot mask is diffused into a noisy mask. The model receives the CT image, the noisy mask, and the diffusion timestep, then predicts the injected noise or the clean mask logits.

Training objective:

```text
L = L_noise(epsilon_pred, epsilon) + lambda * L_seg(mask_logits, mask_gt)
```

The auxiliary segmentation loss keeps the model aligned with the task metric and should help small structures such as tumors and cysts.

## Sampling

DDPM sampling is the faithful reverse process but can be slow. DDIM sampling uses fewer deterministic steps with the same trained model, making it suitable for the assignment's speed-accuracy comparison.

## Uncertainty

Diffusion can generate multiple masks for one CT slice. Repeated sampling gives a distribution over segmentations. The variance or entropy of this distribution can reveal uncertain boundaries and ambiguous lesion regions.

