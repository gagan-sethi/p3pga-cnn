# MO-P3PGA for Pareto-optimal CNN design — rice leaf disease

- Generated: 2026-11-14 08:54:27  |  preset `paper`  |  plan `all`  |  GPU `Tesla T4`
- Dataset: `/kaggle/input/datasets/nirmalsankalana/rice-leaf-disease-image` — 4752 images, 4 classes (Bacterial_Blight, Blast, Brown_Spot, Tungro)
- Exact-duplicate removal before splitting: 5932 -> 4752 images (1180 copies removed, 0 label-conflicting groups dropped)
- Group-aware split: 509 original photos (rotated/flipped/re-encoded copies grouped at thumbnail similarity >= 0.995); split 60%/20%/20% by photo; train 2832 images from 306 photos, validation 924 images from 102 photos, test 101 images (one per photo)
- Near-duplicate audit (rotation/flip-invariant 32x32 thumbnails): test images whose closest training image has similarity >= 0.995: 0, >= 0.98: 14
- Exact duplicates: 0/101 test images also appear in training
- Splits: 2832 train / 924 val / 101 test (stratified, seed 42)
- Objectives (minimised): error, flops
- Search space: 24 decision variables, identical for all optimisers
- Protocol: 7 optimisers x 10 runs x 240 evaluations = 16800 budgeted evaluations (16560 of them needed training; the rest were repeats served from the shared memo)

## Search-phase outcome

- Reference front: 18 solutions pooled from 495 archive members
- Hypervolume ranking (best first): MO-P3PGA-II (1.70), MOGWO (2.90), GDE3 (3.80), MOEA/D (3.90), MO-P3PGA (4.40), Random Search (5.40), NSGA-II (5.90)
- MO-P3PGA-II mean HV = 1.1346 ± 0.0093, average rank 1.70
- Friedman on HV: chi2 = 26.314, p = 0.0001945 (N = 10 runs, k = 7)
- Wilcoxon (Holm-corrected) significant wins for MO-P3PGA-II: 4/6 (MO-P3PGA, MOEA/D, NSGA-II, Random Search)
- Contributions to the reference front: MO-P3PGA=3, MO-P3PGA-II=4, NSGA-II=0, MOEA/D=4, GDE3=2, MOGWO=5, Random Search=0

## Final models (held-out test split)

- Headline model `MO-P3PGA-II (accuracy)`: accuracy **96.04%** (95% CI 92.08–99.01), macro-F1 **93.96%** (95% CI 87.43–98.68)
- Cost: 0.126 M parameters, 114 MMac, 2.68 ms/image at batch 1 on Tesla T4
- MO-P3PGA-II accuracy point: macro-F1 91.83% ± 1.87 over 3 seeds, 0.126 M parameters, 114 MMac
- MO-P3PGA-II knee point: macro-F1 86.71% ± 2.46 over 3 seeds, 0.004 M parameters, 4 MMac
- MO-P3PGA-II efficiency point: macro-F1 86.77% ± 4.92 over 3 seeds, 0.002 M parameters, 4 MMac
- Best model by test macro-F1 overall: `shufflenet_v2_x1_0 (ImageNet)` (97.86%)
- McNemar of `MO-P3PGA-II (accuracy)` against the other 14 models (Holm-corrected): significantly better than 0, significantly worse than 0, no significant difference from 14 (Random Search (knee), MOEA/D (knee), GDE3 (knee), MOGWO (knee), MO-P3PGA-II (knee), shufflenet_v2_x1_0 (ImageNet), resnet18 (ImageNet), mobilenet_v2 (ImageNet), MO-P3PGA-II (efficiency), NSGA-II (knee), MO-P3PGA (knee), mobilenet_v3_small (ImageNet), efficientnet_b0 (ImageNet), Hand-designed CNN)

## Protocol validation

- Proxy fidelity: Spearman rho = 0.545 between proxy macro-F1 and full-training test macro-F1 (n = 12)

## Architecture discovered (headline model)

```json
{
  "n_blocks": 4,
  "base_filters": 24,
  "growth": 1.75,
  "block_type": "mbconv",
  "kernel": 3,
  "use_se": true,
  "se_ratio": 0.25,
  "activation": "gelu",
  "stem_stride": 1,
  "head_pool": "max",
  "dense_units": 128,
  "conv_dropout": 0.0,
  "dense_dropout": 0.115,
  "optimizer": "adam",
  "lr": 0.004254,
  "weight_decay": 8.612e-06,
  "batch_size": 96,
  "label_smooth": 0.0745,
  "aug_strength": 0.2445,
  "mix_alpha": 0.2217,
  "mix_mode": "mixup",
  "p_erase": 0.437,
  "stage_depth": 1,
  "expand_ratio": 3
}
```

## Artefacts

- Tables: `/kaggle/working/outputs_rice/tables/` (CSV + LaTeX)
- Figures: `/kaggle/working/outputs_rice/figures/` (PNG 300 dpi + PDF)
- Checkpoints: `/kaggle/working/p3pga_state_rice/` (delete to force a clean re-run)
- Session wall-clock used: 4h46m07s