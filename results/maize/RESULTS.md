# MO-P3PGA for Pareto-optimal CNN design — maize (corn) leaf disease

- Generated: 2026-11-13 04:57:11  |  preset `paper`  |  plan `analysis`  |  GPU `Tesla T4`
- Dataset: `/kaggle/input/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset/data` — 4188 images, 4 classes (Blight, Common_Rust, Gray_Leaf_Spot, Healthy)
- Exact duplicates: 0/629 test images also appear in training
- Splits: 2931 train / 628 val / 629 test (stratified, seed 42)
- Objectives (minimised): error, flops
- Search space: 24 decision variables, identical for all optimisers
- Protocol: 10 optimisers x 10 runs x 240 evaluations = 24000 budgeted evaluations (23759 of them needed training; the rest were repeats served from the shared memo)

## Search-phase outcome

- Reference front: 16 solutions pooled from 958 archive members
- Hypervolume ranking (best first): MO-P3PGA-II (2.50), GDE3 (3.70), MOGWO (3.70), MO-P3PGA (4.40), MOEA/D (5.00), MOPSO (5.50), Random Search (6.70), NSGA-II (7.50), NSGA-III (7.90), SPEA2 (8.10)
- MO-P3PGA-II mean HV = 1.1339 ± 0.0206, average rank 2.50
- Friedman on HV: chi2 = 38.073, p = 1.692e-05 (N = 10 runs, k = 10)
- Wilcoxon (Holm-corrected) significant wins for MO-P3PGA-II: 5/9 (MOPSO, NSGA-II, NSGA-III, Random Search, SPEA2)
- Contributions to the reference front: MO-P3PGA=2, MO-P3PGA-II=1, NSGA-II=0, NSGA-III=0, MOEA/D=3, SPEA2=0, MOPSO=1, GDE3=0, MOGWO=4, Random Search=1

## Equal-tuning protocol

- GDE3-tuned, MOGWO-tuned replace their default runs (same seeds)
- Hypervolume ranking (best first): GDE3-tuned (2.20), MOGWO-tuned (2.30), MO-P3PGA-II (2.50), MOEA/D (4.90), MOPSO (5.20), Random Search (6.40), NSGA-II (7.10), NSGA-III (7.20), SPEA2 (7.20)
- Friedman on HV: p = 4.135e-08, Nemenyi CD = 3.80
- MO-P3PGA-II vs GDE3-tuned: mean HV 1.1339 vs 1.1399, Holm p = 1, A12 = 0.41
- MO-P3PGA-II vs MOGWO-tuned: mean HV 1.1339 vs 1.1419, Holm p = 1, A12 = 0.40
- GDE3: default HV 1.1187 -> tuned 1.1399 (paired Wilcoxon p = 0.0195)
- MOGWO: default HV 1.1153 -> tuned 1.1419 (paired Wilcoxon p = 0.0645)

## Final models (held-out test split)

- Headline model `MO-P3PGA-II (accuracy)`: accuracy **96.98%** (95% CI 95.55–98.25), macro-F1 **96.21%** (95% CI 94.48–97.79)
- Cost: 0.195 M parameters, 102 MMac, 1.98 ms/image at batch 1 on Tesla T4
- MO-P3PGA-II accuracy point: macro-F1 96.24% ± 0.04 over 3 seeds, 0.195 M parameters, 102 MMac
- MO-P3PGA-II knee point: macro-F1 88.88% ± 0.64 over 3 seeds, 0.002 M parameters, 4 MMac
- MO-P3PGA-II efficiency point: macro-F1 63.31% ± 2.05 over 3 seeds, 0.001 M parameters, 4 MMac
- Best model by test macro-F1 overall: `resnet18 (ImageNet)` (97.06%)
- McNemar of `MO-P3PGA-II (accuracy)` against the other 21 models (Holm-corrected): significantly better than 8, significantly worse than 0, no significant difference from 13 (NSGA-II (knee), MOGWO (knee), Random Search (knee), MOPSO (knee), resnet18 (ImageNet), GDE3 (knee), Hand-designed CNN, MO-P3PGA (accuracy), shufflenet_v2_x1_0 (ImageNet), efficientnet_b0 (ImageNet), mobilenet_v2 (ImageNet), mobilenet_v3_small (ImageNet), MOGWO-tuned (knee))

## Protocol validation

- Proxy fidelity: Spearman rho = 0.726 between proxy macro-F1 and full-training test macro-F1 (n = 25)

## Ablation (10 paired runs; Friedman on HV p = 0.00281)

- MO-P3PGA literal port: HV 1.1099 (+0.64%), 4.0 generations, Holm p = 1 vs full
- MO-P3PGA (full): HV 1.1029 (+0.00%), 4.0 generations
- MO-P3PGA w/o attract: HV 1.1000 (-0.27%), 5.1 generations, Holm p = 1 vs full
- MO-P3PGA NP=1: HV 1.0894 (-1.22%), 4.0 generations, Holm p = 1 vs full
- MO-P3PGA w/o mito: HV 1.0466 (-5.11%), 6.0 generations, Holm p = 0.00781 vs full

## Parameter sensitivity (best setting by mean HV)

- NP: best = 1.0 (HV 1.0477), worst = 8.0 (HV 1.0195)
- p_attract: best = 0.4 (HV 1.0563), worst = 0.1 (HV 1.0214)
- mito_scale: best = 0.4 (HV 1.0707), worst = 0.1 (HV 0.9923)

## Architecture discovered (headline model)

```json
{
  "n_blocks": 4,
  "base_filters": 64,
  "growth": 1.75,
  "block_type": "sep",
  "kernel": 3,
  "use_se": true,
  "se_ratio": 0.25,
  "activation": "gelu",
  "stem_stride": 1,
  "head_pool": "avg",
  "dense_units": 0,
  "conv_dropout": 0.0622,
  "dense_dropout": 0.4057,
  "optimizer": "rmsprop",
  "lr": 0.001341,
  "weight_decay": 3.904e-06,
  "batch_size": 64,
  "label_smooth": 0.0092,
  "aug_strength": 0.5956,
  "mix_alpha": 0.3635,
  "mix_mode": "cutmix",
  "p_erase": 0.4428,
  "stage_depth": 1,
  "expand_ratio": 2
}
```

## Artefacts

- Tables: `/kaggle/working/outputs/tables/` (CSV + LaTeX)
- Figures: `/kaggle/working/outputs/figures/` (PNG 300 dpi + PDF)
- Checkpoints: `/kaggle/working/p3pga_state/` (delete to force a clean re-run)
- Session wall-clock used: 0h02m18s