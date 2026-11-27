# MO-P3PGA-II: multi-objective CNN design for crop disease recognition

Code, configuration and results for the study *"MO-P3PGA-II: An evaluation-efficient parallel three-parent genetic algorithm for multi-objective CNN design in crop disease diagnosis"* (Gagandeep Sethi and Amar Singh, Department of Computer Applications, Lovely Professional University). The manuscript is under review.

The study extends the parallel three-parent genetic algorithm (P3PGA) of Singh et al. (2020) to multi-objective neural architecture search. The search designs a CNN **and** its training recipe so as to minimise two objectives together:

- the validation error (1 − macro-F1), and
- the computational cost (log₁₀ MMac).

## What is in the study

| Component | Description |
|---|---|
| **MO-P3PGA** | Direct multi-objective port of P3PGA: islands, mitochondrial three-parent change and leader pull, with Pareto-rank and crowding survival. It spends about three times its population size in evaluations per generation. |
| **MO-P3PGA-II** | The proposed redesign. It uses the same three mechanisms on one offspring population per island, evaluated once, so a generation costs one population. |
| **Competitors** | NSGA-II, NSGA-III, MOEA/D, SPEA2, MOPSO, GDE3, MOGWO and random search, all with the same search space, decoder, population of 24 and budget of 240 evaluations per run, and 10 runs each. |
| **Protocols** | *Standard*: competitors at their recommended settings. *Equal tuning*: GDE3 and MOGWO tuned with the same 70-configuration screen as MO-P3PGA-II. |
| **Statistics** | Friedman with the Nemenyi critical difference; Wilcoxon and Mann–Whitney with Holm correction; Vargha–Delaney Â₁₂; McNemar on test predictions; bootstrap confidence intervals. |
| **Supporting studies** | Paired ablation, parameter sensitivity, proxy-fidelity check, corruption robustness, Grad-CAM, and a data-leakage audit with a photograph-level split for rice. |

## Main results

**Maize** (4,188 images, 4 classes; 10 runs per optimiser):

| | Avg. HV rank (1 = best) | Notes |
|---|---|---|
| **MO-P3PGA-II** | **2.50** | Significantly better than 5 of 9 competitors (Holm); tied with GDE3/MOGWO |
| GDE3 / MOGWO | 3.70 / 3.70 | Tied with MO-P3PGA-II when all three are tuned |
| MO-P3PGA (direct port) | 4.40 | Completes only 4 generations against 10 |

Its most accurate network reaches **96.98% test accuracy with 0.195 M parameters**. McNemar finds no significant difference from ImageNet-pretrained ResNet18, MobileNetV2 or EfficientNet-B0, which are 11 to 57 times larger.

**Rice** (5,932 images, but only **509 distinct photographs**; split by photograph; settings transferred from maize unchanged):

| | Avg. HV rank | Notes |
|---|---|---|
| **MO-P3PGA-II** | **1.70** | Significantly better than MO-P3PGA, MOEA/D, NSGA-II and random search (Holm) |
| MOGWO / GDE3 | 2.90 / 3.80 | |

The complete tables, statistics and figures are in [`results/`](results/): `results/maize/RESULTS.md` and `results/rice/RESULTS.md` summarise each dataset.

## Repository layout

```
p3pga_mo_nas_maize.ipynb    the experiment, as a Kaggle notebook
p3pga_mo_nas_maize.py       the same code as a script; the source of truth (cells marked "# %%")
tools/py2nb.py              rebuilds the notebook from the script
tools/rice_leakage_check.py sensitivity of the rice test results to residual near-duplicates
tuning_surrogate/           the configuration screen (surrogate benchmark and screening scripts)
results/maize, results/rice tables (CSV/LaTeX), statistics (JSON), figures (PNG) and digests
```

## Running the experiment

### On Kaggle (as used in the study)

1. Import `p3pga_mo_nas_maize.ipynb` (File → Import Notebook). Set GPU T4 and Internet on.
2. Add the dataset input:
   - maize: [`smaranjitghose/corn-or-maize-leaf-disease-dataset`](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset)
   - rice: [`nirmalsankalana/rice-leaf-disease-image`](https://www.kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image)
3. In the first code cell, set `DATASET` (`"maize"` or `"rice"`) and `RUN_PLAN`:
   - `"all"` runs the search, then the analysis;
   - `"search"` and `"analysis"` run one phase each.
4. Save Version → Save & Run All.

A Kaggle session lasts at most 12 hours. The notebook checkpoints every run, stops cleanly before the limit, and resumes when you attach the previous version's output as an input and run again. A full maize study (10 optimiser configurations plus 2 tuned ones, 10 runs, 240 evaluations each) needs roughly 25 GPU-hours on a Tesla T4. The rice study (7 optimisers) needs about 16.

### Locally

```bash
pip install -r requirements.txt
P3PGA_DATASET=maize P3PGA_PLAN=all P3PGA_DATA=/path/to/images python p3pga_mo_nas_maize.py
```

`P3PGA_DATA` points to a folder with one sub-folder per class. `P3PGA_PRESET=smoke` runs a few-minute pipeline check, and `P3PGA_HOURS` sets the session clock.

## Reproducibility notes

- **Seeds:** run *r* of an optimiser uses seed `1000·r + crc32(label) mod 977`. MO-P3PGA-II reuses the MO-P3PGA seeds, and the tuned competitors reuse their default seeds, so these comparisons are paired. The data split uses seed 42.
- **Budget:** a run is charged one evaluation per distinct phenotype it queries. A shared memo avoids retraining identical networks across runs, without changing what each run is charged.
- **Tuning:** the configuration screen used seeds 20–49, which the reported runs (seeds 0–9) never use. See [`tuning_surrogate/README.md`](tuning_surrogate/README.md).
- **Rice data:** exact duplicates are removed first. Rotated, flipped or re-encoded copies of one photograph are then grouped, so every copy of a photograph lands in a single split, and the test split keeps one image per photograph. Most published results on this dataset use image-level splits, which leak copies of training photographs into the test set.

## Citation

If you use this code, please cite the paper (details will be added on publication):

```
Sethi G, Singh A. MO-P3PGA-II: An evaluation-efficient parallel three-parent genetic algorithm
for multi-objective CNN design in crop disease diagnosis. Manuscript under review, 2026.
```

The original P3PGA: Singh A, Kumar S, Singh A, Walia SS. *Parallel 3-Parent Genetic Algorithm with Application to Routing in Wireless Mesh Networks.* In: Implementations and Applications of Machine Learning, Studies in Computational Intelligence vol. 782, Springer, 2020. https://doi.org/10.1007/978-3-030-37830-1_1

## Notes

All reported results were produced by running the code in this repository.

The datasets are not redistributed here; download them from the Kaggle pages above under their own licences.

Code: MIT License (see [`LICENSE`](LICENSE)).
