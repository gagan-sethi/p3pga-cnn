# Configuration screen (surrogate benchmark)

The settings of MO-P3PGA-II, and of the tuned GDE3 and MOGWO in the equal-tuning protocol, were chosen with a screen that runs on a CPU in about two and a half hours and needs no GPU.

## How the surrogate works

`harness.py` answers each fitness query in one of two ways:

- **Exact lookup** of the recorded proxy objectives, for every phenotype trained during the maize standard-protocol search, ablation and sensitivity studies (30,054 phenotypes, stored in [`state/`](state/)).
- **Extra-trees prediction** for any other phenotype: 300 trees for the error, and 200 each for log MMac and log parameters. The feasibility box is applied to the predicted costs.

It keeps the evaluator's budget rules: a repeated phenotype within a run is free, and every new phenotype costs one evaluation. Replaying a recorded run on the surrogate reproduces its hypervolume exactly (`validate.py`).

## Scripts

| Script | What it does |
|---|---|
| `validate.py` | 3-fold CV of the surrogate models; replays recorded runs |
| `screen.py` | 70 MO-P3PGA configurations plus the default competitors on tuning seeds 20–29 |
| `screen2.py` | Stage A: equal-effort grids for GDE3 (70 points) and MOGWO (56 points) on seeds 20–29. Stage B: finalists re-run on fresh seeds 30–49 |
| `screen3.py` | 70 MO-P3PGA-II configurations on seeds 20–29; the top 5 re-run on seeds 30–49 |
| `smoke2.py` | Quick check of MO-P3PGA-II on the surrogate |

The `screen*.pkl` files hold the outputs of the runs reported in the paper.

```bash
pip install numpy scipy pandas scikit-learn
cd tuning_surrogate && python screen3.py
```

Set `P3PGA_SURROGATE_STATE` to another `p3pga_state` folder to rebuild the surrogate from different data. The `.pkl` files are Python pickles, so load only files you trust.
