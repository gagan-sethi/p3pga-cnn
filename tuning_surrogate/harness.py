"""Surrogate benchmark built from the real proxy evaluations of the maize standard-protocol study.

Exact memo lookup for every phenotype that was really trained; a random-forest prediction for
phenotypes never trained. Used only to SCREEN optimiser settings on CPU before GPU runs.
"""
import re, sys, math, pickle, zlib, time
import numpy as np
from pathlib import Path

import os
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# the maize standard-protocol checkpoint the screen was built from (bundled in ./state);
# point P3PGA_SURROGATE_STATE at another p3pga_state folder to rebuild the surrogate from it
STATE = Path(os.environ.get("P3PGA_SURROGATE_STATE", HERE / "state"))
SRC = (REPO / "p3pga_mo_nas_maize.py").read_text()
CELLS = re.split(r'^# %%.*$', SRC, flags=re.M)

NS = {"__name__": "nb"}
exec("""
import os, sys, re, zlib, json, time, math, random, hashlib, warnings, pickle, shutil, gc
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any, Callable
import numpy as np
class MOEvaluator: pass
class BudgetExhausted(Exception): pass
class TimeBudgetExceeded(Exception): pass
class Clock: pass
DATASET = 'maize'; PRESET = 'paper'; PRIMARY_DATASET = 'maize'; RUN_PLAN = 'analysis'
""", NS)
import io, contextlib
# the cells that define the configuration, decoder, Pareto machinery, indicators and optimisers,
# located by content so that edits elsewhere in the notebook do not shift them
NEEDED = ("class Config", "def decode", "def environmental_selection", "def normalize_front",
          "def clip01", "class MOP3PGA(", "OPTIMIZERS: Dict")
for cell in CELLS:
    if any(k in cell for k in NEEDED):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell, NS)
CFG = NS["CFG"]
genotype_key, decode = NS["genotype_key"], NS["decode"]
BudgetExhausted = NS["BudgetExhausted"] if "BudgetExhausted" in NS else None




# MOOptimizer.run catches NS['BudgetExhausted'] and NS['TimeBudgetExceeded']
BudgetExhausted_ = NS["BudgetExhausted"]

PENALTY = np.array([1.0, math.log10(CFG.max_flops_mmac * 10)])

# ---------------------------------------------------------------- data
MEMO = pickle.load(open(STATE / "global_memo.pkl", "rb"))
SEARCH = pickle.load(open(STATE / "search_results.pkl", "rb"))

def obj_of(res):
    if not res.get("ok", False):
        return PENALTY.copy()
    return np.array([1.0 - res["val_f1"], math.log10(max(res["mmac"], 1e-3))])

# phenotype features for every memo entry (phenotypes come from histories)
PH = {}
for f in ("search_results", "ablation_paired", "sensitivity_results"):
    for r in pickle.load(open(STATE / f"{f}.pkl", "rb")):
        for h in r["history"]:
            PH.setdefault(genotype_key(h["x"]), h["ph"])
CATS = {}
def feat(ph):
    v = []
    for k in sorted(ph):
        val = ph[k]
        if isinstance(val, (bool, np.bool_)):
            v.append(float(val))
        elif isinstance(val, (int, float, np.integer, np.floating)):
            x = float(val)
            v.append(math.log10(x) if k in ("lr", "weight_decay") and x > 0 else x)
        else:
            opts = CATS.setdefault(k, sorted({str(p[k]) for p in PH.values()}))
            v += [float(str(val) == o) for o in opts]
    return v

keys = [k for k in MEMO if k in PH]
Xf = np.array([feat(PH[k]) for k in keys])
R = [MEMO[k] for k in keys]
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
ok = np.array([r.get("ok", False) for r in R])
y_err = np.array([1 - r["val_f1"] if r.get("ok") else np.nan for r in R])
y_fl = np.array([math.log10(max(r["mmac"], 1e-3)) for r in R])
y_pa = np.array([math.log10(max(r["params_m"], 1e-5)) for r in R])
M_ERR = ExtraTreesRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=0)
M_FL = ExtraTreesRegressor(200, min_samples_leaf=1, n_jobs=-1, random_state=0)
M_PA = ExtraTreesRegressor(200, min_samples_leaf=1, n_jobs=-1, random_state=0)
if "--cv" in sys.argv:
    for name, m, y, mask in (("error", M_ERR, y_err, ok), ("log mmac", M_FL, y_fl, slice(None)),
                             ("log params", M_PA, y_pa, slice(None))):
        print(name, "CV R2:", cross_val_score(m, Xf[mask], y[mask], cv=3, scoring="r2").round(3))
M_ERR.fit(Xf[ok], y_err[ok]); M_FL.fit(Xf, y_fl); M_PA.fit(Xf, y_pa)


class SurEval:
    """Same contract as MOEvaluator: unique phenotypes cost budget, repeats are free."""
    def __init__(self, budget):
        self.budget, self.n_evals, self.n_queries, self.n_pred = budget, 0, 0, 0
        self.cache, self.history = {}, []

    def __call__(self, x):
        self.n_queries += 1
        key = genotype_key(x)
        if key in self.cache:
            if self.n_queries > 200 * self.budget + 2000:
                raise BudgetExhausted_()
            return self.cache[key].copy()
        if self.n_evals >= self.budget:
            raise BudgetExhausted_()
        self.n_evals += 1
        if key in MEMO:
            f = obj_of(MEMO[key])
        else:
            self.n_pred += 1
            v = np.array([feat(decode(x))])
            fl, pa = M_FL.predict(v)[0], M_PA.predict(v)[0]
            if 10 ** fl > CFG.max_flops_mmac or 10 ** pa > CFG.max_params_m:
                f = PENALTY.copy()
            else:
                f = np.array([float(np.clip(M_ERR.predict(v)[0], 0, 1)), fl])
        self.cache[key] = f
        self.history.append(f)
        return f.copy()

    def batch(self, X):
        return np.vstack([self(x) for x in np.atleast_2d(X)])


OPT = NS["OPTIMIZERS"]
ALL_F = np.vstack([r["archive_F"] for r in SEARCH if r["archive_F"].size])
Z_MIN, Z_MAX = ALL_F.min(0), ALL_F.max(0)
REF_FRONT_N = NS["normalize_front"](ALL_F[NS["non_dominated_mask"](ALL_F)], Z_MIN, Z_MAX)
REF_POINT = np.full(2, 1.1)

def ind(F):
    Fn = NS["normalize_front"](F, Z_MIN, Z_MAX)
    Fn = Fn[NS["non_dominated_mask"](Fn)]
    return NS["hypervolume"](Fn, REF_POINT), NS["igd_plus"](Fn, REF_FRONT_N)

def run(algo, run_idx, budget=240, overrides=None, label=None, seed_label=None):
    seed = 1000 * run_idx + zlib.crc32((seed_label or label or algo).encode()) % 977
    ev = SurEval(budget)
    opt = OPT[algo](CFG, ev, seed=seed, overrides=overrides)
    arch = opt.run()
    hv, igdp = ind(arch.F)
    return {"algo": algo, "label": label or algo, "run": run_idx, "HV": hv, "IGD+": igdp,
            "gen": opt.n_gen, "pred_frac": ev.n_pred / max(1, ev.n_evals), "F": arch.F.copy()}
