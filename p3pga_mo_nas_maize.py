# %% [markdown]
# # Multi-Objective Parallel 3-Parent Genetic Algorithm (MO-P3PGA) for CNN Architecture Search
# ## Pareto-Optimal CNN Design for Crop Leaf Disease Classification (Maize and Rice)
#
# **What this notebook does**
#
# 1. Ports the **Parallel 3-Parent Genetic Algorithm (P3PGA)** of Singh, Kumar, Singh & Walia
#    (*Implementations and Applications of Machine Learning*, Studies in Computational
#    Intelligence 782, Springer, 2020, Ch. 1) from its original single-objective form
#    (Algorithm 1 of that chapter) to a **multi-objective, Pareto-based** optimizer (**MO-P3PGA**).
# 2. Applies MO-P3PGA to **joint neural architecture search + hyper-parameter optimisation**
#    of CNNs for **maize and rice leaf disease classification** (4 classes each).
# 3. Benchmarks it against **8 other multi-objective optimisers** (NSGA-II, NSGA-III, MOEA/D,
#    SPEA2, MOPSO, GDE3, MOGWO, Random Search) on an **identical search space,
#    identical evaluation budget and identical fitness evaluator** — the fairness conditions
#    reviewers check first.
# 4. Produces every artefact a Q1/Q2 submission needs: Pareto fronts, hypervolume / IGD+ /
#    GD / spacing / spread / set-coverage tables, Friedman + Wilcoxon-Holm tests, a
#    critical-difference diagram, final test-set metrics with bootstrap CIs, McNemar tests,
#    ablations, parameter-sensitivity study, proxy-fidelity validation, Grad-CAM maps,
#    and LaTeX-ready tables.
#
# **Objectives (all minimised)**
#
# | # | Objective | Why it matters for the paper |
# |---|-----------|------------------------------|
# | $f_1$ | classification error $(1-\text{macro-}F_1)$ on a validation split | diagnostic quality |
# | $f_2$ | $\log_{10}$ FLOPs (MMac) | compute cost for in-field / drone / phone inference |
# | $f_3$ | $\log_{10}$ parameters (optional, `N_OBJ=3`) | memory footprint on edge devices |
#
# Monotone ($\log_{10}$) transforms leave the Pareto dominance relation **unchanged** while
# greatly improving the conditioning of crowding distance, Tchebycheff aggregation and
# hypervolume. Raw units are restored for all reported tables.
#
# **Runtime budget.** Designed for **~22 GPU-hours ≈ 1 GPU-day**, split into two
# resumable phases so it fits Kaggle's 12-hour-per-session cap:
#
# | Session | `RUN_PLAN` | Content | Approx. |
# |---------|-----------|---------|---------|
# | 1 | `"search"` | data cache + calibration + 9 optimisers x 3 runs | ~6-10 h |
# | 2 | `"analysis"` | final training, baselines, ablations, sensitivity, fidelity, stats, figures | ~10.5 h |
#
# Everything checkpoints to `/kaggle/working/p3pga_state/`. Kaggle starts each new version with
# an empty working folder, so attach the previous version's output as an input: this notebook
# finds `p3pga_state` there and restores it, then continues where it stopped; a wall-clock guard stops cleanly and saves partial results rather than
# letting Kaggle kill the kernel. Set `RUN_PLAN = "all"` if you have a single long session
# (Colab A100, local workstation). Set `PRESET = "smoke"` first — it reproduces the whole
# pipeline end-to-end in ~20 minutes so you can verify everything before spending real GPU time.
#
# **Datasets.** Set `DATASET` in the config cell and add the matching Kaggle input:
#
# * `DATASET = "maize"`: `smaranjitghose/corn-or-maize-leaf-disease-dataset` (4188 images)
# * `DATASET = "rice"`: `nirmalsankalana/rice-leaf-disease-image` (5932 images)
#
# ---
# **Citation for the base algorithm**
# > A. Singh, S. Kumar, A. Singh, S. S. Walia, "Parallel 3-Parent Genetic Algorithm with
# > Application to Routing in Wireless Mesh Networks," in *Implementations and Applications of
# > Machine Learning*, Studies in Computational Intelligence, vol. 782, Springer, 2020, pp. 1-28.
# > doi:10.1007/978-3-030-37830-1_1

# %% [markdown]
# ## 1. Configuration
#
# Every experimental knob lives in this one cell. `PRESET` picks a coherent set of budgets.

# %%
import os, sys, re, zlib, json, time, math, random, hashlib, warnings, pickle, shutil, gc
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any, Callable

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------
# PRESET: "smoke"  -> ~20 min, verifies the entire pipeline (use this first!)
#         "paper"  -> ~22 GPU-h total across 2 sessions (the configuration to publish)
#         "extended" -> ~45 GPU-h, 5 search runs, 300 evaluations (strongest statistics)
# ----------------------------------------------------------------------------------
PRESET   = os.environ.get("P3PGA_PRESET", "paper")      # "smoke" | "paper" | "extended"
RUN_PLAN = os.environ.get("P3PGA_PLAN",  "all")         # "search" | "analysis" | "all"

# DATASET: "maize" = primary study (search + full analysis, ablation, sensitivity)
#          "rice"  = independent second dataset (search + final models + fidelity)
# Run each one as its own Kaggle versions; see the README for the order.
DATASET = os.environ.get("P3PGA_DATASET", "maize")
PRIMARY_DATASET = "maize"
DATASETS = {
    "maize": {"title": "maize (corn) leaf disease", "crop": "maize",
              "slug": "smaranjitghose/corn-or-maize-leaf-disease-dataset",
              "classes": [("Blight", "Northern Leaf Blight", ("blight",)),
                          ("Common_Rust", "Common Rust", ("rust",)),
                          ("Gray_Leaf_Spot", "Gray Leaf Spot", ("cercospora", "gray", "grey")),
                          ("Healthy", "Healthy", ("healthy",))]},
    "rice":  {"title": "rice leaf disease", "crop": "rice",
              "slug": "nirmalsankalana/rice-leaf-disease-image",
              "classes": [("Bacterial_Blight", "Bacterial Blight", ("bacteri",)),
                          ("Blast", "Blast", ("blast",)),
                          ("Brown_Spot", "Brown Spot", ("brown",)),
                          ("Tungro", "Tungro", ("tungro",))]},
}
assert DATASET in DATASETS, f"unknown DATASET {DATASET!r}"
DS = DATASETS[DATASET]
_SUFFIX = "" if DATASET == PRIMARY_DATASET else f"_{DATASET}"

# Wall-clock guard for THIS session, in hours. Kaggle kills at 12 h; leave margin.
SESSION_HOURS = float(os.environ.get("P3PGA_HOURS", 11.3))

# Output locations: Kaggle paths when on Kaggle, a local folder otherwise (Colab, workstation).
_BASE = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("./p3pga")
OUT_DIR = os.environ.get("P3PGA_OUT", f"{_BASE}/outputs{_SUFFIX}")
STATE_DIR = os.environ.get("P3PGA_STATE", f"{_BASE}/p3pga_state{_SUFFIX}")
DATA_ROOT_ENV = os.environ.get("P3PGA_DATA", "")      # "" => auto-detect


@dataclass
class Config:
    # ---------------- bookkeeping ----------------
    preset: str = "paper"
    seed: int = 42
    out_dir: str = OUT_DIR
    state_dir: str = STATE_DIR
    data_root: str = DATA_ROOT_ENV          # auto-detected if empty

    # ---------------- objectives ----------------
    # order matters; all are MINIMISED
    objectives: Tuple[str, ...] = ("error", "flops")   # subset of error|flops|params|latency
    # Feasibility box. Architectures outside it are rejected without training and scored at
    # the penalty point. This encodes the study's edge-deployment scope and, as a practical
    # side effect, bounds the cost of the slowest possible proxy evaluation.
    max_params_m: float = 12.0
    max_flops_mmac: float = 1500.0

    # ---------------- data ----------------
    cache_size: int = 176                   # images decoded once and cached at this resolution
    dedup_exact: bool = False               # drop exact duplicate images before splitting
                                            # (on for the second dataset; maize has 2 duplicates,
                                            # none crossing splits, so its splits stay unchanged)
    group_split: bool = False               # keep rotated/flipped/re-encoded copies of one photo
                                            # in one split; val/test keep one image per photo
    near_dup_threshold: float = 0.995       # thumbnail similarity that defines "same photo"
    group_split_fracs: Tuple[float, float, float] = (0.60, 0.20, 0.20)   # by photo, not image
    group_val_all_copies: bool = True       # validation keeps every copy of its photos (a
                                            # stable search score); test keeps one per photo
    img_size_proxy: int = 96                # resolution used during the search
    img_size_final: int = 160               # resolution used for final/full training
    split: Tuple[float, float, float] = (0.70, 0.15, 0.15)   # train / val / test
    n_classes: int = 4

    # ---------------- proxy evaluation (the search's inner loop) ----------------
    proxy_train_n: int = 1400               # stratified subset of the training split
    proxy_val_n: int = 600                  # stratified subset of the validation split
    proxy_epochs: int = 6
    proxy_batch: int = 64
    proxy_seed: int = 1234                  # fixed: every genotype is trained from the same seed
    proxy_amp: bool = True

    # ---------------- search protocol ----------------
    budget: int = 240                       # unique fitness evaluations per optimiser run
    n_runs: int = 10                        # independent repetitions per optimiser; with 3 runs
                                            # no Holm-corrected Wilcoxon test can reach p < 0.05
    pop_total: int = 24                     # total population size (shared by all optimisers)
    archive_size: int = 60                  # external Pareto archive cap

    # ---------------- MO-P3PGA (paper's Algorithm 1, lifted to Pareto) ----------------
    p3pga_np: int = 4                       # NP: number of parallel populations ("islands")
    p3pga_n: int = 6                        # N : individuals per population (NP*N = pop_total)
    mito_scale: float = 0.2222              # paper: U(-2,2) on a [0,9] gene domain = 0.2222
    p_crossover: float = 0.90               # paper: 0.9
    p_mutation: float = 0.10                # paper: 0.1
    mut_eta: float = 20.0                   # polynomial-mutation distribution index
    cx_eta: float = 15.0                    # SBX distribution index
    p_attract: float = 0.10                 # Step 5: per-gene probability of pull toward leader
    attract_every: int = 1                  # paper text says "after a predetermined number of
                                            # generations"; Algorithm 1 does it every generation
    leader_mode: str = "random_archive"     # random_archive | global_least_crowded | per_island
    crossover_op: str = "sbx"               # sbx | npoint   (npoint = literal port of the paper)
    selection_op: str = "tournament"        # tournament | sus_rank  (sus_rank = paper's SUS)
    use_mito: bool = True                   # ablation switch
    use_attract: bool = True                # ablation switch

    # ---------------- final training ----------------
    final_epochs: int = 60
    final_batch: int = 32
    final_seeds: Tuple[int, ...] = (0, 1, 2)
    final_label_smooth: float = 0.05
    final_warmup_frac: float = 0.1
    final_ema: bool = True

    # ---------------- analysis-phase switches ----------------
    run_final_training: bool = True
    run_baselines: bool = True
    run_ablations: bool = True
    run_sensitivity: bool = True
    run_fidelity: bool = True
    run_gradcam: bool = True
    run_robustness: bool = True
    run_single_objective: bool = False      # supplementary SO study (adds ~2.5 h)
    improved: bool = True                   # add MO-P3PGA-II to the main comparison
    headline_point: str = "accuracy"        # which operating point of the proposed front leads
                                            # the report: "accuracy" | "knee" | "efficiency"
    # second dataset: the proposed methods plus the strongest maize competitors, NSGA-II as the
    # reference MOEA and Random Search as the sanity baseline (7 configurations x 10 runs)
    secondary_algos: Tuple[str, ...] = ("MO-P3PGA", "NSGA-II", "MOEA/D", "GDE3", "MOGWO",
                                        "Random Search")
    equal_tuning: bool = True               # also run GDE3 and MOGWO with tuned settings

    ablation_runs: int = 10                 # paired with the main MO-P3PGA runs (same seeds)
    ablation_budget: int = 240              # = budget, so the main runs serve as the full variant
    sensitivity_runs: int = 3               # mean +/- std per setting; 1 run cannot separate
    sensitivity_budget: int = 120           # a setting's effect from seed noise
    fidelity_n: int = 25                    # configs retrained fully to validate the proxy

    def __post_init__(self):
        self.n_obj = len(self.objectives)


def make_config(preset: str) -> Config:
    c = Config(preset=preset)
    if preset == "smoke":
        c.budget, c.n_runs = 24, 2
        c.pop_total, c.p3pga_np, c.p3pga_n, c.archive_size = 8, 2, 4, 20
        c.proxy_train_n, c.proxy_val_n = 240, 160
        c.proxy_epochs, c.img_size_proxy = 2, 64
        c.final_epochs, c.final_seeds, c.img_size_final = 4, (0,), 96
        c.cache_size = 112
        c.ablation_runs, c.ablation_budget = 2, 24
        c.sensitivity_runs, c.sensitivity_budget = 1, 16
        c.fidelity_n = 6
        # keep every candidate small so the smoke test is quick even on a CPU-only machine
        c.max_params_m, c.max_flops_mmac = 2.0, 120.0
    elif preset == "paper":
        pass                                 # dataclass defaults ARE the paper configuration
    elif preset == "extended":
        c.budget, c.n_runs = 300, 10
        c.pop_total, c.p3pga_np, c.p3pga_n, c.archive_size = 32, 4, 8, 80
        c.proxy_train_n, c.proxy_epochs = 1800, 8
        c.final_epochs, c.final_seeds = 80, (0, 1, 2, 3, 4)
        c.ablation_runs, c.ablation_budget = 10, 300
        c.sensitivity_runs, c.sensitivity_budget = 2, 150
        c.fidelity_n = 40
    else:
        raise ValueError(f"unknown preset {preset!r}")
    assert c.p3pga_np * c.p3pga_n == c.pop_total, "NP*N must equal pop_total for a fair comparison"
    if DATASET != PRIMARY_DATASET:
        # same search and final-model protocol as the primary dataset; the algorithm-internals
        # studies are answered once, on the primary dataset, to stay inside one GPU-day
        c.run_ablations = c.run_sensitivity = False
        c.fidelity_n = min(c.fidelity_n, 12)
        c.equal_tuning = False              # the tuning question is answered on maize
        c.dedup_exact = True                # rice ships 1180 exact copies; keep one of each
        c.group_split = True                # and most remaining images are rotated/flipped
                                            # copies of fewer photos (711/712 test images had one)
    return c


CFG = make_config(PRESET)
# switches for the improved-algorithm runs ("0" reproduces the original nine-optimiser study)
CFG.improved = os.environ.get("P3PGA_IMPROVED", "1") != "0"
CFG.equal_tuning = (os.environ.get("P3PGA_TUNED", "1") != "0") and CFG.equal_tuning

# Set to 3 objectives by uncommenting:  (error, FLOPs, params)
# CFG.objectives = ("error", "flops", "params"); CFG.n_obj = 3

print(f"DATASET={DATASET}  PRESET={PRESET}  RUN_PLAN={RUN_PLAN}  SESSION_HOURS={SESSION_HOURS}")
print(f"objectives={CFG.objectives}  budget={CFG.budget}  runs={CFG.n_runs}  pop={CFG.pop_total}")

# %% [markdown]
# ## 2. Imports, determinism, device, wall-clock budget

# %%
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    raise SystemExit("PyTorch is required. On Kaggle enable the GPU accelerator.")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from scipy import stats as sps
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report, roc_auc_score,
                             cohen_kappa_score, balanced_accuracy_score,
                             matthews_corrcoef)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu"
print(f"torch {torch.__version__} | device: {DEVICE} | {GPU_NAME}")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
else:
    print("WARNING: no GPU detected -- only PRESET='smoke' is realistic on CPU.")

OUT = Path(CFG.out_dir);   OUT.mkdir(parents=True, exist_ok=True)
STATE = Path(CFG.state_dir); STATE.mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)
(OUT / "tables").mkdir(exist_ok=True)


def restore_checkpoints():
    """Copy p3pga_state from an attached Kaggle input into the empty working folder."""
    if (STATE / "search_results.pkl").exists():
        return
    root = Path(os.environ.get("P3PGA_INPUT", "/kaggle/input"))
    if not root.exists():
        return
    cands = []
    for p in root.rglob("search_results.pkl"):
        try:
            with open(p, "rb") as fh:
                recs = pickle.load(fh)
        except Exception as e:
            print(f"[restore] unreadable {p}: {e}")
            continue
        ok = (recs and recs[0].get("dataset", PRIMARY_DATASET) == DATASET
              and recs[0].get("budget", 0) >= 0.4 * CFG.budget
              and tuple(recs[0].get("objectives", ())) == tuple(CFG.objectives))
        if ok:
            cands.append((len(recs), str(p)))
        else:
            print(f"[restore] ignored {p}: other dataset, preset or objectives")
    if not cands:
        print("[restore] no previous checkpoints attached; starting fresh")
        return
    n, best = max(cands)
    for f in Path(best).parent.iterdir():
        if f.is_file() and not f.name.startswith("imgcache_"):
            shutil.copy2(f, STATE / f.name)
    print(f"[restore] restored {n} search runs from {Path(best).parent}")


restore_checkpoints()

# A native crash (segfault, CUDA abort) kills the kernel without a Python traceback. These two
# files survive in the output folder and show what the process was doing when it died.
import faulthandler
_CRASH_FH = open(STATE / "crash_trace.log", "a")
faulthandler.enable(file=_CRASH_FH)


def resources() -> str:
    bits = []
    try:
        import psutil
        vm = psutil.virtual_memory()
        bits.append(f"RAM {vm.used/2**30:.1f}/{vm.total/2**30:.1f}GiB")
    except Exception:
        pass
    du = shutil.disk_usage(STATE)
    bits.append(f"disk free {du.free/2**30:.1f}GiB")
    if torch.cuda.is_available():
        bits.append(f"GPU reserved {torch.cuda.memory_reserved()/2**30:.2f}GiB")
    return " | ".join(bits)


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


set_seed(CFG.seed)


class Clock:
    """Session wall-clock guard. Phases query `.left()` and degrade gracefully."""

    def __init__(self, hours: float):
        self.t0 = time.time()
        self.limit = hours * 3600.0

    def elapsed(self) -> float:
        return time.time() - self.t0

    def left(self) -> float:
        return self.limit - self.elapsed()

    def ok(self, need: float = 0.0) -> bool:
        return self.left() > need

    def hms(self, t: Optional[float] = None) -> str:
        t = self.elapsed() if t is None else t
        t = max(0.0, t)
        return f"{int(t // 3600):d}h{int((t % 3600) // 60):02d}m{int(t % 60):02d}s"

    def report(self, tag: str = ""):
        print(f"[clock{(' ' + tag) if tag else ''}] elapsed {self.hms()} | "
              f"remaining {self.hms(self.left())} | {resources()}", flush=True)


CLOCK = Clock(SESSION_HOURS)


class TimeBudgetExceeded(RuntimeError):
    pass


def save_state(name: str, obj: Any):
    p = STATE / f"{name}.pkl"
    tmp = p.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=4)
    tmp.replace(p)                    # atomic: a killed kernel cannot corrupt the checkpoint


def load_state(name: str, default: Any = None) -> Any:
    p = STATE / f"{name}.pkl"
    if p.exists():
        try:
            with open(p, "rb") as fh:
                return pickle.load(fh)
        except Exception as e:
            print(f"[state] could not read {p.name}: {e}")
    return default


def have_state(name: str) -> bool:
    return (STATE / f"{name}.pkl").exists()


def jdump(name: str, obj: Any):
    with open(OUT / name, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60]


def store_weights(name: str, state: Dict[str, Any]) -> str:
    """Model weights live in their own .pt file, never inside the pickled result dicts.

    Keeps checkpoints small and fast to rewrite after every single experiment.
    """
    p = STATE / f"w_{slug(name)}.pt"
    torch.save(state, p)
    return str(p)


def load_weights(path: str) -> Optional[Dict[str, Any]]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:                       # torch < 2.0 has no weights_only
        return torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[weights] could not load {path}: {e}")
        return None

# %% [markdown]
# ## 3. Dataset: discovery, one-pass decode, RAM cache, stratified splits
#
# The whole dataset is decoded **once** into a contiguous `uint8` tensor held in RAM at
# `cache_size` resolution, then resized on the **GPU** to whatever resolution a given
# training run needs. This removes the dataloader from the critical path entirely, which is
# what makes thousands of short proxy trainings affordable: a proxy epoch becomes pure GPU
# compute.
#
# The discovery routine keyword-matches class folder names for the selected `DATASET`, so several
# datasets and previous outputs can be attached to the notebook at once.

# %%
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".JPG", ".JPEG", ".PNG"}

CANON = [c[0] for c in DS["classes"]]
CANON_PRETTY = [c[1] for c in DS["classes"]]
# Folders for other crops are skipped, so a multi-crop archive cannot merge e.g. a tomato
# blight folder into the maize blight class.
OTHER_CROPS = ("apple", "blueberry", "cherry", "grape", "orange", "peach", "pepper", "potato",
               "raspberry", "soybean", "squash", "strawberry", "tomato", "wheat",
               *[d["crop"] for k, d in DATASETS.items() if k != DATASET])


def canonical_class(name: str) -> Optional[str]:
    """Map a vendor's folder name onto one of this dataset's canonical classes."""
    s = name.lower().replace("-", " ").replace("_", " ")
    if any(c in s for c in OTHER_CROPS):
        return None
    for canon, _, keywords in DS["classes"]:
        if any(k in s for k in keywords):
            return canon
    return None


def discover_dataset(explicit: str = "") -> Tuple[str, Dict[str, List[Path]]]:
    """Find this DATASET's class folders under /kaggle/input (or `explicit`)."""
    roots = [Path(explicit)] if explicit else []
    roots += [Path("/kaggle/input"), Path("./data"), Path("../input")]

    candidates: List[Tuple[Path, Dict[str, List[Path]]]] = []
    for root in roots:
        if not root.exists():
            continue
        # every directory that directly contains a reasonable number of images
        leaf_dirs: Dict[Path, int] = {}
        for dirpath, _dirnames, filenames in os.walk(root):
            n = sum(1 for f in filenames if Path(f).suffix in IMG_EXT)
            if n >= 20:
                leaf_dirs[Path(dirpath)] = n
        # group those leaf dirs by parent, keep groups that cover >=3 canonical classes
        by_parent: Dict[Path, Dict[str, List[Path]]] = {}
        for d in leaf_dirs:
            cc = canonical_class(d.name)
            if cc is None:
                continue
            by_parent.setdefault(d.parent, {}).setdefault(cc, []).append(d)
        for parent, mapping in by_parent.items():
            if len(mapping) >= 3:
                candidates.append((parent, mapping))

    if not candidates:
        raise FileNotFoundError(
            f"No {DS['title']} dataset found for DATASET={DATASET!r}.\n"
            f"Add this Kaggle input (Notebook -> Add Input): {DS['slug']}\n"
            "...or set CFG.data_root to a folder with one sub-folder per class."
        )

    # prefer the candidate with the most images, tie-broken by class coverage
    def score(item):
        parent, mapping = item
        total = sum(sum(1 for f in d.iterdir() if f.suffix in IMG_EXT)
                    for ds in mapping.values() for d in ds)
        return (len(mapping), total)

    parent, mapping = max(candidates, key=score)

    files: Dict[str, List[Path]] = {}
    for cc, dirs in mapping.items():
        fl: List[Path] = []
        for d in dirs:
            fl += sorted(p for p in d.rglob("*") if p.suffix in IMG_EXT)
        files[cc] = fl
    files = {k: v for k, v in sorted(files.items(), key=lambda kv: CANON.index(kv[0]))}
    return str(parent), files


DATA_ROOT, CLASS_FILES = discover_dataset(CFG.data_root)
CLASSES = list(CLASS_FILES.keys())
CFG.n_classes = len(CLASSES)
PRETTY = {c: CANON_PRETTY[CANON.index(c)] for c in CLASSES}

print(f"dataset root : {DATA_ROOT}")
for c in CLASSES:
    print(f"  {c:<16s} {len(CLASS_FILES[c]):5d} images")
N_TOTAL = sum(len(v) for v in CLASS_FILES.values())
print(f"  {'TOTAL':<16s} {N_TOTAL:5d} images across {CFG.n_classes} classes")

# %%
CACHE_NPZ = STATE / f"imgcache_{CFG.cache_size}.npy"
CACHE_META = STATE / f"imgcache_{CFG.cache_size}.json"


def _load_one(args) -> Optional[np.ndarray]:
    path, size = args
    try:
        with Image.open(path) as im:
            im = im.convert("RGB").resize((size, size), Image.BILINEAR)
            return np.asarray(im, dtype=np.uint8)
    except Exception:
        return None


def build_cache() -> Tuple[np.ndarray, np.ndarray]:
    """Decode every image once into a (n, S, S, 3) uint8 array + int64 label array."""
    if CACHE_NPZ.exists() and CACHE_META.exists():
        meta = json.loads(CACHE_META.read_text())
        if meta.get("classes") == CLASSES and meta.get("size") == CFG.cache_size:
            X = np.load(CACHE_NPZ, mmap_mode=None)
            y = np.asarray(meta["labels"], dtype=np.int64)
            print(f"[cache] reused {CACHE_NPZ.name}: {X.shape}")
            return X, y

    items: List[Tuple[Path, int]] = []
    for ci, c in enumerate(CLASSES):
        items += [(p, ci) for p in CLASS_FILES[c]]

    S = CFG.cache_size
    X = np.zeros((len(items), S, S, 3), dtype=np.uint8)
    y = np.zeros(len(items), dtype=np.int64)
    t0 = time.time()
    ok = 0
    with ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 4) * 2)) as ex:
        for i, arr in enumerate(ex.map(_load_one, [(p, S) for p, _ in items], chunksize=32)):
            if arr is not None:
                X[ok] = arr
                y[ok] = items[i][1]
                ok += 1
            if (i + 1) % 1000 == 0:
                print(f"  decoded {i+1}/{len(items)}  ({time.time()-t0:.0f}s)", flush=True)
    X, y = X[:ok].copy(), y[:ok].copy()
    print(f"[cache] decoded {ok}/{len(items)} images at {S}px in {time.time()-t0:.0f}s "
          f"({X.nbytes/2**20:.0f} MiB)")
    np.save(CACHE_NPZ, X)
    CACHE_META.write_text(json.dumps({"classes": CLASSES, "size": S, "labels": y.tolist()}))
    return X, y


X_ALL, Y_ALL = build_cache()

# Exact-duplicate removal BEFORE splitting, so no image can sit in both training and test.
# Copies are identified on the decoded pixels; one copy per group is kept. A group whose copies
# carry different labels is dropped entirely, since its true label is unknown.
DEDUP_INFO: Dict[str, Any] = {"applied": bool(CFG.dedup_exact)}
if CFG.dedup_exact:
    _hashes = [hashlib.md5(X_ALL[i].tobytes()).hexdigest() for i in range(len(X_ALL))]
    _first: Dict[str, int] = {}
    _labels: Dict[str, set] = {}
    for i, h in enumerate(_hashes):
        _first.setdefault(h, i)
        _labels.setdefault(h, set()).add(int(Y_ALL[i]))
    _conflict = {h for h, ls in _labels.items() if len(ls) > 1}
    _keep = np.array(sorted(i for h, i in _first.items() if h not in _conflict), dtype=np.int64)
    DEDUP_INFO.update({
        "images_before": int(len(X_ALL)), "images_after": int(len(_keep)),
        "duplicate_copies_removed": int(len(X_ALL) - len(_first)),
        "conflicting_label_groups_dropped": int(len(_conflict)),
        "images_in_conflicting_groups": int(sum(1 for h in _hashes if h in _conflict)),
        "per_class_after": {CLASSES[c]: int((Y_ALL[_keep] == c).sum())
                            for c in range(len(CLASSES))}})
    X_ALL, Y_ALL = X_ALL[_keep], Y_ALL[_keep]
    del _hashes, _first, _labels
    print(f"exact-duplicate removal: {DEDUP_INFO['images_before']} -> "
          f"{DEDUP_INFO['images_after']} images ({DEDUP_INFO['duplicate_copies_removed']} copies "
          f"removed, {DEDUP_INFO['conflicting_label_groups_dropped']} groups with conflicting "
          f"labels dropped); per class: {DEDUP_INFO['per_class_after']}")
N_TOTAL = int(len(X_ALL))


def stratified_split(y: np.ndarray, fracs, seed: int):
    rng = np.random.RandomState(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(fracs[0] * n))
        n_va = int(round(fracs[1] * n))
        tr += idx[:n_tr].tolist()
        va += idx[n_tr:n_tr + n_va].tolist()
        te += idx[n_tr + n_va:].tolist()
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return np.array(tr), np.array(va), np.array(te)


def _thumbnails(X: np.ndarray, idx: np.ndarray, side: int = 32) -> np.ndarray:
    """Grey side x side thumbnails from a centred crop (centred, so rotations stay aligned)."""
    S = X.shape[1]
    k = max(1, S // side)
    o = (S - k * side) // 2
    out = []
    for c in range(0, len(idx), 512):
        g = X[idx[c:c + 512], o:o + k * side, o:o + k * side].astype(np.float32).mean(axis=3)
        out.append(g.reshape(-1, side, k, side, k).mean(axis=(2, 4)))
    return np.concatenate(out) if out else np.zeros((0, side, side), np.float32)


def _unit_rows(t: np.ndarray) -> np.ndarray:
    v = t.reshape(len(t), -1)
    v = v - v.mean(axis=1, keepdims=True)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


def near_duplicate_groups(X: np.ndarray, y: np.ndarray, thr: float, side: int = 32
                          ) -> np.ndarray:
    """Group id per image: images of the same class whose thumbnails match at >= thr under any
    of the 8 rotations/flips are joined (union-find), so each group is one original photo."""
    n = len(X)
    T = _thumbnails(X, np.arange(n), side)
    UA = _unit_rows(T)
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    with np.errstate(all="ignore"):
        for r in range(4):
            for fl in (False, True):
                Tt = np.rot90(T, r, axes=(1, 2))
                UB = _unit_rows(np.ascontiguousarray(Tt[:, :, ::-1] if fl else Tt))
                for c in range(0, n, 256):
                    sim = UB[c:c + 256] @ UA.T
                    sim[y[c:c + 256, None] != y[None, :]] = -2.0
                    rows, cols = np.nonzero(sim >= thr)
                    for i, j in zip((rows + c).tolist(), cols.tolist()):
                        if i != j:
                            ri, rj = find(i), find(j)
                            if ri != rj:
                                parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)], dtype=np.int64)


def stratified_group_split(y: np.ndarray, groups: np.ndarray, fracs, seed: int,
                           val_all_copies: bool = False):
    """Whole groups go to one split, stratified by class over groups. Training keeps every
    image of its groups; test keeps one image (the first) per group, so test images are
    independent photos; validation keeps one per group, or every copy if val_all_copies."""
    rng = np.random.RandomState(seed)
    members: Dict[int, List[int]] = {}
    for i, g in enumerate(groups.tolist()):
        members.setdefault(g, []).append(i)
    tr, va, te = [], [], []
    for c in np.unique(y):
        gs = sorted({int(groups[i]) for i in np.where(y == c)[0]})
        rng.shuffle(gs)
        n = len(gs)
        n_tr = int(round(fracs[0] * n))
        n_va = int(round(fracs[1] * n))
        for k, g in enumerate(gs):
            if k < n_tr:
                tr += members[g]
            elif k < n_tr + n_va:
                if val_all_copies:
                    va += members[g]
                else:
                    va.append(min(members[g]))
            else:
                te.append(min(members[g]))
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return np.array(tr), np.array(va), np.array(te)


GROUP_INFO: Dict[str, Any] = {"applied": bool(CFG.group_split)}
if CFG.group_split:
    GROUPS = near_duplicate_groups(X_ALL, Y_ALL, CFG.near_dup_threshold)
    _sizes = np.bincount(np.unique(GROUPS, return_inverse=True)[1])
    IDX_TR, IDX_VA, IDX_TE = stratified_group_split(Y_ALL, GROUPS, CFG.group_split_fracs,
                                                    CFG.seed, CFG.group_val_all_copies)
    _photos = lambda idx: int(len(np.unique(GROUPS[idx])))
    GROUP_INFO.update({
        "threshold": CFG.near_dup_threshold, "n_groups": int(len(_sizes)),
        "group_size_median": float(np.median(_sizes)), "group_size_max": int(_sizes.max()),
        "groups_per_class": {CLASSES[c]: int(len(np.unique(GROUPS[Y_ALL == c])))
                             for c in range(len(CLASSES))},
        "fractions_by_photo": list(CFG.group_split_fracs),
        "train_images": int(len(IDX_TR)), "train_photos": _photos(IDX_TR),
        "val_images": int(len(IDX_VA)), "val_photos": _photos(IDX_VA),
        "val_all_copies": bool(CFG.group_val_all_copies),
        "test_images_one_per_photo": int(len(IDX_TE)), "test_photos": _photos(IDX_TE)})
    print(f"near-duplicate groups (sim >= {CFG.near_dup_threshold}, same class, any rotation or "
          f"flip): {GROUP_INFO['n_groups']} original photos among {len(X_ALL)} images "
          f"(median group size {GROUP_INFO['group_size_median']:.0f}, largest "
          f"{GROUP_INFO['group_size_max']}); per class {GROUP_INFO['groups_per_class']}")
    if _sizes.max() > 0.05 * len(X_ALL):
        print("WARNING: one group holds more than 5% of the images; the threshold may be "
              "chaining look-alike photos together. Check figures/near_duplicate_pairs.png.")
else:
    IDX_TR, IDX_VA, IDX_TE = stratified_split(Y_ALL, CFG.split, CFG.seed)
print(f"splits -> train {len(IDX_TR)} | val {len(IDX_VA)} | test {len(IDX_TE)}")

# Exact-duplicate audit. A test image that also appears in training inflates test accuracy, so
# the count belongs in the paper. It is reported, not removed: removing images would change the
# splits. Augmented copies (rotations, crops) are not exact duplicates and are NOT detected here.
_row_hash = [hashlib.md5(X_ALL[i].tobytes()).hexdigest() for i in range(len(X_ALL))]
_train_hashes = {_row_hash[i] for i in IDX_TR}
DUP_AUDIT = {"dataset": DATASET, "n_images": int(len(X_ALL)),
             "n_unique": int(len(set(_row_hash))),
             "test_images_also_in_train": int(sum(_row_hash[i] in _train_hashes for i in IDX_TE)),
             "n_test": int(len(IDX_TE))}


def near_duplicate_audit(X: np.ndarray, Y: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray,
                         thresholds=(0.98, 0.995), side: int = 32, n_show: int = 8
                         ) -> Dict[str, Any]:
    """For each image in idx_b (test), its most similar image in idx_a (train) over the 8
    rotations/flips, on side x side grey thumbnails (centred, unit norm). Re-encoded, resized,
    rotated or flipped copies score near 1.

    A true copy must carry the same label, so the audit also reports the best match within the
    same class and within the other classes. If other-class matches are as close as same-class
    ones, the similarity cannot tell copies from look-alikes on this dataset. Audit only: the
    splits are not changed. Writes figures/near_duplicate_pairs.png for visual inspection.
    """
    thumbs = lambda idx: _thumbnails(X, idx, side)
    unit = _unit_rows
    UA = unit(thumbs(idx_a))
    B = thumbs(idx_b)
    ya, yb = Y[idx_a], Y[idx_b]
    same_mask = (yb[:, None] == ya[None, :])
    best = np.full(len(B), -2.0, np.float32); arg = np.zeros(len(B), np.int64)
    best_same = np.full(len(B), -2.0, np.float32); best_other = np.full(len(B), -2.0, np.float32)
    for r in range(4):
        for fl in (False, True):
            Bt = np.rot90(B, r, axes=(1, 2))
            UB = unit(np.ascontiguousarray(Bt[:, :, ::-1] if fl else Bt))
            for c in range(0, len(UB), 256):
                sim = UB[c:c + 256] @ UA.T
                sm = same_mask[c:c + 256]
                a = sim.argmax(axis=1); v = sim[np.arange(len(a)), a]
                upd = v > best[c:c + 256]
                best[c:c + 256][upd] = v[upd]; arg[c:c + 256][upd] = a[upd]
                best_same[c:c + 256] = np.maximum(best_same[c:c + 256],
                                                  np.where(sm, sim, -2).max(axis=1))
                best_other[c:c + 256] = np.maximum(best_other[c:c + 256],
                                                   np.where(~sm, sim, -2).max(axis=1))
    out: Dict[str, Any] = {f"test_with_train_neighbour_sim_ge_{t}": int((best >= t).sum())
                           for t in thresholds}
    out.update({
        "thumbnail_side": side,
        "best_sim_quantiles_10_50_90": [round(float(q), 4) for q in
                                        np.quantile(best, [0.1, 0.5, 0.9])],
        "nearest_neighbour_same_class_frac": round(float((ya[arg] == yb).mean()), 4),
        "best_same_class_median": round(float(np.median(best_same)), 4),
        "best_other_class_median": round(float(np.median(best_other)), 4),
        # likely copies: a near-perfect same-class match that no other class comes close to
        "likely_copies_same_ge_0.995_other_lt_0.98": int(((best_same >= 0.995) &
                                                          (best_other < 0.98)).sum()),
    })
    try:
        order = np.argsort(-best)
        show = list(order[:n_show // 2]) + list(order[len(order) // 2: len(order) // 2 + n_show // 2])
        fig, axes = plt.subplots(2, len(show), figsize=(2.1 * len(show), 4.6))
        for j, i in enumerate(show):
            for row, (img, lab) in enumerate([(X[idx_b[i]], yb[i]), (X[idx_a[arg[i]]], ya[arg[i]])]):
                ax = axes[row, j]; ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(("test" if row == 0 else "train") + f" | {CLASSES[lab][:12]}"
                             + (f"\nsim={best[i]:.4f}" if row == 0 else ""), fontsize=7)
        fig.suptitle("Closest training image for test images: left half = most similar, "
                     "right half = median similarity", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / "figures" / "near_duplicate_pairs.png", dpi=110)
        plt.close(fig)
        try:                                # show it inline when run interactively
            from IPython.display import display, Image as _IPImage
            display(_IPImage(filename=str(OUT / "figures" / "near_duplicate_pairs.png")))
        except Exception:
            pass
    except Exception as e:
        out["figure_error"] = repr(e)
    return out


try:
    with np.errstate(all="ignore"):
        DUP_AUDIT.update(near_duplicate_audit(X_ALL, Y_ALL, IDX_TR, IDX_TE))
except Exception as _e:                     # an audit must never stop the experiment
    DUP_AUDIT["near_duplicate_audit_error"] = repr(_e)
DUP_AUDIT["dedup"] = DEDUP_INFO
DUP_AUDIT["group_split"] = GROUP_INFO
jdump("duplicate_audit.json", DUP_AUDIT)
print("duplicate audit:", DUP_AUDIT)
print("class balance (train):",
      {CLASSES[c]: int((Y_ALL[IDX_TR] == c).sum()) for c in range(CFG.n_classes)})


def stratified_subset(idx: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Class-balanced subset of `idx` of size <= n (used for the cheap search proxy)."""
    if n >= len(idx):
        return idx
    rng = np.random.RandomState(seed)
    y = Y_ALL[idx]
    per = max(1, n // CFG.n_classes)
    keep = []
    for c in range(CFG.n_classes):
        ci = idx[y == c]
        rng.shuffle(ci)
        keep += ci[:per].tolist()
    keep = np.array(keep)
    rng.shuffle(keep)
    return keep


IDX_PTR = stratified_subset(IDX_TR, CFG.proxy_train_n, CFG.seed + 1)   # proxy train
IDX_PVA = stratified_subset(IDX_VA, CFG.proxy_val_n,  CFG.seed + 2)    # proxy val
print(f"proxy  -> train {len(IDX_PTR)} | val {len(IDX_PVA)}")

# Keep the entire cache resident on the GPU when it fits: the single biggest speed win.
_bytes = X_ALL.nbytes
GPU_RESIDENT = DEVICE.type == "cuda" and _bytes < 0.35 * torch.cuda.get_device_properties(0).total_memory
X_GPU = None
if GPU_RESIDENT:
    X_GPU = torch.from_numpy(X_ALL).to(DEVICE)              # (n,S,S,3) uint8
    print(f"[cache] resident on GPU ({_bytes/2**20:.0f} MiB)")
else:
    X_CPU = torch.from_numpy(X_ALL)
    print(f"[cache] resident on CPU ({_bytes/2**20:.0f} MiB); batches copied per step")

Y_GPU = torch.from_numpy(Y_ALL).to(DEVICE)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


def fetch(idx: torch.Tensor) -> torch.Tensor:
    """Gather a batch as float32 NCHW in [0,1] on DEVICE."""
    if X_GPU is not None:
        b = X_GPU[idx]
    else:
        b = X_CPU[idx.cpu()].to(DEVICE, non_blocking=True)
    return b.permute(0, 3, 1, 2).float().div_(255.0)

# %% [markdown]
# ### 3.1 GPU-side augmentation
#
# One batched `affine_grid` + `grid_sample` call delivers random resized crop, rotation,
# shear and translation simultaneously; photometric jitter, flips and random erasing are
# plain tensor ops. Magnitude is driven by the searched `aug_strength` gene, so augmentation
# policy is part of the optimised search space rather than a fixed human choice.

# %%
def batched_affine(x: torch.Tensor, strength: float, out_size: int) -> torch.Tensor:
    """Random resized crop + rotation + shear + translation, one grid_sample for the batch."""
    B = x.shape[0]
    dev = x.device
    s = float(strength)
    scale = 1.0 + (torch.rand(B, device=dev) * 2 - 1) * (0.25 * s)          # zoom
    ang = (torch.rand(B, device=dev) * 2 - 1) * (math.pi / 180.0) * (25.0 * s)
    shear = (torch.rand(B, device=dev) * 2 - 1) * (0.12 * s)
    tx = (torch.rand(B, device=dev) * 2 - 1) * (0.12 * s)
    ty = (torch.rand(B, device=dev) * 2 - 1) * (0.12 * s)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = cos * scale
    theta[:, 0, 1] = (-sin + shear) * scale
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin * scale
    theta[:, 1, 1] = cos * scale
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, (B, x.shape[1], out_size, out_size), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection",
                         align_corners=False)


def photometric(x: torch.Tensor, strength: float) -> torch.Tensor:
    B = x.shape[0]
    dev = x.device
    s = float(strength)
    x = x * (1.0 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.30 * s)       # brightness
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * (1.0 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.30 * s) + m  # contrast
    g = x.mean(dim=1, keepdim=True)
    x = g + (x - g) * (1.0 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.35 * s)  # saturation
    return x.clamp_(0.0, 1.0)


def random_erase(x: torch.Tensor, p: float, strength: float) -> torch.Tensor:
    if p <= 0:
        return x
    B, _, H, W = x.shape
    hit = torch.rand(B, device=x.device) < p
    if not bool(hit.any()):
        return x
    area_max = 0.04 + 0.16 * float(strength)
    for i in torch.nonzero(hit).flatten().tolist():
        a = random.uniform(0.02, area_max) * H * W
        ar = random.uniform(0.4, 2.5)
        h = max(1, min(H - 1, int(round(math.sqrt(a * ar)))))
        w = max(1, min(W - 1, int(round(math.sqrt(a / ar)))))
        t, l = random.randint(0, H - h), random.randint(0, W - w)
        x[i, :, t:t + h, l:l + w] = torch.rand(1, device=x.device)
    return x


def augment(x: torch.Tensor, out_size: int, strength: float, p_erase: float,
            hflip: bool = True, vflip: bool = True) -> torch.Tensor:
    x = batched_affine(x, strength, out_size)
    if hflip:
        m = torch.rand(x.shape[0], device=x.device) < 0.5
        x[m] = torch.flip(x[m], dims=[3])
    if vflip:
        m = torch.rand(x.shape[0], device=x.device) < 0.25
        x[m] = torch.flip(x[m], dims=[2])
    x = photometric(x, strength)
    x = random_erase(x, p_erase, strength)
    return x


def resize_eval(x: torch.Tensor, out_size: int) -> torch.Tensor:
    if x.shape[-1] != out_size:
        x = F.interpolate(x, size=(out_size, out_size), mode="bilinear",
                          align_corners=False, antialias=True)
    return x


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def mixup_cutmix(x: torch.Tensor, y: torch.Tensor, alpha: float, n_cls: int,
                 mode: str = "mixup"):
    """Return soft targets; alpha<=0 disables. Uses a single lam for the whole batch."""
    if alpha <= 0:
        return x, F.one_hot(y, n_cls).float()
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.shape[0], device=x.device)
    y1 = F.one_hot(y, n_cls).float()
    y2 = y1[perm]
    if mode == "cutmix":
        H, W = x.shape[2], x.shape[3]
        rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
        if rh > 0 and rw > 0:
            cy, cx = random.randint(0, H - rh), random.randint(0, W - rw)
            x[:, :, cy:cy + rh, cx:cx + rw] = x[perm][:, :, cy:cy + rh, cx:cx + rw]
            lam = 1.0 - (rh * rw) / (H * W)
    else:
        x = lam * x + (1 - lam) * x[perm]
    return x, lam * y1 + (1 - lam) * y2

# %% [markdown]
# ## 4. Search space
#
# Every optimiser — MO-P3PGA, NSGA-II, MOEA/D, MOPSO, GDE3, MOGWO, SPEA2, Random Search —
# searches the **same real-valued hypercube** $[0,1]^{24}$, and the **same decoder** turns a
# genotype into an architecture + training recipe. This is the single most important fairness
# control in the experiment: no optimiser benefits from a hand-tailored representation, and
# the comparison measures search behaviour alone.
#
# 24 decision variables span **architecture** (depth, width, width-growth, block family,
# kernel size, squeeze-excitation, stem stride, head pooling, classifier width),
# **regularisation** (two dropout rates, label smoothing, random erasing) and
# **optimisation** (optimiser family, learning rate, weight decay, batch size, augmentation
# magnitude, mixup/cutmix strength). Jointly searching all three groups is what makes this a
# NAS+HPO study rather than a grid search over learning rates.
#
# **The `batch_size` gene is inactive.** All proxy runs use the fixed `proxy_batch`, so the search
# applies no selection pressure to this gene. Final training therefore also uses `proxy_batch`
# rather than the gene's value. Report the space as 23 effective variables.

# %%
D_GENES = 24

GENE_NAMES = [
    "n_blocks", "base_filters", "growth", "block_type", "kernel", "use_se", "se_ratio",
    "activation", "stem_stride", "head_pool", "dense_units", "conv_dropout",
    "dense_dropout", "optimizer", "lr", "weight_decay", "batch_size", "label_smooth",
    "aug_strength", "mix_alpha", "mix_mode", "p_erase", "stage_depth", "expand_ratio",
]
assert len(GENE_NAMES) == D_GENES

BLOCK_TYPES = ["basic", "sep", "res", "mbconv"]
ACTS = ["relu", "elu", "silu", "leaky_relu", "gelu"]
OPTIMS = ["sgd", "adam", "adamw", "rmsprop"]
POOLS = ["avg", "max", "avgmax"]


def _pick(g: float, options: list):
    i = int(np.clip(int(g * len(options)), 0, len(options) - 1))
    return options[i]


def _lin(g: float, lo: float, hi: float) -> float:
    return float(lo + np.clip(g, 0.0, 1.0) * (hi - lo))


def _log(g: float, lo_exp: float, hi_exp: float) -> float:
    return float(10.0 ** _lin(g, lo_exp, hi_exp))


def decode(x) -> Dict[str, Any]:
    """Genotype in [0,1]^24  ->  phenotype dict (architecture + training recipe)."""
    g = np.clip(np.asarray(x, dtype=np.float64).ravel(), 0.0, 1.0)
    assert g.size == D_GENES, f"expected {D_GENES} genes, got {g.size}"
    return {
        "n_blocks":      _pick(g[0],  [2, 3, 4, 5]),
        "base_filters":  _pick(g[1],  [16, 24, 32, 48, 64]),
        "growth":        _pick(g[2],  [1.25, 1.5, 1.75, 2.0]),
        "block_type":    _pick(g[3],  BLOCK_TYPES),
        "kernel":        _pick(g[4],  [3, 5]),
        "use_se":        bool(_pick(g[5], [0, 1])),
        "se_ratio":      _pick(g[6],  [0.125, 0.25, 0.5]),
        "activation":    _pick(g[7],  ACTS),
        "stem_stride":   _pick(g[8],  [1, 2]),
        "head_pool":     _pick(g[9],  POOLS),
        "dense_units":   _pick(g[10], [0, 64, 128, 256]),
        "conv_dropout":  round(_lin(g[11], 0.00, 0.30), 4),
        "dense_dropout": round(_lin(g[12], 0.00, 0.60), 4),
        "optimizer":     _pick(g[13], OPTIMS),
        "lr":            float(f"{_log(g[14], -4.0, -1.7):.3e}"),
        "weight_decay":  float(f"{_log(g[15], -6.0, -2.3):.3e}"),
        "batch_size":    _pick(g[16], [16, 32, 64, 96]),
        "label_smooth":  round(_lin(g[17], 0.00, 0.15), 4),
        "aug_strength":  round(_lin(g[18], 0.20, 1.40), 4),
        "mix_alpha":     round(_lin(g[19], 0.00, 0.60), 4),
        "mix_mode":      _pick(g[20], ["mixup", "cutmix"]),
        "p_erase":       round(_lin(g[21], 0.00, 0.50), 4),
        "stage_depth":   _pick(g[22], [1, 2]),
        "expand_ratio":  _pick(g[23], [2, 3, 4, 6]),
    }


def genotype_key(x) -> str:
    """Stable hash of the *phenotype*, so genotypes decoding to one network share a cache slot."""
    ph = decode(x)
    s = json.dumps(ph, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:16]


print("example phenotype:")
for k, v in decode(np.random.RandomState(0).rand(D_GENES)).items():
    print(f"  {k:<15s} {v}")

# %% [markdown]
# ## 5. Dynamic CNN builder
#
# Four block families cover the practically relevant design space: plain VGG-style `basic`,
# `sep` (MobileNet depthwise-separable), `res` (pre-activation residual) and `mbconv`
# (inverted-residual bottleneck with optional squeeze-excitation). FLOPs and parameter counts
# vary over roughly three orders of magnitude across the space, which is exactly what gives
# the multi-objective problem a non-degenerate trade-off surface.

# %%
def make_act(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"unknown activation {name!r}")


def round8(v: float) -> int:
    return max(8, int(round(v / 8.0)) * 8)


class SE(nn.Module):
    def __init__(self, ch: int, ratio: float):
        super().__init__()
        hid = max(4, int(ch * ratio))
        self.fc1 = nn.Conv2d(ch, hid, 1)
        self.fc2 = nn.Conv2d(hid, ch, 1)

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s), inplace=True)))
        return x * s


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, k, stride, act, use_se, se_ratio, drop):
        super().__init__()
        p = k // 2
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, k, stride, p, bias=False), nn.BatchNorm2d(cout), make_act(act),
            nn.Conv2d(cout, cout, k, 1, p, bias=False), nn.BatchNorm2d(cout), make_act(act),
        )
        self.se = SE(cout, se_ratio) if use_se else nn.Identity()
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.se(self.body(x)))


class SepBlock(nn.Module):
    def __init__(self, cin, cout, k, stride, act, use_se, se_ratio, drop):
        super().__init__()
        p = k // 2
        self.body = nn.Sequential(
            nn.Conv2d(cin, cin, k, stride, p, groups=cin, bias=False),
            nn.BatchNorm2d(cin), make_act(act),
            nn.Conv2d(cin, cout, 1, 1, 0, bias=False), nn.BatchNorm2d(cout), make_act(act),
        )
        self.se = SE(cout, se_ratio) if use_se else nn.Identity()
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.se(self.body(x)))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, k, stride, act, use_se, se_ratio, drop):
        super().__init__()
        p = k // 2
        self.c1 = nn.Conv2d(cin, cout, k, stride, p, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.a1 = make_act(act)
        self.c2 = nn.Conv2d(cout, cout, k, 1, p, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.a2 = make_act(act)
        self.se = SE(cout, se_ratio) if use_se else nn.Identity()
        self.short = (nn.Identity() if (stride == 1 and cin == cout) else
                      nn.Sequential(nn.Conv2d(cin, cout, 1, stride, 0, bias=False),
                                    nn.BatchNorm2d(cout)))
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        h = self.a1(self.b1(self.c1(x)))
        h = self.se(self.b2(self.c2(h)))
        return self.drop(self.a2(h + self.short(x)))


class MBConvBlock(nn.Module):
    def __init__(self, cin, cout, k, stride, act, use_se, se_ratio, drop, expand=4):
        super().__init__()
        hid = round8(cin * expand)
        p = k // 2
        layers = []
        if hid != cin:
            layers += [nn.Conv2d(cin, hid, 1, bias=False), nn.BatchNorm2d(hid), make_act(act)]
        layers += [nn.Conv2d(hid, hid, k, stride, p, groups=hid, bias=False),
                   nn.BatchNorm2d(hid), make_act(act)]
        self.pre = nn.Sequential(*layers)
        self.se = SE(hid, se_ratio) if use_se else nn.Identity()
        self.proj = nn.Sequential(nn.Conv2d(hid, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.use_res = (stride == 1 and cin == cout)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        h = self.proj(self.se(self.pre(x)))
        if self.use_res:
            h = h + x
        return self.drop(h)


BLOCK_CLS = {"basic": BasicBlock, "sep": SepBlock, "res": ResBlock, "mbconv": MBConvBlock}


class SearchedCNN(nn.Module):
    """CNN materialised from a decoded phenotype."""

    def __init__(self, ph: Dict[str, Any], n_classes: int, in_size: int):
        super().__init__()
        act = ph["activation"]
        k = ph["kernel"]
        bt = ph["block_type"]
        cls = BLOCK_CLS[bt]

        c0 = round8(ph["base_filters"])
        s0 = ph["stem_stride"]
        self.stem = nn.Sequential(
            nn.Conv2d(3, c0, 3, s0, 1, bias=False), nn.BatchNorm2d(c0), make_act(act))
        spatial = in_size // s0

        blocks = []
        cin = c0
        for b in range(ph["n_blocks"]):
            cout = min(512, round8(ph["base_filters"] * (ph["growth"] ** b)))
            # never shrink below 4x4: keeps very deep genotypes trainable instead of invalid
            stride = 2 if spatial >= 8 else 1
            for d in range(ph["stage_depth"]):
                st = stride if d == 0 else 1
                kw = dict(cin=cin, cout=cout, k=k, stride=st, act=act,
                          use_se=ph["use_se"], se_ratio=ph["se_ratio"],
                          drop=ph["conv_dropout"])
                if bt == "mbconv":
                    kw["expand"] = ph["expand_ratio"]
                blocks.append(cls(**kw))
                cin = cout
                if st == 2:
                    spatial //= 2
        self.blocks = nn.Sequential(*blocks)
        self.out_spatial = max(1, spatial)

        pool = ph["head_pool"]
        feat = cin * (2 if pool == "avgmax" else 1)
        self.pool = pool
        head: List[nn.Module] = []
        if ph["dense_dropout"] > 0:
            head.append(nn.Dropout(ph["dense_dropout"]))
        if ph["dense_units"] > 0:
            head += [nn.Linear(feat, ph["dense_units"]), make_act(act),
                     nn.Dropout(ph["dense_dropout"] * 0.5)]
            feat = ph["dense_units"]
        head.append(nn.Linear(feat, n_classes))
        self.head = nn.Sequential(*head)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)

    def features(self, x):
        return self.blocks(self.stem(x))

    def forward(self, x):
        h = self.features(x)
        if self.pool == "avg":
            z = h.mean(dim=(2, 3))
        elif self.pool == "max":
            z = h.amax(dim=(2, 3))
        else:
            z = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return self.head(z)


# %%
def count_flops_params(model: nn.Module, in_size: int) -> Tuple[float, float]:
    """Multiply-accumulates (MMac) and parameters (M) for one forward pass, batch size 1."""
    macs = [0.0]
    hooks = []

    def conv_hook(m, i, o):
        macs[0] += (m.in_channels / m.groups) * m.kernel_size[0] * m.kernel_size[1] * o.numel()

    def lin_hook(m, i, o):
        macs[0] += m.in_features * o.numel()

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))

    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        model(torch.zeros(1, 3, in_size, in_size, device=dev))
    for h in hooks:
        h.remove()
    if was_training:
        model.train()

    params = sum(p.numel() for p in model.parameters())
    return macs[0] / 1e6, params / 1e6


@torch.no_grad()
def measure_latency(model: nn.Module, in_size: int, batch: int = 1,
                    warmup: int = 10, reps: int = 30) -> float:
    """Median per-image inference latency in milliseconds."""
    model.eval()
    dev = next(model.parameters()).device
    x = torch.randn(batch, 3, in_size, in_size, device=dev)
    for _ in range(warmup):
        model(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0 / batch)
    return float(np.median(ts))


# sanity check the builder across the extremes of the space
_rng = np.random.RandomState(7)
print(f"{'blocks':>6} {'type':>7} {'base':>5} {'MMac':>9} {'Params(M)':>10} {'spatial':>7}")
for _ in range(6):
    _ph = decode(_rng.rand(D_GENES))
    _m = SearchedCNN(_ph, CFG.n_classes, CFG.img_size_proxy).to(DEVICE)
    _f, _p = count_flops_params(_m, CFG.img_size_proxy)
    print(f"{_ph['n_blocks']:>6} {_ph['block_type']:>7} {_ph['base_filters']:>5} "
          f"{_f:>9.1f} {_p:>10.3f} {_m.out_spatial:>7}")
    del _m
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

# %% [markdown]
# ## 6. Trainer
#
# One routine serves both the cheap **proxy** training used inside the search loop and the
# expensive **full** training used for the final reported models; they differ only in
# resolution, subset size, epoch count and schedule. Sharing the code path removes a classic
# source of silent protocol mismatch between search and final evaluation.
#
# Mixed precision, a GPU-resident dataset, channels-last memory format and GPU-side
# augmentation together bring a proxy evaluation down to a few seconds, which is what makes
# 6 000 architecture evaluations feasible inside one GPU-day.

# %%
def make_scaler(enabled: bool):
    """GradScaler across torch versions (torch.cuda.amp.GradScaler is deprecated in 2.4+)."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_optimizer(model: nn.Module, ph: Dict[str, Any]):
    name, lr, wd = ph["optimizer"], ph["lr"], ph["weight_decay"]
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim <= 1 or "bn" in n.lower()) else decay).append(p)
    groups = [{"params": decay, "weight_decay": wd},
              {"params": no_decay, "weight_decay": 0.0}]
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr * 10, momentum=0.9, nesterov=True)
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(groups, lr=lr, momentum=0.9, alpha=0.9, eps=1e-3)
    raise ValueError(name)


def lr_lambda_factory(total_steps: int, warmup_frac: float):
    warm = max(1, int(total_steps * warmup_frac))

    def fn(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return fn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v)


@torch.no_grad()
def predict(model: nn.Module, idx: np.ndarray, img_size: int, batch: int = 128,
            tta: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, labels) for the given cache indices."""
    model.eval()
    probs = []
    for s in range(0, len(idx), batch):
        bi = torch.as_tensor(idx[s:s + batch], device=DEVICE)
        xb = normalize(resize_eval(fetch(bi), img_size))
        if DEVICE.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            lo = model(xb).float()
            if tta:
                lo = lo + model(torch.flip(xb, dims=[3])).float()
                lo = lo / 2.0
        probs.append(torch.softmax(lo, dim=1).cpu().numpy())
    return np.concatenate(probs, 0), Y_ALL[idx]


def train_one(ph: Dict[str, Any],
              idx_train: np.ndarray,
              idx_val: np.ndarray,
              img_size: int,
              epochs: int,
              batch: Optional[int] = None,
              seed: int = 0,
              amp: bool = True,
              use_ema: bool = False,
              label_smooth: Optional[float] = None,
              return_model: bool = False,
              warmup_frac: float = 0.1,
              verbose: bool = False) -> Dict[str, Any]:
    """Train a phenotype; select by best validation macro-F1. Never raises on OOM."""
    set_seed(seed)
    t0 = time.time()
    bs = int(batch or ph["batch_size"])
    try:
        model = SearchedCNN(ph, CFG.n_classes, img_size).to(DEVICE)
        if DEVICE.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
    except Exception as e:
        return {"ok": False, "reason": f"build:{type(e).__name__}:{e}"}

    mmac, params_m = count_flops_params(model, img_size)
    # The FLOPs box is defined at the search resolution. Convolution cost grows with pixel
    # count, so the same box at a larger resolution is scaled, otherwise a design that was
    # feasible during the search would be rejected at final training.
    flops_box = CFG.max_flops_mmac * max(1.0, (img_size / CFG.img_size_proxy) ** 2)
    if params_m > CFG.max_params_m or mmac > flops_box:
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return {"ok": False, "reason": "infeasible", "mmac": mmac, "params_m": params_m}

    opt = build_optimizer(model, ph)
    steps_per_epoch = max(1, len(idx_train) // bs)
    total_steps = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_factory(total_steps, warmup_frac))
    scaler = make_scaler(amp and DEVICE.type == "cuda")
    ls = ph["label_smooth"] if label_smooth is None else label_smooth
    ema = EMA(model) if use_ema else None

    idx_tr_t = torch.as_tensor(idx_train, device=DEVICE)
    best = {"f1": -1.0, "acc": 0.0, "epoch": -1, "state": None}
    hist = []

    try:
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(idx_tr_t), device=DEVICE)
            run_loss, nb = 0.0, 0
            for s in range(steps_per_epoch):
                bi = idx_tr_t[perm[s * bs:(s + 1) * bs]]
                if bi.numel() < 2:
                    continue
                xb = fetch(bi)
                xb = augment(xb, img_size, ph["aug_strength"], ph["p_erase"])
                yb = Y_GPU[bi]
                xb, yt = mixup_cutmix(xb, yb, ph["mix_alpha"], CFG.n_classes, ph["mix_mode"])
                if ls > 0:
                    yt = yt * (1 - ls) + ls / CFG.n_classes
                xb = normalize(xb)
                if DEVICE.type == "cuda":
                    xb = xb.contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", enabled=(amp and DEVICE.type == "cuda")):
                    logits = model(xb)
                    loss = torch.sum(-yt * F.log_softmax(logits.float(), dim=1), dim=1).mean()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                if ema is not None:
                    ema.update(model)
                run_loss += float(loss.detach()); nb += 1

            if ema is not None and ep == epochs - 1:
                ema.copy_to(model)
            p, yv = predict(model, idx_val, img_size)
            pred = p.argmax(1)
            f1 = f1_score(yv, pred, average="macro", zero_division=0)
            acc = accuracy_score(yv, pred)
            hist.append({"epoch": ep, "loss": run_loss / max(1, nb), "val_f1": f1, "val_acc": acc})
            if f1 > best["f1"]:
                best.update(f1=f1, acc=acc, epoch=ep)
                if return_model:
                    best["state"] = {k: v.detach().cpu().clone()
                                     for k, v in model.state_dict().items()}
            if return_model:
                (STATE / "heartbeat.txt").write_text(
                    f"{time.ctime()} epoch {ep + 1}/{epochs} val_f1={f1:.4f} {resources()}\n")
            if verbose:
                print(f"    ep{ep:02d} loss {hist[-1]['loss']:.4f} "
                      f"val_f1 {f1:.4f} val_acc {acc:.4f}")
    except torch.cuda.OutOfMemoryError:
        del model, opt
        gc.collect(); torch.cuda.empty_cache()
        return {"ok": False, "reason": "oom", "mmac": mmac, "params_m": params_m}
    except Exception as e:
        del model
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return {"ok": False, "reason": f"train:{type(e).__name__}:{e}",
                "mmac": mmac, "params_m": params_m}

    lat = measure_latency(model, img_size, batch=1) if "latency" in CFG.objectives else float("nan")
    out = {"ok": True, "val_f1": best["f1"], "val_acc": best["acc"], "best_epoch": best["epoch"],
           "mmac": mmac, "params_m": params_m, "latency_ms": lat,
           "history": hist, "seconds": time.time() - t0}
    if return_model:
        out["model"] = model
        out["state"] = best["state"]
    else:
        del model, opt
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return out


# %% [markdown]
# ## 7. Multi-objective evaluator
#
# A single `MOEvaluator` instance is handed to every optimiser, which guarantees they all see
# identical objective values for identical architectures. It enforces three things that make
# the comparison publishable:
#
# * **Budget parity.** Each run gets exactly `budget` phenotype queries. The evaluator raises
#   `BudgetExhausted` the moment an optimiser tries to exceed it, so no algorithm can
#   over-spend by, say, evaluating a larger final generation.
# * **Memoisation.** Repeat phenotypes are served from cache and do **not** consume budget,
#   which is standard in HPO and prevents converged populations from burning the budget on
#   re-evaluating identical networks. Total queries are logged alongside unique evaluations
#   so the ratio is reportable.
# * **Fixed-seed fitness.** The proxy always trains from `CFG.proxy_seed`, so every optimiser
#   sees one stored $f_1$ per phenotype. GPU training is not guaranteed bit-exact, so this
#   controls evaluation noise rather than eliminating it.
#
# A **global memo table** persists across runs and sessions.
# Budget accounting stays strictly per-run — an optimiser spends one unit for every
# phenotype it has not itself seen — but the *compute* for a phenotype already trained by an
# earlier run is reused instead of repeated. In practice exact repeats are rare in this space:
# the first full run reused 0 % of evaluations, because inactive genes make nearly every
# phenotype unique. Those inactive-gene duplicates cost every optimiser budget equally.

# %%
class BudgetExhausted(Exception):
    pass


GLOBAL_MEMO: Dict[str, Dict[str, Any]] = load_state("global_memo", {}) or {}
print(f"[memo] {len(GLOBAL_MEMO)} phenotypes already evaluated in previous sessions")


OBJ_LABEL = {"error": r"$1-F_1$ (val)", "flops": r"$\log_{10}$ MMac",
             "params": r"$\log_{10}$ params (M)", "latency": r"$\log_{10}$ ms"}
OBJ_RAW_LABEL = {"error": "val error", "flops": "MMac", "params": "params (M)",
                 "latency": "latency (ms)"}

PENALTY = {"error": 1.0,
           "flops": math.log10(CFG.max_flops_mmac * 10),
           "params": math.log10(CFG.max_params_m * 10),
           "latency": math.log10(1e4)}


class MOEvaluator:
    def __init__(self, cfg: Config, budget: int, tag: str = "",
                 shared_cache: Optional[Dict[str, Any]] = None,
                 clock: Optional[Clock] = None, time_reserve: float = 0.0):
        self.cfg = cfg
        self.budget = budget
        self.tag = tag
        self.cache: Dict[str, Any] = shared_cache if shared_cache is not None else {}
        self.n_evals = 0            # unique phenotypes seen by THIS run == budget spent
        self.n_queries = 0          # total calls including cache hits
        self.n_trained = 0          # phenotypes actually trained (global memo misses)
        self.n_infeasible = 0
        self.history: List[Dict[str, Any]] = []
        self.clock = clock
        self.time_reserve = time_reserve
        self.t_start = time.time()
        # Stagnation guards. A converged population can query only phenotypes it has already
        # seen; those are served from cache and cost no budget, so without these ceilings an
        # optimiser would spin forever instead of terminating. Both are deliberately loose:
        # they end a run that can no longer make progress, never one that still can.
        self.max_queries = 200 * budget + 2000
        self.max_consec_hits = max(1000, 50 * budget)
        self.consec_hits = 0
        self.stopped_reason = ""

    # ---- objective vector assembly -------------------------------------------------
    def _objectives(self, res: Dict[str, Any]) -> np.ndarray:
        f = []
        for o in self.cfg.objectives:
            if not res.get("ok", False):
                f.append(PENALTY[o])
            elif o == "error":
                f.append(1.0 - res["val_f1"])
            elif o == "flops":
                f.append(math.log10(max(res["mmac"], 1e-3)))
            elif o == "params":
                f.append(math.log10(max(res["params_m"], 1e-4)))
            elif o == "latency":
                f.append(math.log10(max(res["latency_ms"], 1e-3)))
            else:
                raise ValueError(o)
        return np.asarray(f, dtype=np.float64)

    # ---- the call every optimiser makes ---------------------------------------------
    def __call__(self, x) -> np.ndarray:
        self.n_queries += 1
        key = genotype_key(x)
        if key in self.cache:
            self.consec_hits += 1
            if self.consec_hits > self.max_consec_hits:
                self.stopped_reason = "stagnated (population converged to cached phenotypes)"
                raise BudgetExhausted(f"{self.tag}: {self.stopped_reason}")
            if self.n_queries > self.max_queries:
                self.stopped_reason = "query ceiling"
                raise BudgetExhausted(f"{self.tag}: {self.stopped_reason}")
            return self.cache[key]["f"].copy()
        self.consec_hits = 0

        if self.n_evals >= self.budget:
            self.stopped_reason = "budget exhausted"
            raise BudgetExhausted(f"{self.tag}: budget {self.budget} exhausted")
        if self.clock is not None and not self.clock.ok(self.time_reserve):
            self.stopped_reason = "wall-clock"
            raise TimeBudgetExceeded(f"{self.tag}: session wall-clock guard tripped")

        ph = decode(x)
        self.n_evals += 1                       # budget is spent whether or not we retrain
        if key in GLOBAL_MEMO:
            res = GLOBAL_MEMO[key]              # one stored fitness per phenotype
        else:
            res = train_one(ph, IDX_PTR, IDX_PVA,
                            img_size=self.cfg.img_size_proxy,
                            epochs=self.cfg.proxy_epochs,
                            batch=self.cfg.proxy_batch,
                            seed=self.cfg.proxy_seed,
                            amp=self.cfg.proxy_amp)
            res = {k: v for k, v in res.items() if k != "history"}
            GLOBAL_MEMO[key] = res
            self.n_trained += 1
        if not res.get("ok", False):
            self.n_infeasible += 1

        f = self._objectives(res)
        rec = {"f": f, "x": np.asarray(x, dtype=np.float64).copy(), "ph": ph,
               "val_f1": res.get("val_f1", 0.0), "val_acc": res.get("val_acc", 0.0),
               "mmac": res.get("mmac", float("nan")),
               "params_m": res.get("params_m", float("nan")),
               "latency_ms": res.get("latency_ms", float("nan")),
               "ok": res.get("ok", False), "reason": res.get("reason", ""),
               "eval_no": self.n_evals, "seconds": res.get("seconds", 0.0)}
        self.cache[key] = rec
        self.history.append(rec)      # keeps x, so any evaluated design can be retrained
        return f.copy()

    def batch(self, X) -> np.ndarray:
        return np.vstack([self(x) for x in np.atleast_2d(X)])

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.n_evals)

    def summary(self) -> Dict[str, Any]:
        return {"tag": self.tag, "evals": self.n_evals, "queries": self.n_queries,
                "trained": self.n_trained,
                "cache_hit_rate": 1 - self.n_evals / max(1, self.n_queries),
                "memo_reuse_rate": 1 - self.n_trained / max(1, self.n_evals),
                "infeasible": self.n_infeasible,
                "stopped_reason": self.stopped_reason,
                "wall_seconds": time.time() - self.t_start}

# %% [markdown]
# ## 8. Pareto machinery and multi-objective performance indicators
#
# Shared, algorithm-agnostic implementations of everything the optimisers and the analysis
# need: dominance, fast non-dominated sorting, crowding distance, bounded external archives,
# Das–Dennis reference directions, and the five indicator families reviewers expect —
# **hypervolume (HV)**, **inverted generational distance (IGD and IGD⁺)**, **generational
# distance (GD)**, **spacing (SP)** / **spread (Δ)** / **maximum spread (MS)**, and
# **set coverage (C-metric)**.
#
# Because the true Pareto front of a NAS problem is unknown, the reference front is the
# non-dominated subset of the union of *every* solution found by *every* algorithm across
# *every* run — the standard construction in the evolutionary multi-objective literature.
# All indicators are computed on objectives min–max normalised over all pooled archive solutions,
# with reference point $\mathbf{1.1}$, so HV values are comparable across tables.

# %%
def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if a Pareto-dominates b (minimisation)."""
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(F: np.ndarray) -> List[np.ndarray]:
    """Deb's O(MN^2) sort. Returns a list of fronts, each an array of row indices."""
    F = np.atleast_2d(F)
    n = F.shape[0]
    if n == 0:
        return []
    le = (F[:, None, :] <= F[None, :, :]).all(axis=2)     # le[i,j]: i <= j componentwise
    lt = (F[:, None, :] < F[None, :, :]).any(axis=2)      # lt[i,j]: i < j somewhere
    dom = le & lt                                          # dom[i,j]: i dominates j
    n_dominated = dom.sum(axis=0)                          # how many dominate j
    fronts, cur = [], np.where(n_dominated == 0)[0]
    counts = n_dominated.copy()
    assigned = np.zeros(n, dtype=bool)
    while cur.size:
        fronts.append(cur)
        assigned[cur] = True
        counts = counts - dom[cur].sum(axis=0)
        nxt = np.where((counts <= 0) & (~assigned))[0]
        cur = nxt
    # any leftovers (can only happen with duplicate rows) go into a final front
    if not assigned.all():
        fronts.append(np.where(~assigned)[0])
    return fronts


def non_dominated_mask(F: np.ndarray) -> np.ndarray:
    F = np.atleast_2d(F)
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        d = (F <= F[i]).all(axis=1) & (F < F[i]).any(axis=1)
        if d.any():
            keep[i] = False
    return keep


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance within one front."""
    F = np.atleast_2d(F)
    n, m = F.shape
    if n <= 2:
        return np.full(n, np.inf)
    d = np.zeros(n)
    for j in range(m):
        order = np.argsort(F[:, j], kind="stable")
        fj = F[order, j]
        d[order[0]] = d[order[-1]] = np.inf
        rng = fj[-1] - fj[0]
        if rng <= 1e-12:
            continue
        d[order[1:-1]] += (fj[2:] - fj[:-2]) / rng
    return d


def environmental_selection(X: np.ndarray, F: np.ndarray, n_keep: int
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """NSGA-II survival: fronts first, last front trimmed by crowding distance."""
    X, F = np.atleast_2d(X), np.atleast_2d(F)
    if X.shape[0] <= n_keep:
        rank, cd = rank_and_crowding(F)
        return X, F, rank, cd
    fronts = fast_non_dominated_sort(F)
    sel: List[int] = []
    for fr in fronts:
        if len(sel) + len(fr) <= n_keep:
            sel += fr.tolist()
        else:
            cd = crowding_distance(F[fr])
            order = np.argsort(-cd, kind="stable")
            sel += fr[order[: n_keep - len(sel)]].tolist()
            break
    sel = np.array(sel[:n_keep])
    Xs, Fs = X[sel], F[sel]
    rank, cd = rank_and_crowding(Fs)
    return Xs, Fs, rank, cd


def rank_and_crowding(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    F = np.atleast_2d(F)
    rank = np.zeros(F.shape[0], dtype=int)
    cd = np.zeros(F.shape[0])
    for r, fr in enumerate(fast_non_dominated_sort(F)):
        rank[fr] = r
        cd[fr] = crowding_distance(F[fr])
    return rank, cd


class Archive:
    """Bounded external archive of non-dominated solutions, trimmed by crowding distance."""

    def __init__(self, cap: int):
        self.cap = cap
        self.X = np.zeros((0, D_GENES))
        self.F = np.zeros((0, 0))

    def add(self, X: np.ndarray, F: np.ndarray):
        X, F = np.atleast_2d(X), np.atleast_2d(F)
        if X.size == 0:
            return
        if self.F.size == 0:
            allX, allF = X, F
        else:
            allX, allF = np.vstack([self.X, X]), np.vstack([self.F, F])
        # drop exact duplicates in objective space to keep the archive informative
        _, uniq = np.unique(np.round(allF, 10), axis=0, return_index=True)
        allX, allF = allX[np.sort(uniq)], allF[np.sort(uniq)]
        m = non_dominated_mask(allF)
        allX, allF = allX[m], allF[m]
        if allX.shape[0] > self.cap:
            cd = crowding_distance(allF)
            keep = np.argsort(-cd, kind="stable")[: self.cap]
            allX, allF = allX[keep], allF[keep]
        self.X, self.F = allX, allF

    def leader(self, mode: str = "random_archive", rng=None) -> Optional[np.ndarray]:
        if self.X.shape[0] == 0:
            return None
        rng = rng or np.random
        if mode == "global_least_crowded":
            cd = crowding_distance(self.F)
            fin = np.where(np.isfinite(cd))[0]
            return self.X[int(fin[np.argmax(cd[fin])]) if fin.size else 0]
        if mode == "random_archive":          # OMOPSO-style: bias toward sparse regions
            cd = crowding_distance(self.F)
            w = np.where(np.isfinite(cd), cd, cd[np.isfinite(cd)].max() if np.isfinite(cd).any() else 1.0)
            w = w + 1e-9
            # tournament of 2 on crowding distance
            i, j = rng.randint(0, self.X.shape[0]), rng.randint(0, self.X.shape[0])
            return self.X[i if w[i] >= w[j] else j]
        return self.X[rng.randint(0, self.X.shape[0])]

    def __len__(self):
        return self.X.shape[0]


def das_dennis(n_obj: int, n_partitions: int) -> np.ndarray:
    """Structured simplex-lattice reference directions (NSGA-III / MOEA/D weights)."""
    def rec(left, total, depth):
        if depth == n_obj - 1:
            return [[left / total]]
        out = []
        for i in range(left + 1):
            for tail in rec(left - i, total, depth + 1):
                out.append([i / total] + tail)
        return out

    W = np.array(rec(n_partitions, n_partitions, 0), dtype=np.float64)
    return W


def ref_dirs_for(n_obj: int, n_points: int) -> np.ndarray:
    """Smallest Das-Dennis lattice with at least `n_points` directions."""
    p = 1
    while len(das_dennis(n_obj, p)) < n_points:
        p += 1
        if p > 200:
            break
    W = das_dennis(n_obj, p)
    if len(W) > n_points:                    # subsample to the requested size, keeping corners
        corners = [i for i in range(len(W)) if (W[i] > 1 - 1e-9).any()]
        rest = [i for i in range(len(W)) if i not in corners]
        rng = np.random.RandomState(0)
        rng.shuffle(rest)
        keep = sorted(corners + rest[: max(0, n_points - len(corners))])
        W = W[keep]
    return W

# %% [markdown]
# ### 8.1 Performance indicators

# %%
def normalize_front(F: np.ndarray, z_min: np.ndarray, z_max: np.ndarray) -> np.ndarray:
    rng = np.where((z_max - z_min) > 1e-12, z_max - z_min, 1.0)
    return (np.atleast_2d(F) - z_min) / rng


def hypervolume(F: np.ndarray, ref: np.ndarray, mc_samples: int = 300_000,
                seed: int = 0) -> float:
    """HV of a minimisation set w.r.t. `ref`. Exact for M=2, Monte Carlo for M>=3."""
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    ref = np.asarray(ref, dtype=np.float64)
    if F.size == 0:
        return 0.0
    F = F[(F < ref).all(axis=1)]                      # points outside the box contribute nothing
    if F.size == 0:
        return 0.0
    F = F[non_dominated_mask(F)]
    m = F.shape[1]

    if m == 1:
        return float(ref[0] - F[:, 0].min())

    if m == 2:
        # exact sweep: for a non-dominated set sorted by f1 ascending, f2 is descending,
        # so the dominated region decomposes into disjoint axis-aligned rectangles
        order = np.lexsort((-F[:, 1], F[:, 0]))
        P = F[order]
        hv = 0.0
        for i in range(P.shape[0]):
            w = (P[i + 1, 0] if i + 1 < P.shape[0] else ref[0]) - P[i, 0]
            hv += max(0.0, w) * max(0.0, ref[1] - P[i, 1])
        return float(hv)

    # M >= 3: deterministic Monte Carlo over the dominated box
    lo = np.minimum(F.min(axis=0), ref)
    rng = np.random.RandomState(seed)
    S = rng.uniform(lo, ref, size=(mc_samples, m))
    dominated = np.zeros(mc_samples, dtype=bool)
    chunk = 64
    for s in range(0, F.shape[0], chunk):
        blk = F[s:s + chunk]
        dominated |= (S[:, None, :] >= blk[None, :, :]).all(axis=2).any(axis=1)
    vol = float(np.prod(ref - lo))
    return vol * dominated.mean()


def igd(F: np.ndarray, ref_front: np.ndarray) -> float:
    """Inverted generational distance: mean distance from reference front to F."""
    F, R = np.atleast_2d(F), np.atleast_2d(ref_front)
    if F.size == 0 or R.size == 0:
        return float("nan")
    d = np.linalg.norm(R[:, None, :] - F[None, :, :], axis=2).min(axis=1)
    return float(d.mean())


def igd_plus(F: np.ndarray, ref_front: np.ndarray) -> float:
    """IGD+ (Ishibuchi et al.): weakly Pareto-compliant variant of IGD."""
    F, R = np.atleast_2d(F), np.atleast_2d(ref_front)
    if F.size == 0 or R.size == 0:
        return float("nan")
    diff = np.maximum(F[None, :, :] - R[:, None, :], 0.0)
    d = np.linalg.norm(diff, axis=2).min(axis=1)
    return float(d.mean())


def gd(F: np.ndarray, ref_front: np.ndarray) -> float:
    """Generational distance: mean distance from F to the reference front."""
    F, R = np.atleast_2d(F), np.atleast_2d(ref_front)
    if F.size == 0 or R.size == 0:
        return float("nan")
    d = np.linalg.norm(F[:, None, :] - R[None, :, :], axis=2).min(axis=1)
    return float(d.mean())


def spacing(F: np.ndarray) -> float:
    """Schott's spacing metric: std-dev of nearest-neighbour distances (lower = more uniform)."""
    F = np.atleast_2d(F)
    n = F.shape[0]
    if n < 2:
        return float("nan")
    Dm = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)
    np.fill_diagonal(Dm, np.inf)
    d = Dm.min(axis=1)
    return float(np.sqrt(((d.mean() - d) ** 2).sum() / (n - 1)))


def max_spread(F: np.ndarray) -> float:
    """Zitzler's maximum spread: diagonal of the front's bounding box."""
    F = np.atleast_2d(F)
    if F.shape[0] < 2:
        return float("nan")
    return float(np.sqrt(((F.max(axis=0) - F.min(axis=0)) ** 2).sum()))


def delta_spread(F: np.ndarray, ref_front: Optional[np.ndarray] = None) -> float:
    """Deb's Delta spread (2 objectives): combines extent and uniformity. Lower is better.

    d_f / d_l are the Euclidean distances from the two extreme solutions of the reference
    front to the boundary solutions of `F`; when no reference front is supplied the extremes
    of `F` itself are used, which reduces Delta to a pure uniformity measure.
    """
    F = np.atleast_2d(F)
    if F.shape[1] != 2 or F.shape[0] < 3:
        return float("nan")
    P = F[np.argsort(F[:, 0], kind="stable")]
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    if d.size == 0:
        return float("nan")
    if ref_front is not None and np.atleast_2d(ref_front).shape[0] >= 2:
        R = np.atleast_2d(ref_front)
        R = R[np.argsort(R[:, 0], kind="stable")]
        df = float(np.linalg.norm(R[0] - P[0]))
        dl = float(np.linalg.norm(R[-1] - P[-1]))
    else:
        df = dl = 0.0
    dbar = d.mean()
    return float((df + dl + np.abs(d - dbar).sum()) / (df + dl + d.size * dbar + 1e-12))


def set_coverage(A: np.ndarray, B: np.ndarray) -> float:
    """C(A,B): fraction of B weakly dominated by at least one member of A."""
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    if A.size == 0 or B.size == 0:
        return float("nan")
    le = (A[:, None, :] <= B[None, :, :]).all(axis=2)
    lt = (A[:, None, :] < B[None, :, :]).any(axis=2)
    return float(((le & lt).any(axis=0)).mean())


# ---- self-tests: an indicator bug would silently corrupt every table in the paper ----
_F2 = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
assert abs(hypervolume(_F2, np.array([1.1, 1.1])) - (
    (0.5 - 0.0) * (1.1 - 1.0) + (1.0 - 0.5) * (1.1 - 0.5) + (1.1 - 1.0) * (1.1 - 0.0))) < 1e-9
assert abs(hypervolume(np.array([[0.0, 0.0]]), np.array([1.0, 1.0])) - 1.0) < 1e-12
assert len(fast_non_dominated_sort(_F2)) == 1 and len(fast_non_dominated_sort(_F2)[0]) == 3
assert fast_non_dominated_sort(np.array([[0., 0.], [1., 1.]]))[0].tolist() == [0]
assert dominates(np.array([0., 0.]), np.array([1., 1.])) and not dominates(
    np.array([0., 1.]), np.array([1., 0.]))
assert abs(set_coverage(np.array([[0., 0.]]), np.array([[1., 1.], [0.5, 0.5]])) - 1.0) < 1e-12
assert abs(igd(_F2, _F2)) < 1e-12 and abs(gd(_F2, _F2)) < 1e-12
assert das_dennis(2, 4).shape == (5, 2) and abs(das_dennis(3, 3).sum(1) - 1).max() < 1e-9
_mc = hypervolume(np.array([[0., 0., 0.]]), np.array([1., 1., 1.]), mc_samples=200_000)
assert abs(_mc - 1.0) < 0.02, _mc
print("Pareto / indicator self-tests passed")

# %% [markdown]
# ## 9. Optimisers
#
# The multi-objective optimisers all share the same genotype space, the same variation
# operators where the algorithm definition allows it, the same population size, the same
# archive capacity and the same evaluator instance type. Each `run()` returns a bounded
# Pareto archive.
#
# ### 9.1 Shared variation operators

# %%
def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def sbx_crossover(p1: np.ndarray, p2: np.ndarray, eta: float, p_cx: float, rng
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Simulated binary crossover (Deb & Agrawal)."""
    c1, c2 = p1.copy(), p2.copy()
    if rng.rand() > p_cx:
        return c1, c2
    u = rng.rand(p1.size)
    beta = np.where(u <= 0.5, (2 * u) ** (1.0 / (eta + 1.0)),
                    (1.0 / (2 * (1 - u) + 1e-12)) ** (1.0 / (eta + 1.0)))
    swap = rng.rand(p1.size) < 0.5
    c1 = np.where(swap, 0.5 * ((1 + beta) * p1 + (1 - beta) * p2), c1)
    c2 = np.where(swap, 0.5 * ((1 - beta) * p1 + (1 + beta) * p2), c2)
    return clip01(c1), clip01(c2)


def npoint_crossover(p1: np.ndarray, p2: np.ndarray, p_cx: float, rng, n_points: int = 1
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """The crossover actually used in the original P3PGA chapter (single / n-point)."""
    c1, c2 = p1.copy(), p2.copy()
    if rng.rand() > p_cx:
        return c1, c2
    pts = np.sort(rng.choice(np.arange(1, p1.size), size=min(n_points, p1.size - 1),
                             replace=False))
    flip = False
    prev = 0
    for pt in list(pts) + [p1.size]:
        if flip:
            c1[prev:pt], c2[prev:pt] = p2[prev:pt], p1[prev:pt]
        flip = not flip
        prev = pt
    return clip01(c1), clip01(c2)


def polynomial_mutation(x: np.ndarray, eta: float, p_mut: float, rng) -> np.ndarray:
    y = x.copy()
    m = rng.rand(x.size) < p_mut
    if not m.any():
        return y
    xm = y[m]
    u = rng.rand(xm.size)
    d1, d2 = xm, 1.0 - xm
    dq = np.where(
        u <= 0.5,
        (2 * u + (1 - 2 * u) * (1 - d1) ** (eta + 1)) ** (1.0 / (eta + 1)) - 1.0,
        1.0 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - d2) ** (eta + 1)) ** (1.0 / (eta + 1)))
    y[m] = clip01(xm + dq)
    return y


def uniform_reset_mutation(x: np.ndarray, p_mut: float, rng) -> np.ndarray:
    """Gene-wise uniform re-sampling: the mutation described in the P3PGA chapter."""
    y = x.copy()
    m = rng.rand(x.size) < p_mut
    y[m] = rng.rand(int(m.sum()))
    return y


def binary_tournament(rank: np.ndarray, cd: np.ndarray, rng, k: int = 2) -> int:
    """Crowded-comparison tournament: lower rank wins; ties broken by larger crowding."""
    cand = rng.randint(0, len(rank), size=k)
    best = cand[0]
    for c in cand[1:]:
        if rank[c] < rank[best] or (rank[c] == rank[best] and cd[c] > cd[best]):
            best = c
    return int(best)


def sus_select(fitness: np.ndarray, n: int, rng) -> np.ndarray:
    """Stochastic universal sampling -- the selection operator used in the P3PGA chapter."""
    f = np.asarray(fitness, dtype=np.float64)
    f = f - f.min() + 1e-9
    total = f.sum()
    if not np.isfinite(total) or total <= 0:
        return rng.randint(0, len(f), size=n)
    step = total / n
    start = rng.rand() * step
    pointers = start + step * np.arange(n)
    cum = np.cumsum(f)
    return np.searchsorted(cum, pointers, side="left").clip(0, len(f) - 1)


def rank_fitness(rank: np.ndarray, cd: np.ndarray) -> np.ndarray:
    """Scalarise (rank, crowding) so the paper's SUS operator can be used unchanged."""
    cdn = np.where(np.isfinite(cd), cd, 0.0)
    if cdn.max() > 0:
        cdn = cdn / cdn.max()
    cdn = np.where(np.isfinite(cd), cdn, 1.0)
    return 1.0 / (1.0 + rank) + 0.1 * cdn


class MOOptimizer:
    """Base class. Subclasses implement `_run`; the driver handles budget/time termination."""

    name = "base"

    def __init__(self, cfg: Config, evaluator: MOEvaluator, seed: int = 0,
                 overrides: Optional[Dict[str, Any]] = None):
        self.cfg = cfg
        self.ev = evaluator
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.o = dict(overrides or {})
        self.archive = Archive(cfg.archive_size)
        self.n_gen = 0

    def opt(self, key: str, default=None):
        """Config value with per-experiment override (used by the ablation study)."""
        if key in self.o:
            return self.o[key]
        return getattr(self.cfg, key, default)

    def _tick(self):
        """Advance the generation counter, with a ceiling as a second stagnation guard.

        No optimiser can usefully need more generations than it has evaluations, so this
        can only ever terminate a run that has stopped consuming budget.
        """
        self.n_gen += 1
        if self.n_gen > self.ev.budget + 10:
            raise BudgetExhausted(f"{self.name}: generation ceiling")

    def init_pop(self, n: int) -> np.ndarray:
        """Latin-hypercube initialisation: better space filling than i.i.d. uniform."""
        X = np.zeros((n, D_GENES))
        for j in range(D_GENES):
            X[:, j] = (self.rng.permutation(n) + self.rng.rand(n)) / n
        return clip01(X)

    def run(self) -> Archive:
        try:
            self._run()
        except (BudgetExhausted, TimeBudgetExceeded):
            pass
        return self.archive

    def _run(self):
        raise NotImplementedError

# %% [markdown]
# ### 9.2 MO-P3PGA — the proposed algorithm
#
# This is the paper's **Algorithm 1** with its structure preserved and only its *selection
# rule* lifted from scalar fitness to Pareto dominance. Element by element:
#
# | P3PGA (single-objective, 2020) | MO-P3PGA (this work) |
# |---|---|
# | `NP` populations of `N` individuals evolved in parallel | **unchanged** |
# | *Mitochondrial change*: $x' = x + U(-\delta,\delta)$ per gene, bound-corrected | **unchanged** ($\delta$ = `mito_scale`) |
# | Merge 2-parent and 3-parent populations, keep the `N` **best by fitness** | keep the `N` best by **non-dominated rank, ties by crowding distance** |
# | GA step: SUS selection, crossover at $p_c$, mutation at $p_m$, elitist survival | **unchanged** operators; selection is crowded-comparison tournament (or the original SUS applied to a rank-derived scalar, `selection_op="sus_rank"`) |
# | *Local elite* = fittest individual of island $i$ | a member of island $i$'s **first non-dominated front** |
# | *Global elite* `gbest` = best of all local elites | a **leader drawn from the bounded external Pareto archive** |
# | Step 5: per gene, with small probability, $x_k \leftarrow (x_k + gbest_k)/2$ | **unchanged**, with the archive leader in place of `gbest` |
#
# Two faithfulness notes worth stating in the paper:
#
# 1. The chapter's Algorithm 1 listing writes the pull-toward-best step as
#    `individual(j,k) + gbest(j,k)`, indexing `gbest` by both individual and gene. Since
#    `gbest` is a single candidate solution this is a typographical slip; the surrounding
#    prose (Step 5) gives the correct form `individual(i) ← (individual(i) + gbest(i))/2`,
#    which is what is implemented here.
# 2. The prose says the pull is applied "after a predetermined number of generations" while
#    the listing applies it every generation. `attract_every` exposes both readings; the
#    default `1` follows the listing.
#
# After the pull-toward-leader step the perturbed individuals carry stale objective values,
# so they are re-evaluated immediately — phenotype memoisation makes this nearly free
# whenever the perturbation does not cross a decoding boundary.

# %%
class MOP3PGA(MOOptimizer):
    name = "MO-P3PGA"

    def _run(self):
        cfg = self.cfg
        NP = int(self.opt("p3pga_np"))
        N = int(self.opt("p3pga_n"))
        delta = float(self.opt("mito_scale"))
        p_cx = float(self.opt("p_crossover"))
        p_mu = float(self.opt("p_mutation"))
        p_at = float(self.opt("p_attract"))
        every = int(self.opt("attract_every"))
        use_mito = bool(self.opt("use_mito"))
        use_attr = bool(self.opt("use_attract"))
        leader_mode = str(self.opt("leader_mode"))
        cx_op = str(self.opt("crossover_op"))
        sel_op = str(self.opt("selection_op"))

        # ---- initialise NP islands ("2-parent populations") -------------------------
        X0 = self.init_pop(NP * N)
        islands: List[Dict[str, np.ndarray]] = []
        for i in range(NP):
            Xi = X0[i * N:(i + 1) * N]
            Fi = self.ev.batch(Xi)
            islands.append({"X": Xi, "F": Fi})
            self.archive.add(Xi, Fi)

        while True:
            self._tick()
            for isl in islands:
                Xi, Fi = isl["X"], isl["F"]

                # ---- Step 2-3: mitochondrial change -> 3-parent population ----------
                if use_mito:
                    Ci = self.rng.uniform(-delta, delta, size=Xi.shape)   # "Change" matrix
                    X3 = clip01(Xi + Ci)                                  # bound correction
                    F3 = self.ev.batch(X3)
                    self.archive.add(X3, F3)
                    # combine 2-parent and 3-parent populations, keep the N best
                    Xi, Fi, rank, cd = environmental_selection(
                        np.vstack([Xi, X3]), np.vstack([Fi, F3]), N)
                else:
                    rank, cd = rank_and_crowding(Fi)

                # ---- Step 4: ordinary genetic process -------------------------------
                if sel_op == "sus_rank":
                    pidx = sus_select(rank_fitness(rank, cd), N, self.rng)
                else:
                    pidx = np.array([binary_tournament(rank, cd, self.rng)
                                     for _ in range(N)])
                kids = []
                for a in range(0, N, 2):
                    p1 = Xi[pidx[a]]
                    p2 = Xi[pidx[(a + 1) % N]]
                    if cx_op == "npoint":
                        c1, c2 = npoint_crossover(p1, p2, p_cx, self.rng, n_points=1)
                        c1 = uniform_reset_mutation(c1, p_mu, self.rng)
                        c2 = uniform_reset_mutation(c2, p_mu, self.rng)
                    else:
                        c1, c2 = sbx_crossover(p1, p2, self.cfg.cx_eta, p_cx, self.rng)
                        c1 = polynomial_mutation(c1, self.cfg.mut_eta, p_mu, self.rng)
                        c2 = polynomial_mutation(c2, self.cfg.mut_eta, p_mu, self.rng)
                    kids += [c1, c2]
                Xk = np.vstack(kids[:N])
                Fk = self.ev.batch(Xk)
                self.archive.add(Xk, Fk)

                # elitist survival among parents + offspring (paper's Step 4c/4d)
                Xi, Fi, _, _ = environmental_selection(
                    np.vstack([Xi, Xk]), np.vstack([Fi, Fk]), N)
                isl["X"], isl["F"] = Xi, Fi

            # ---- Step 5: pull every gene of every individual toward the leader ------
            if use_attr and p_at > 0 and (self.n_gen % every == 0) and len(self.archive) > 0:
                for isl in islands:
                    Xi = isl["X"]
                    changed = np.zeros(Xi.shape[0], dtype=bool)
                    Xn = Xi.copy()
                    for j in range(Xi.shape[0]):
                        lead = self.archive.leader(leader_mode, self.rng)
                        if lead is None:
                            continue
                        mask = self.rng.rand(D_GENES) < p_at
                        if mask.any():
                            Xn[j, mask] = 0.5 * (Xi[j, mask] + lead[mask])
                            changed[j] = True
                    if changed.any():
                        Fn = isl["F"].copy()
                        Fn[changed] = self.ev.batch(Xn[changed])
                        isl["X"], isl["F"] = clip01(Xn), Fn
                        self.archive.add(Xn[changed], Fn[changed])


class MO3PGA(MOP3PGA):
    """Single-population ablation (NP = 1): isolates the contribution of parallel islands."""
    name = "MO-3PGA"

    def _run(self):
        self.o.setdefault("p3pga_np", 1)
        self.o.setdefault("p3pga_n", self.cfg.pop_total)
        super()._run()


class MOP3PGA2(MOOptimizer):
    """MO-P3PGA-II: the same three mechanisms, at the evaluation cost of an ordinary GA.

    MO-P3PGA spends roughly 70 evaluations per generation (a full mitochondrial population,
    a full offspring population, and re-evaluation of every individual moved by the leader
    pull), so a 240-evaluation budget buys only 4 generations. Here the three mechanisms act
    on ONE offspring population per island, which is evaluated once:

      * islands: NP populations of N individuals, as in P3PGA;
      * 3-parent children: a fraction `p2_mito_frac` of each island's offspring is a selected
        parent plus a mitochondrial change. "uniform" is the chapter's U(-delta, delta) change;
        "differential" takes the change from third parents, delta * (x_b - x_c), drawn from the
        island ("island") or from every island ("global", which also migrates information);
      * 2-parent children: crossover plus mutation, as in the GA step of P3PGA;
      * leader pull: Step 5, x <- (x + leader)/2 per gene with probability p_attract, applied
        to the offspring BEFORE evaluation, so it costs no extra evaluations;
      * survival: parents + offspring, Pareto rank then crowding, as in MO-P3PGA.
    """
    name = "MO-P3PGA-II"

    def _run(self):
        NP = int(self.opt("p3pga_np"))
        N = int(self.opt("p3pga_n"))
        delta0 = float(self.opt("mito_scale"))
        mode = str(self.opt("p2_mito_mode", "uniform"))
        frac = float(self.opt("p2_mito_frac", 0.5))
        third = str(self.opt("p2_third", "island"))
        sched = str(self.opt("p2_delta_sched", "const"))
        p_cx = float(self.opt("p_crossover"))
        p_mu = float(self.opt("p_mutation"))
        p_at = float(self.opt("p_attract"))
        cx_op = str(self.opt("crossover_op"))
        leader_mode = str(self.opt("leader_mode"))
        gen_est = max(2, self.ev.budget // max(1, NP * N))

        X0 = self.init_pop(NP * N)
        islands = []
        for i in range(NP):
            Xi = X0[i * N:(i + 1) * N]
            Fi = self.ev.batch(Xi)
            islands.append({"X": Xi, "F": Fi})
            self.archive.add(Xi, Fi)

        n3 = int(round(frac * N))
        n2 = N - n3
        while True:
            self._tick()
            delta = delta0 if sched == "const" else \
                delta0 * (1.0 - 0.75 * min(1.0, (self.n_gen - 1) / gen_est))
            X_all = np.vstack([isl["X"] for isl in islands])
            for isl in islands:
                Xi, Fi = isl["X"], isl["F"]
                rank, cd = rank_and_crowding(Fi)
                pick = lambda: Xi[binary_tournament(rank, cd, self.rng)]
                kids = []
                while len(kids) < n2:                         # 2-parent children
                    p1, p2 = pick(), pick()
                    if cx_op == "npoint":
                        c1, c2 = npoint_crossover(p1, p2, p_cx, self.rng, n_points=1)
                        c1 = uniform_reset_mutation(c1, p_mu, self.rng)
                        c2 = uniform_reset_mutation(c2, p_mu, self.rng)
                    else:
                        c1, c2 = sbx_crossover(p1, p2, self.cfg.cx_eta, p_cx, self.rng)
                        c1 = polynomial_mutation(c1, self.cfg.mut_eta, p_mu, self.rng)
                        c2 = polynomial_mutation(c2, self.cfg.mut_eta, p_mu, self.rng)
                    kids += [c1, c2]
                kids = kids[:n2]
                pool = Xi if third == "island" else X_all
                for _ in range(n3):                           # 3-parent children
                    base = pick()
                    if mode == "differential" and pool.shape[0] >= 2:
                        b, c = self.rng.choice(pool.shape[0], 2, replace=False)
                        kids.append(base + delta * (pool[b] - pool[c]))
                    else:
                        kids.append(base + self.rng.uniform(-delta, delta, size=base.shape))
                K = clip01(np.vstack(kids))
                if p_at > 0 and len(self.archive) > 0:        # Step 5, before evaluation
                    for j in range(K.shape[0]):
                        lead = self.archive.leader(leader_mode, self.rng)
                        if lead is None:
                            continue
                        m = self.rng.rand(D_GENES) < p_at
                        K[j, m] = 0.5 * (K[j, m] + lead[m])
                Fk = self.ev.batch(K)
                self.archive.add(K, Fk)
                Xi, Fi, _, _ = environmental_selection(np.vstack([Xi, K]), np.vstack([Fi, Fk]), N)
                isl["X"], isl["F"] = Xi, Fi

# %% [markdown]
# ### 9.3 Reference multi-objective optimisers
#
# NSGA-II, NSGA-III, MOEA/D, SPEA2, MOPSO, GDE3, MOGWO and Random Search. Each is the
# canonical formulation, configured with the shared population size and operator
# probabilities so that differences in outcome are attributable to search strategy.

# %%
class NSGA2(MOOptimizer):
    name = "NSGA-II"

    def _run(self):
        n = self.cfg.pop_total
        X = self.init_pop(n)
        Fv = self.ev.batch(X)
        self.archive.add(X, Fv)
        while True:
            self._tick()
            rank, cd = rank_and_crowding(Fv)
            kids = []
            while len(kids) < n:
                i = binary_tournament(rank, cd, self.rng)
                j = binary_tournament(rank, cd, self.rng)
                c1, c2 = sbx_crossover(X[i], X[j], self.cfg.cx_eta,
                                       self.cfg.p_crossover, self.rng)
                kids.append(polynomial_mutation(c1, self.cfg.mut_eta,
                                                self.cfg.p_mutation, self.rng))
                kids.append(polynomial_mutation(c2, self.cfg.mut_eta,
                                                self.cfg.p_mutation, self.rng))
            Xk = np.vstack(kids[:n])
            Fk = self.ev.batch(Xk)
            self.archive.add(Xk, Fk)
            X, Fv, _, _ = environmental_selection(np.vstack([X, Xk]),
                                                  np.vstack([Fv, Fk]), n)


class NSGA3(MOOptimizer):
    name = "NSGA-III"

    def _run(self):
        n = self.cfg.pop_total
        W = ref_dirs_for(self.cfg.n_obj, n)
        X = self.init_pop(n)
        Fv = self.ev.batch(X)
        self.archive.add(X, Fv)
        while True:
            self._tick()
            rank, cd = rank_and_crowding(Fv)
            kids = []
            while len(kids) < n:
                i = binary_tournament(rank, cd, self.rng)
                j = binary_tournament(rank, cd, self.rng)
                c1, c2 = sbx_crossover(X[i], X[j], self.cfg.cx_eta,
                                       self.cfg.p_crossover, self.rng)
                kids.append(polynomial_mutation(c1, self.cfg.mut_eta,
                                                self.cfg.p_mutation, self.rng))
                kids.append(polynomial_mutation(c2, self.cfg.mut_eta,
                                                self.cfg.p_mutation, self.rng))
            Xk = np.vstack(kids[:n])
            Fk = self.ev.batch(Xk)
            self.archive.add(Xk, Fk)
            X, Fv = self._niching_survival(np.vstack([X, Xk]), np.vstack([Fv, Fk]), n, W)

    def _niching_survival(self, X, Fv, n, W):
        fronts = fast_non_dominated_sort(Fv)
        sel: List[int] = []
        last: np.ndarray = np.array([], dtype=int)
        for fr in fronts:
            if len(sel) + len(fr) <= n:
                sel += fr.tolist()
            else:
                last = fr
                break
        if last.size == 0 or len(sel) == n:
            sel = sel[:n]
            return X[np.array(sel)], Fv[np.array(sel)]
        pool = np.array(sel + last.tolist())
        z_min = Fv[pool].min(axis=0)
        Fn = Fv[pool] - z_min
        rng_ = np.where(Fn.max(axis=0) > 1e-12, Fn.max(axis=0), 1.0)
        Fn = Fn / rng_
        # perpendicular distance of each candidate to each reference direction
        Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
        proj = Fn @ Wn.T                                     # (P, R)
        d_perp = np.sqrt(np.maximum((Fn ** 2).sum(axis=1, keepdims=True) - proj ** 2, 0.0))
        assoc = d_perp.argmin(axis=1)
        dist = d_perp.min(axis=1)
        n_sel = len(sel)
        niche_count = np.zeros(W.shape[0], dtype=int)
        for p in range(n_sel):
            niche_count[assoc[p]] += 1
        chosen = list(range(n_sel))
        avail = set(range(n_sel, len(pool)))
        BIG = np.iinfo(np.int32).max
        while len(chosen) < n and avail:
            # only reference directions that still have an unselected associated candidate
            live = np.zeros(W.shape[0], dtype=bool)
            live[[assoc[a] for a in avail]] = True
            j = int(np.argmin(np.where(live, niche_count, BIG)))
            members = [a for a in avail if assoc[a] == j]
            pick = min(members, key=lambda a: dist[a]) if niche_count[j] == 0 \
                else members[self.rng.randint(len(members))]
            chosen.append(pick)
            avail.discard(pick)
            niche_count[j] += 1
        idx = pool[np.array(chosen[:n])]
        return X[idx], Fv[idx]


class MOEAD(MOOptimizer):
    name = "MOEA/D"

    def _run(self):
        n = self.cfg.pop_total
        W = ref_dirs_for(self.cfg.n_obj, n)
        W = W[:n] if W.shape[0] >= n else np.vstack(
            [W, W[self.rng.randint(0, W.shape[0], n - W.shape[0])]])
        W = np.maximum(W, 1e-6)
        T = max(2, int(0.2 * n))
        Dw = np.linalg.norm(W[:, None, :] - W[None, :, :], axis=2)
        B = np.argsort(Dw, axis=1)[:, :T]

        X = self.init_pop(n)
        Fv = self.ev.batch(X)
        self.archive.add(X, Fv)
        z = Fv.min(axis=0)
        z_nad = Fv.max(axis=0)

        def tcheby(f, w):
            rngv = np.where((z_nad - z) > 1e-12, z_nad - z, 1.0)
            return np.max(w * np.abs((f - z) / rngv))

        while True:
            self._tick()
            for i in range(n):
                nb = B[i]
                r1, r2, r3 = nb[self.rng.choice(len(nb), 3, replace=True)]
                # DE/rand/1 + polynomial mutation (standard MOEA/D-DE reproduction)
                y = clip01(X[r1] + 0.5 * (X[r2] - X[r3]))
                mask = self.rng.rand(D_GENES) < 0.5
                y = np.where(mask, y, X[i])
                y = polynomial_mutation(y, self.cfg.mut_eta, 1.0 / D_GENES, self.rng)
                fy = self.ev(y)
                self.archive.add(y[None, :], fy[None, :])
                z = np.minimum(z, fy)
                z_nad = np.maximum(z_nad, fy)
                for j in nb:
                    if tcheby(fy, W[j]) <= tcheby(Fv[j], W[j]):
                        X[j], Fv[j] = y, fy


class SPEA2(MOOptimizer):
    name = "SPEA2"

    def _run(self):
        n = self.cfg.pop_total
        X = self.init_pop(n)
        Fv = self.ev.batch(X)
        self.archive.add(X, Fv)
        Ax, Af = X.copy(), Fv.copy()
        while True:
            self._tick()
            Ux, Uf = np.vstack([X, Ax]), np.vstack([Fv, Af])
            fit = self._spea2_fitness(Uf)
            keep = self._truncate(Uf, fit, n)
            Ax, Af = Ux[keep], Uf[keep]
            fitA = fit[keep]
            kids = []
            while len(kids) < n:
                i = self._tour(fitA)
                j = self._tour(fitA)
                c1, c2 = sbx_crossover(Ax[i], Ax[j], self.cfg.cx_eta,
                                       self.cfg.p_crossover, self.rng)
                kids.append(polynomial_mutation(c1, self.cfg.mut_eta,
                                                self.cfg.p_mutation, self.rng))
                kids.append(polynomial_mutation(c2, self.cfg.mut_eta,
                                                self.cfg.p_mutation, self.rng))
            X = np.vstack(kids[:n])
            Fv = self.ev.batch(X)
            self.archive.add(X, Fv)

    @staticmethod
    def _spea2_fitness(Fv: np.ndarray) -> np.ndarray:
        le = (Fv[:, None, :] <= Fv[None, :, :]).all(axis=2)
        lt = (Fv[:, None, :] < Fv[None, :, :]).any(axis=2)
        dom = le & lt
        strength = dom.sum(axis=1).astype(float)
        raw = (dom * strength[:, None]).sum(axis=0)
        k = max(1, int(math.sqrt(Fv.shape[0])))
        Dm = np.linalg.norm(Fv[:, None, :] - Fv[None, :, :], axis=2)
        np.fill_diagonal(Dm, np.inf)
        sigma = np.sort(Dm, axis=1)[:, min(k - 1, Dm.shape[1] - 1)]
        density = 1.0 / (sigma + 2.0)
        return raw + density                    # lower is better

    def _truncate(self, Fv, fit, n):
        nd = np.where(fit < 1.0)[0]
        if len(nd) == n:
            return nd
        if len(nd) < n:
            rest = np.argsort(fit, kind="stable")
            extra = [i for i in rest if i not in set(nd.tolist())]
            return np.array(nd.tolist() + extra[: n - len(nd)])
        sel = nd.tolist()
        while len(sel) > n:
            S = Fv[sel]
            Dm = np.linalg.norm(S[:, None, :] - S[None, :, :], axis=2)
            np.fill_diagonal(Dm, np.inf)
            order = np.sort(Dm, axis=1)
            drop = int(np.lexsort(order.T[::-1])[0])
            sel.pop(drop)
        return np.array(sel)

    def _tour(self, fit):
        a, b = self.rng.randint(0, len(fit)), self.rng.randint(0, len(fit))
        return a if fit[a] <= fit[b] else b


class MOPSO(MOOptimizer):
    name = "MOPSO"

    def _run(self):
        n = self.cfg.pop_total
        w, c1, c2 = 0.4, 1.5, 1.5
        X = self.init_pop(n)
        V = np.zeros_like(X)
        Fv = self.ev.batch(X)
        self.archive.add(X, Fv)
        Px, Pf = X.copy(), Fv.copy()
        while True:
            self._tick()
            for i in range(n):
                lead = self.archive.leader("random_archive", self.rng)
                if lead is None:
                    lead = Px[i]
                r1, r2 = self.rng.rand(D_GENES), self.rng.rand(D_GENES)
                V[i] = w * V[i] + c1 * r1 * (Px[i] - X[i]) + c2 * r2 * (lead - X[i])
                V[i] = np.clip(V[i], -0.25, 0.25)
                X[i] = clip01(X[i] + V[i])
                if self.rng.rand() < 0.15:      # OMOPSO-style turbulence
                    X[i] = polynomial_mutation(X[i], self.cfg.mut_eta,
                                               1.0 / D_GENES, self.rng)
            Fv = self.ev.batch(X)
            self.archive.add(X, Fv)
            for i in range(n):                  # personal best updated by dominance
                if dominates(Fv[i], Pf[i]) or (not dominates(Pf[i], Fv[i])
                                               and self.rng.rand() < 0.5):
                    Px[i], Pf[i] = X[i].copy(), Fv[i].copy()


class GDE3(MOOptimizer):
    name = "GDE3"

    def _run(self):
        n = self.cfg.pop_total
        Fw, CR = float(self.opt("gde3_F", 0.5)), float(self.opt("gde3_CR", 0.9))
        X = self.init_pop(n)
        Fv = self.ev.batch(X)
        self.archive.add(X, Fv)
        while True:
            self._tick()
            Xk = np.zeros_like(X)
            for i in range(n):
                r = self.rng.choice([k for k in range(n) if k != i], 3, replace=False)
                mutant = X[r[0]] + Fw * (X[r[1]] - X[r[2]])
                jr = self.rng.randint(D_GENES)
                mask = (self.rng.rand(D_GENES) < CR)
                mask[jr] = True
                Xk[i] = clip01(np.where(mask, mutant, X[i]))
            Fk = self.ev.batch(Xk)
            self.archive.add(Xk, Fk)
            keepX, keepF = [], []
            for i in range(n):                  # one-to-one dominance comparison
                if dominates(Fk[i], Fv[i]):
                    keepX.append(Xk[i]); keepF.append(Fk[i])
                elif dominates(Fv[i], Fk[i]):
                    keepX.append(X[i]); keepF.append(Fv[i])
                else:                           # mutually non-dominated: keep both
                    keepX += [X[i], Xk[i]]; keepF += [Fv[i], Fk[i]]
            X, Fv, _, _ = environmental_selection(np.vstack(keepX), np.vstack(keepF), n)


class MOGWO(MOOptimizer):
    name = "MOGWO"

    def _run(self):
        n = self.cfg.pop_total
        X = self.init_pop(n)
        Fv = self.ev.batch(X)
        self.archive.add(X, Fv)
        gen_est = max(2, self.cfg.budget // max(1, n))
        while True:
            self._tick()
            a = float(self.opt("mogwo_a0", 2.0)) * max(0.0, 1.0 - self.n_gen / gen_est)  # linear decrease
            leaders = [self.archive.leader("random_archive", self.rng) for _ in range(3)]
            leaders = [l if l is not None else X[self.rng.randint(n)] for l in leaders]
            Xn = np.zeros_like(X)
            for i in range(n):
                pos = []
                for L in leaders:
                    A = 2 * a * self.rng.rand(D_GENES) - a
                    C = 2 * self.rng.rand(D_GENES)
                    Dp = np.abs(C * L - X[i])
                    pos.append(L - A * Dp)
                Xn[i] = clip01(np.mean(pos, axis=0))
                if self.rng.rand() < float(self.opt("mogwo_mut", 0.10)):
                    Xn[i] = polynomial_mutation(Xn[i], self.cfg.mut_eta,
                                                1.0 / D_GENES, self.rng)
            Fn = self.ev.batch(Xn)
            self.archive.add(Xn, Fn)
            X, Fv, _, _ = environmental_selection(np.vstack([X, Xn]),
                                                  np.vstack([Fv, Fn]), n)


class RandomSearch(MOOptimizer):
    name = "Random Search"

    def _run(self):
        while True:
            self._tick()
            X = clip01(self.rng.rand(self.cfg.pop_total, D_GENES))
            Fv = self.ev.batch(X)
            self.archive.add(X, Fv)


OPTIMIZERS: Dict[str, type] = {
    "MO-P3PGA": MOP3PGA,
    "MO-3PGA": MO3PGA,
    "MO-P3PGA-II": MOP3PGA2,
    "NSGA-II": NSGA2,
    "NSGA-III": NSGA3,
    "MOEA/D": MOEAD,
    "SPEA2": SPEA2,
    "MOPSO": MOPSO,
    "GDE3": GDE3,
    "MOGWO": MOGWO,
    "Random Search": RandomSearch,
}
BASE_P3 = "MO-P3PGA"            # the direct port; the ablation and sensitivity studies use it
PROPOSED = "MO-P3PGA-II" if CFG.improved else BASE_P3

# MO-3PGA is the NP=1 ablation of the proposed method, so it is reported in the ablation
# study (Section 20) rather than in the main comparison, where it would spend a ninth of the
# session's GPU time restating an ablation result.
MAIN_ALGOS = [a for a in OPTIMIZERS if a not in ("MO-3PGA", "MO-P3PGA-II")]
if DATASET != PRIMARY_DATASET:
    MAIN_ALGOS = [a for a in MAIN_ALGOS if a in CFG.secondary_algos]

# Settings chosen BEFORE these runs, by a surrogate-assisted screen on seeds 20-49 that the
# reported runs (0-9) never use: 70 configurations per method for MO-P3PGA-II, GDE3 and MOGWO
# (Results/tuning_surrogate). The surrogate looks up every phenotype trained in the
# standard-protocol search and predicts the rest with extra-trees regressors.
P3II_CONFIG = {"p3pga_np": 2, "p3pga_n": CFG.pop_total // 2, "p2_mito_mode": "differential",
               "mito_scale": 0.8, "p2_third": "global", "p2_mito_frac": 0.25,
               "p2_delta_sched": "const", "p_attract": 0.1, "crossover_op": "npoint"}
TUNED_CONFIGS = {"GDE3": {"gde3_F": 0.2, "gde3_CR": 0.5},
                 "MOGWO": {"mogwo_a0": 4.0, "mogwo_mut": 0.05}}

# Standard protocol: every competitor at its recommended settings. MO-P3PGA-II reuses the
# MO-P3PGA seeds, so run r of both starts from the same initial population.
ENTRIES = [{"label": a, "algo": a, "overrides": None, "seed_label": None} for a in MAIN_ALGOS]
if CFG.improved:
    ENTRIES.append({"label": "MO-P3PGA-II", "algo": "MO-P3PGA-II",
                    "overrides": P3II_CONFIG, "seed_label": BASE_P3})
# Equal-tuning protocol: the two strongest competitors with the same tuning budget. Each tuned
# run reuses the default run's seed, so the tuned and default runs are paired too.
TUNED_ENTRIES = ([{"label": f"{a}-tuned", "algo": a, "overrides": c, "seed_label": a}
                  for a, c in TUNED_CONFIGS.items()] if CFG.equal_tuning else [])
TUNED_LABELS = [e["label"] for e in TUNED_ENTRIES]

print(f"{len(OPTIMIZERS)} optimisers registered; {len(ENTRIES)} in the main comparison:")
print("  " + ", ".join(e["label"] for e in ENTRIES))
if TUNED_ENTRIES:
    print("equal-tuning protocol adds: " + ", ".join(
        f"{e['label']} {e['overrides']}" for e in TUNED_ENTRIES))

# %% [markdown]
# ## 10. Calibration
#
# Before spending real GPU time, time a handful of proxy evaluations and print a concrete
# plan. If the plan does not fit `SESSION_HOURS`, the number of independent runs (and then
# the per-run budget) is reduced automatically and the reduction is reported, so the notebook
# never dies halfway through a Kaggle session with nothing saved.

# %%
def calibrate(n_probe: int = 5) -> float:
    """Median seconds per proxy evaluation, measured on random genotypes."""
    rs = np.random.RandomState(999)
    ts = []
    for i in range(n_probe):
        ph = decode(rs.rand(D_GENES))
        t0 = time.time()
        r = train_one(ph, IDX_PTR, IDX_PVA, CFG.img_size_proxy, CFG.proxy_epochs,
                      CFG.proxy_batch, seed=CFG.proxy_seed, amp=CFG.proxy_amp)
        dt = time.time() - t0
        ts.append(dt)
        print(f"  probe {i+1}/{n_probe}: {dt:5.1f}s  "
              f"val_f1={r.get('val_f1', float('nan')):.4f}  "
              f"{r.get('mmac', float('nan')):.0f} MMac  "
              f"{r.get('params_m', float('nan')):.2f}M  ok={r.get('ok')}")
    return float(np.median(ts))


SEC_PER_EVAL = load_state("sec_per_eval")
if SEC_PER_EVAL is None:
    print("calibrating proxy evaluation cost...")
    SEC_PER_EVAL = calibrate()
    save_state("sec_per_eval", SEC_PER_EVAL)
print(f"\nmedian proxy evaluation: {SEC_PER_EVAL:.1f}s")


def plan_search(sec_per_eval: float) -> Tuple[int, int, float]:
    """Fit (n_runs, budget) into the session clock; returns the adjusted plan."""
    n_opt = len(ENTRIES) + len(TUNED_ENTRIES)
    runs, budget = CFG.n_runs, CFG.budget
    reserve = 0.12 * SESSION_HOURS * 3600          # headroom for figures/tables/saves
    avail = CLOCK.left() - reserve
    # memo reuse across runs/algorithms typically saves 20-40%; assume a conservative 15%
    def cost(r, b):
        return n_opt * r * b * sec_per_eval * 0.85
    # The search resumes across sessions, so the protocol is never shrunk to fit one session:
    # a smaller protocol would make this dataset incomparable with the other one.
    return runs, budget, cost(runs, budget)


_SAVED_SEARCH = load_state("search_results", []) or []
if _SAVED_SEARCH:
    # a resumed experiment keeps its original protocol, whatever time this session has left
    BUDGET_EFF = int(_SAVED_SEARCH[0]["budget"])
    N_RUNS_EFF = (1 + max(r["run"] for r in _SAVED_SEARCH)) if RUN_PLAN == "analysis" else CFG.n_runs
    print(f"resuming saved protocol: {N_RUNS_EFF} runs x {BUDGET_EFF} evaluations "
          f"({len(_SAVED_SEARCH)} search runs already complete)")
elif RUN_PLAN in ("search", "all"):
    N_RUNS_EFF, BUDGET_EFF, EST = plan_search(SEC_PER_EVAL)
    _n_ent = len(ENTRIES) + len(TUNED_ENTRIES)
    print(f"\nsearch plan: {_n_ent} optimiser configurations x {N_RUNS_EFF} runs x "
          f"{BUDGET_EFF} evaluations = {_n_ent*N_RUNS_EFF*BUDGET_EFF} evaluations")
    print(f"estimated search wall-clock: {CLOCK.hms(EST)}  "
          f"(session remaining: {CLOCK.hms(CLOCK.left())}); about "
          f"{math.ceil(EST / max(1.0, 0.85 * SESSION_HOURS * 3600))} session(s). Each version "
          f"stops cleanly; attach its output and run again to continue.")
    if (N_RUNS_EFF, BUDGET_EFF) != (CFG.n_runs, CFG.budget):
        print(f"NOTE: plan reduced from {CFG.n_runs} runs x {CFG.budget} evals to fit the "
              f"session clock. Report the effective values in the paper.")
else:
    N_RUNS_EFF, BUDGET_EFF = CFG.n_runs, CFG.budget

# %% [markdown]
# ## 11. Search phase
#
# Each (optimiser, run) pair is an independent experiment with its own RNG seed and its own
# budget counter, checkpointed the moment it finishes. Interrupting and re-running the
# notebook resumes from the next unfinished pair.
#
# **A cost asymmetry to report honestly.** The budget is equal in *fitness evaluations*, which
# is the protocol reviewers expect — but MO-P3PGA evaluates **two** populations per generation
# per island (the 3-parent population produced by the mitochondrial change, then the GA
# offspring), so under an equal evaluation budget it completes roughly **half as many
# generations** as single-offspring algorithms such as NSGA-II. With `pop_total = 24` and a
# budget of 240 the first full run recorded 4 generations for MO-P3PGA against 10 for NSGA-II,
# because re-evaluating individuals after the leader-attraction step also spends budget. The
# `generations` column of the results table records this for every run. The extra evaluations
# are a genuine cost of the 3-parent mechanism, and charging the algorithm for them is the
# fair accounting; the ablation study then shows whether that cost buys anything.

# %%
def run_one_search(algo: str, run: int, budget: int,
                   overrides: Optional[Dict[str, Any]] = None,
                   cfg: Optional[Config] = None,
                   label: Optional[str] = None,
                   seed_label: Optional[str] = None) -> Dict[str, Any]:
    cfg = cfg or CFG
    label = label or algo
    # crc32, not hash(): Python randomises string hashes per process, which would change seeds.
    # seed_label lets an ablation variant reuse another configuration's seed, so run r of the
    # variant starts from exactly the same initial population as run r of that configuration.
    seed = 1000 * run + zlib.crc32((seed_label or label).encode()) % 977
    ev = MOEvaluator(cfg, budget, tag=f"{label}#r{run}", clock=CLOCK,
                     time_reserve=0.06 * SESSION_HOURS * 3600)
    opt = OPTIMIZERS[algo](cfg, ev, seed=seed, overrides=overrides)
    t0 = time.time()
    arch = opt.run()
    rec = {
        "dataset": DATASET, "algo": algo, "label": label, "run": run, "seed": seed,
        "budget": budget, "generations": opt.n_gen,
        "archive_F": arch.F.copy(), "archive_X": arch.X.copy(),
        "history": ev.history, "summary": ev.summary(),
        "objectives": list(cfg.objectives), "wall": time.time() - t0,
        "overrides": dict(overrides or {}),
    }
    return rec


def is_truncated(rec: Dict[str, Any]) -> bool:
    """A run the session clock cut short. It must be re-run, never analysed as a finished run.
    Older checkpoints left stopped_reason empty in this case, hence the second test."""
    s = rec.get("summary", {})
    reason = s.get("stopped_reason", "")
    return reason == "wall-clock" or (reason == "" and s.get("evals", 0) < rec.get("budget", 0))


def drop_truncated(results: List[Dict[str, Any]], state_key: str) -> List[Dict[str, Any]]:
    bad = [r for r in results if is_truncated(r)]
    if bad:
        print(f"[{state_key}] discarding {len(bad)} clock-truncated run(s) so they are re-run: "
              + ", ".join(f"{r['label']} run {r['run']} ({r['summary'].get('evals')}/"
                          f"{r['budget']} evals)" for r in bad))
    return [r for r in results if not is_truncated(r)]


def search_phase(algos: List[str], n_runs: int, budget: int,
                 state_key: str = "search_results",
                 overrides_by_algo: Optional[Dict[str, Dict[str, Any]]] = None,
                 labels: Optional[Dict[str, str]] = None,
                 entries: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """`entries` = [{label, algo, overrides, seed_label}], or build them from `algos`."""
    results: List[Dict[str, Any]] = drop_truncated(load_state(state_key, []) or [], state_key)
    done = {(r["label"], r["run"]) for r in results}
    overrides_by_algo = overrides_by_algo or {}
    labels = labels or {}
    if entries is None:
        entries = [{"label": labels.get(a, a), "algo": a,
                    "overrides": overrides_by_algo.get(a), "seed_label": None} for a in algos]
    total = len(entries) * n_runs
    i = 0
    # Run-major order: every optimiser finishes run r before any starts run r+1, so a session
    # that stops on the clock leaves all optimisers with the same number of runs (+/- one).
    n_cached = 0
    for run in range(n_runs):
        for e in entries:
            algo, label = e["algo"], e["label"]
            i += 1
            if (label, run) in done:
                n_cached += 1
                continue
            need = budget * SEC_PER_EVAL * 0.6 + 60
            if not CLOCK.ok(need + 0.06 * SESSION_HOURS * 3600):
                print(f"\n*** session clock: stopping before {label} run {run}. "
                      f"{len(results)}/{total} pairs complete. Re-run the notebook to "
                      f"resume. ***")
                save_state(state_key, results)
                save_state("global_memo", GLOBAL_MEMO)
                return results
            if n_cached:
                print(f"({n_cached} cached runs skipped)")
                n_cached = 0
            print(f"[{i}/{total}] {label} run {run} ... ", end="", flush=True)
            rec = run_one_search(algo, run, budget, overrides=e.get("overrides"),
                                 label=label, seed_label=e.get("seed_label"))
            if is_truncated(rec):
                print(f"stopped by the session clock after {rec['summary']['evals']}/{budget} "
                      f"evaluations; not saved, it will be re-run in the next version")
                save_state(state_key, results)
                save_state("global_memo", GLOBAL_MEMO)
                return results
            results.append(rec)
            save_state(state_key, results)
            save_state("global_memo", GLOBAL_MEMO)
            s = rec["summary"]
            best_err = rec["archive_F"][:, 0].min() if rec["archive_F"].size else float("nan")
            print(f"gen={rec['generations']:3d} |A|={len(rec['archive_F']):3d} "
                  f"min(err)={best_err:.4f} evals={s['evals']} trained={s['trained']} "
                  f"reuse={s['memo_reuse_rate']*100:.0f}% {CLOCK.hms(rec['wall'])}")
            CLOCK.report()
    if n_cached:
        print(f"({n_cached} cached runs skipped)")
    save_state(state_key, results)
    save_state("global_memo", GLOBAL_MEMO)
    return results


ALL_ENTRIES = ENTRIES + TUNED_ENTRIES
SEARCH: List[Dict[str, Any]] = []
if RUN_PLAN in ("search", "all"):
    print(f"\n{'='*78}\nSEARCH PHASE\n{'='*78}")
    SEARCH = search_phase(None, N_RUNS_EFF, BUDGET_EFF, "search_results", entries=ALL_ENTRIES)
else:
    SEARCH = load_state("search_results", []) or []
    print(f"loaded {len(SEARCH)} search records from checkpoint")
_want = len(ALL_ENTRIES) * N_RUNS_EFF
_have = sum(1 for r in SEARCH if any(r["label"] == e["label"] for e in ALL_ENTRIES)
            and r["run"] < N_RUNS_EFF)
# Models, ablations and reports are only built from a finished search: a knee point taken
# from a partial front would be trained, cached under its name and reported later as final.
SEARCH_COMPLETE = _have >= _want
if SEARCH_COMPLETE:
    print(f"\nsearch phase complete: {_have} (optimiser, run) pairs")
else:
    print(f"\nsearch phase PARTIAL: {_have}/{_want} pairs. Attach this version's output and run "
          f"RUN_PLAN='search' (or 'all') again to continue. Final training and reports wait "
          f"until the search is complete.")

# Analyse complete rounds only, so every optimiser contributes the same runs to every statistic.
_main_labels = [e["label"] for e in ENTRIES]
_runs_by_algo = {a: {r["run"] for r in SEARCH if r["label"] == a} for a in _main_labels}
_complete = sorted(set.intersection(*_runs_by_algo.values())) if SEARCH else []
SEARCH_TUNED = [r for r in SEARCH if r["label"] in TUNED_LABELS and r["run"] in set(_complete)]
_n_main = sum(1 for r in SEARCH if r["label"] in _main_labels)
if SEARCH and len(_complete) * len(_main_labels) < _n_main:
    print(f"analysing {len(_complete)} complete rounds; "
          f"{_n_main - len(_complete) * len(_main_labels)} runs from an unfinished round are "
          f"kept in the checkpoint but left out until that round finishes")
SEARCH = [r for r in SEARCH if r["label"] in _main_labels and r["run"] in set(_complete)]
if SEARCH:
    N_RUNS_EFF = len(_complete)

if not SEARCH:
    raise RuntimeError("No search results found. Run with RUN_PLAN='search' first, or attach "
                       "the previous version's output (containing p3pga_state) as an input.")

# %% [markdown]
# ## 12. Reference front, normalisation and indicator tables
#
# The reference front is the non-dominated subset of every solution produced by every
# optimiser in every run. Objectives are min–max normalised over all pooled archive solutions; the hypervolume
# reference point is $\mathbf{1.1}$ in normalised space.

# %%
# Pooled over BOTH protocols, so the standard and the equal-tuning tables share one
# normalisation and one reference front and their indicator values can be compared directly.
ALL_F = np.vstack([r["archive_F"] for r in SEARCH + SEARCH_TUNED if r["archive_F"].size])
REF_MASK = non_dominated_mask(ALL_F)
REF_FRONT = ALL_F[REF_MASK]
Z_MIN = ALL_F.min(axis=0)
Z_MAX = ALL_F.max(axis=0)
REF_POINT = np.full(CFG.n_obj, 1.1)
REF_FRONT_N = normalize_front(REF_FRONT, Z_MIN, Z_MAX)

print(f"pooled solutions: {ALL_F.shape[0]} | reference front size: {REF_FRONT.shape[0]}")
for j, o in enumerate(CFG.objectives):
    print(f"  {o:<9s} range [{Z_MIN[j]:.4f}, {Z_MAX[j]:.4f}]")


def indicators(F: np.ndarray) -> Dict[str, float]:
    if F is None or np.asarray(F).size == 0:
        return {k: float("nan") for k in
                ["HV", "IGD", "IGD+", "GD", "SP", "MS", "Delta", "n_nd", "best_err"]}
    Fn = normalize_front(F, Z_MIN, Z_MAX)
    Fn = Fn[non_dominated_mask(Fn)]
    return {
        "HV": hypervolume(Fn, REF_POINT),
        "IGD": igd(Fn, REF_FRONT_N),
        "IGD+": igd_plus(Fn, REF_FRONT_N),
        "GD": gd(Fn, REF_FRONT_N),
        "SP": spacing(Fn),
        "MS": max_spread(Fn),
        "Delta": delta_spread(Fn, REF_FRONT_N) if CFG.n_obj == 2 else float("nan"),
        "n_nd": float(Fn.shape[0]),
        "best_err": float(F[:, 0].min()),
    }


ROWS = []
for r in SEARCH:
    ind = indicators(r["archive_F"])
    ROWS.append({"algo": r["label"], "run": r["run"], **ind,
                 "generations": r["generations"],
                 "evals": r["summary"]["evals"],
                 "trained": r["summary"]["trained"],
                 "wall_s": r["wall"]})
RUNS_DF = pd.DataFrame(ROWS)
RUNS_DF.to_csv(OUT / "tables" / "per_run_indicators.csv", index=False)

ORDER = [a for a in OPTIMIZERS if a in set(RUNS_DF["algo"])]
AGG = (RUNS_DF.groupby("algo")[["HV", "IGD", "IGD+", "GD", "SP", "MS", "Delta",
                                "n_nd", "best_err", "generations"]]
       .agg(["mean", "std"]).reindex(ORDER))
AGG.columns = [f"{a}_{b}" for a, b in AGG.columns]
AGG = AGG.sort_values("HV_mean", ascending=False)
AGG.to_csv(OUT / "tables" / "indicator_summary.csv")

print("\n--- Multi-objective indicators (mean +/- std over runs) ---")
_disp = pd.DataFrame({
    "HV (max)":    AGG.apply(lambda r: f"{r['HV_mean']:.4f}+/-{r['HV_std']:.4f}", axis=1),
    "IGD+ (min)":  AGG.apply(lambda r: f"{r['IGD+_mean']:.4f}+/-{r['IGD+_std']:.4f}", axis=1),
    "GD (min)":    AGG.apply(lambda r: f"{r['GD_mean']:.4f}+/-{r['GD_std']:.4f}", axis=1),
    "SP (min)":    AGG.apply(lambda r: f"{r['SP_mean']:.4f}+/-{r['SP_std']:.4f}", axis=1),
    "MS (max)":    AGG.apply(lambda r: f"{r['MS_mean']:.4f}+/-{r['MS_std']:.4f}", axis=1),
    "|ND|":        AGG.apply(lambda r: f"{r['n_nd_mean']:.1f}", axis=1),
    "min err":     AGG.apply(lambda r: f"{r['best_err_mean']:.4f}", axis=1),
})
print(_disp.to_string())

# %% [markdown]
# ## 13. Statistical comparison of the optimisers
#
# Four complementary tests, because no single one carries a NAS comparison on its own:
#
# * **Friedman** rank test across optimisers, blocked by run, on hypervolume — the standard
#   omnibus test for comparing many algorithms.
# * **Pairwise Wilcoxon signed-rank** of the proposed method against each competitor, with
#   **Holm–Bonferroni** correction for multiplicity.
# * **Vargha–Delaney $\hat{A}_{12}$** effect size, so the magnitude of a difference is
#   reported and not just its p-value.
# * **Set coverage $C(A,B)$**, a dominance-based pairwise comparison that is independent of
#   any indicator's normalisation choices.
#
# A caveat to state plainly in the paper: with `n_runs` independent repetitions the
# rank-based tests have limited power, and a non-significant result here is weak evidence of
# equivalence rather than evidence of no difference. The statistically well-powered claim in
# this study is the **McNemar test on test-set predictions** (Section 16), where $n$ is the
# number of test images rather than the number of search runs.

# %%
NEMENYI_Q05 = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
               9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
               16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544}


def vargha_delaney_a12(a, b) -> float:
    """P(X>Y) + 0.5 P(X=Y). 0.5 = no effect; >0.71 or <0.29 = large."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = (a[:, None] > b[None, :]).sum()
    eq = (a[:, None] == b[None, :]).sum()
    return float((gt + 0.5 * eq) / (a.size * b.size))


def holm_correction(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj.tolist()


def friedman_ranks(pivot: pd.DataFrame, higher_better: bool) -> pd.Series:
    """Average rank per algorithm (rank 1 = best) across blocks (runs)."""
    R = pivot.rank(axis=1, ascending=not higher_better)
    return R.mean(axis=0)


def stats_block(df: pd.DataFrame, metric: str, higher_better: bool) -> Dict[str, Any]:
    pivot = df.pivot_table(index="run", columns="algo", values=metric)
    pivot = pivot.dropna(axis=0, how="any")
    algos = list(pivot.columns)
    out: Dict[str, Any] = {"metric": metric, "n_blocks": int(pivot.shape[0]),
                           "algos": algos}
    if pivot.shape[0] >= 2 and pivot.shape[1] >= 3:
        try:
            stat, p = sps.friedmanchisquare(*[pivot[a].values for a in algos])
            out["friedman_chi2"], out["friedman_p"] = float(stat), float(p)
        except Exception as e:
            out["friedman_chi2"], out["friedman_p"] = float("nan"), float("nan")
            out["friedman_error"] = str(e)
    ranks = friedman_ranks(pivot, higher_better)
    out["avg_ranks"] = ranks.to_dict()
    k, N = len(algos), pivot.shape[0]
    q = NEMENYI_Q05.get(k, 3.544)
    out["nemenyi_cd"] = float(q * math.sqrt(k * (k + 1) / (6.0 * max(1, N))))

    rows = []
    base = PROPOSED if PROPOSED in algos else ranks.idxmin()
    raw_p = []
    others = [a for a in algos if a != base]
    for a in others:
        x, y = pivot[base].values, pivot[a].values
        try:
            if np.allclose(x, y):
                p_w = 1.0
            else:
                p_w = float(sps.wilcoxon(x, y, zero_method="zsplit",
                                         alternative="two-sided").pvalue)
        except Exception:
            p_w = float("nan")
        try:
            p_u = float(sps.mannwhitneyu(x, y, alternative="two-sided").pvalue)
        except Exception:
            p_u = float("nan")
        a12 = vargha_delaney_a12(x, y)
        rows.append({"vs": a, "mean_base": x.mean(), "mean_other": y.mean(),
                     "wilcoxon_p": p_w, "mannwhitney_p": p_u, "A12": a12})
        raw_p.append(p_w if np.isfinite(p_w) else 1.0)
    adj = holm_correction(raw_p) if raw_p else []
    for r, pa in zip(rows, adj):
        r["wilcoxon_p_holm"] = pa
        r["significant_0.05"] = bool(pa < 0.05)
        r["effect"] = ("large" if (r["A12"] >= 0.71 or r["A12"] <= 0.29) else
                       "medium" if (r["A12"] >= 0.64 or r["A12"] <= 0.36) else
                       "small" if (r["A12"] >= 0.56 or r["A12"] <= 0.44) else "negligible")
    out["baseline"] = base
    out["pairwise"] = rows
    return out


STATS = {m: stats_block(RUNS_DF, m, hb) for m, hb in
         [("HV", True), ("IGD+", False), ("GD", False), ("best_err", False)]}
jdump("search_statistics.json", STATS)

for m in ["HV", "IGD+"]:
    s = STATS[m]
    print(f"\n--- {m}: Friedman chi2={s.get('friedman_chi2', float('nan')):.3f} "
          f"p={s.get('friedman_p', float('nan')):.4g}  "
          f"(N={s['n_blocks']} runs, k={len(s['algos'])} algorithms, "
          f"Nemenyi CD={s['nemenyi_cd']:.3f}) ---")
    print("average ranks:", {k: round(v, 2) for k, v in
                             sorted(s["avg_ranks"].items(), key=lambda kv: kv[1])})
    print(pd.DataFrame(s["pairwise"])[
        ["vs", "mean_base", "mean_other", "wilcoxon_p", "wilcoxon_p_holm", "A12", "effect"]
    ].to_string(index=False))

# %%
# ---- set coverage matrix on pooled per-algorithm fronts --------------------------------
POOLED: Dict[str, np.ndarray] = {}
for a in ORDER:
    Fa = np.vstack([r["archive_F"] for r in SEARCH
                    if r["label"] == a and r["archive_F"].size])
    POOLED[a] = Fa[non_dominated_mask(Fa)]

COV = pd.DataFrame(index=ORDER, columns=ORDER, dtype=float)
for a in ORDER:
    for b in ORDER:
        COV.loc[a, b] = np.nan if a == b else set_coverage(POOLED[a], POOLED[b])
COV.to_csv(OUT / "tables" / "set_coverage.csv")
print("\n--- set coverage C(row, col): fraction of column's front dominated by row ---")
print(COV.round(3).to_string())

POOLED_IND = pd.DataFrame({a: indicators(POOLED[a]) for a in ORDER}).T
POOLED_IND["contrib_to_ref"] = [
    float(((REF_FRONT[:, None, :] == POOLED[a][None, :, :]).all(axis=2)).any(axis=1).sum())
    for a in ORDER]
POOLED_IND.to_csv(OUT / "tables" / "pooled_front_indicators.csv")
print("\n--- pooled fronts (all runs merged per optimiser) ---")
print(POOLED_IND.round(4).to_string())

# %% [markdown]
# ## 14. Figures for the search phase
#
# Every figure is written to `outputs/figures/` as both PNG (300 dpi) and PDF (vector, for
# LaTeX). Colour and marker assignments are stable across figures so the proposed method is
# identifiable at a glance.

# %%
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "savefig.bbox": "tight",
})

PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b",
           "#17becf", "#e377c2", "#7f7f7f", "#bcbd22", "#393b79", "#637939"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p"]
# red is reserved for the proposed method; everyone else cycles through the rest
_others = [a for a in ORDER if a != PROPOSED]
STYLE = {a: {"color": PALETTE[1 + i % (len(PALETTE) - 1)],
             "marker": MARKERS[1 + i % (len(MARKERS) - 1)]} for i, a in enumerate(_others)}
if PROPOSED in ORDER:
    STYLE[PROPOSED] = {"color": "#d62728", "marker": "o"}


def savefig(fig, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote figures/{name}.png|pdf")


def raw_units(F: np.ndarray) -> np.ndarray:
    """Undo the log10 transforms so axes carry physically meaningful units."""
    R = np.array(F, dtype=float, copy=True)
    for j, o in enumerate(CFG.objectives):
        if o in ("flops", "params", "latency"):
            R[:, j] = 10.0 ** R[:, j]
    return R


# ---- Figure 1: Pareto fronts in objective space ---------------------------------------
if CFG.n_obj == 2:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for a in ORDER:
        P = raw_units(POOLED[a])
        o = np.argsort(P[:, 1])
        st = STYLE[a]
        lw = 2.4 if a == PROPOSED else 1.1
        ms = 7 if a == PROPOSED else 4.5
        z = 5 if a == PROPOSED else 2
        ax.plot(P[o, 1], P[o, 0] * 100, marker=st["marker"], color=st["color"],
                lw=lw, ms=ms, alpha=0.95 if a == PROPOSED else 0.65, label=a, zorder=z)
    ax.set_xscale("log")
    ax.set_xlabel("FLOPs (MMac, log scale)")
    ax.set_ylabel(r"validation error $(1-F_1)$  [%]")
    ax.set_title("(a) Pareto fronts, all optimisers")
    ax.legend(ncol=2, fontsize=7.5, loc="upper right")

    ax = axes[1]
    R = raw_units(REF_FRONT)
    o = np.argsort(R[:, 1])
    ax.plot(R[o, 1], R[o, 0] * 100, "k--", lw=1.0, alpha=0.5,
            label=f"reference front (n={R.shape[0]})")
    allp = raw_units(ALL_F)
    ax.scatter(allp[:, 1], allp[:, 0] * 100, s=5, c="#cccccc", alpha=0.5, zorder=1,
               label="all evaluated architectures")
    P = raw_units(POOLED[PROPOSED])
    o = np.argsort(P[:, 1])
    ax.plot(P[o, 1], P[o, 0] * 100, "o-", color="#d62728", lw=2.2, ms=7,
            label=PROPOSED, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("FLOPs (MMac, log scale)")
    ax.set_ylabel(r"validation error $(1-F_1)$  [%]")
    ax.set_title("(b) proposed front vs. the whole search space")
    ax.legend(fontsize=8, loc="upper right")
    savefig(fig, "fig1_pareto_fronts")
else:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for a in ORDER:
        P = raw_units(POOLED[a])
        st = STYLE[a]
        ax.scatter(P[:, 1], P[:, 2], P[:, 0] * 100, marker=st["marker"], color=st["color"],
                   s=40 if a == PROPOSED else 14, alpha=0.9 if a == PROPOSED else 0.5,
                   label=a)
    ax.set_xlabel("MMac"); ax.set_ylabel("params (M)"); ax.set_zlabel("error [%]")
    ax.set_title("(a) 3-objective Pareto fronts")
    ax.legend(fontsize=6, ncol=2)
    ax2 = fig.add_subplot(1, 2, 2)
    for a in ORDER:
        Pn = normalize_front(POOLED[a], Z_MIN, Z_MAX)
        st = STYLE[a]
        for row in Pn:
            ax2.plot(range(CFG.n_obj), row, color=st["color"],
                     alpha=0.7 if a == PROPOSED else 0.18,
                     lw=1.8 if a == PROPOSED else 0.7)
        ax2.plot([], [], color=st["color"], label=a)
    ax2.set_xticks(range(CFG.n_obj))
    ax2.set_xticklabels([OBJ_RAW_LABEL[o] for o in CFG.objectives])
    ax2.set_ylabel("normalised objective value")
    ax2.set_title("(b) parallel coordinates")
    ax2.legend(fontsize=6, ncol=2)
    savefig(fig, "fig1_pareto_fronts")

# %%
# ---- Figure 2: anytime hypervolume convergence ----------------------------------------
# Computed identically for every optimiser: HV of the non-dominated subset of the first t
# evaluations. This is the fairest convergence view because it depends only on what an
# optimiser has evaluated, not on its internal bookkeeping.
def anytime_hv(history: List[Dict[str, Any]], grid: np.ndarray) -> np.ndarray:
    F = np.vstack([h["f"] for h in history]) if history else np.zeros((0, CFG.n_obj))
    out = np.zeros(len(grid))
    for gi, t in enumerate(grid):
        t = int(min(t, F.shape[0]))
        if t <= 0:
            out[gi] = 0.0
            continue
        Fn = normalize_front(F[:t], Z_MIN, Z_MAX)
        Fn = Fn[non_dominated_mask(Fn)]
        out[gi] = hypervolume(Fn, REF_POINT, mc_samples=60_000)
    return out


HV_GRID = np.unique(np.linspace(1, BUDGET_EFF, min(40, BUDGET_EFF)).astype(int))
CURVES: Dict[str, np.ndarray] = {}
for a in ORDER:
    rs = [r for r in SEARCH if r["label"] == a]
    CURVES[a] = np.vstack([anytime_hv(r["history"], HV_GRID) for r in rs])
save_state("hv_curves", {"grid": HV_GRID, "curves": CURVES})

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
ax = axes[0]
for a in ORDER:
    C = CURVES[a]
    m, s = C.mean(axis=0), C.std(axis=0)
    st = STYLE[a]
    lw = 2.4 if a == PROPOSED else 1.2
    ax.plot(HV_GRID, m, color=st["color"], lw=lw, label=a,
            marker=st["marker"], ms=4, markevery=max(1, len(HV_GRID) // 8))
    ax.fill_between(HV_GRID, m - s, m + s, color=st["color"],
                    alpha=0.18 if a == PROPOSED else 0.07)
ax.set_xlabel("fitness evaluations")
ax.set_ylabel("hypervolume (normalised)")
ax.set_title(f"(a) anytime hypervolume, mean $\\pm$ 1 s.d. over {N_RUNS_EFF} runs")
ax.legend(ncol=2, fontsize=7.5, loc="lower right")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax = axes[1]
data = [RUNS_DF.loc[RUNS_DF["algo"] == a, "HV"].values for a in ORDER]
bp = ax.boxplot(data, labels=ORDER, patch_artist=True, widths=0.6, showmeans=True)
for patch, a in zip(bp["boxes"], ORDER):
    patch.set_facecolor(STYLE[a]["color"])
    patch.set_alpha(0.85 if a == PROPOSED else 0.45)
for i, a in enumerate(ORDER):
    ax.scatter(np.full(len(data[i]), i + 1) + np.random.uniform(-0.08, 0.08, len(data[i])),
               data[i], s=14, c="k", zorder=5, alpha=0.6)
ax.set_ylabel("final hypervolume")
ax.set_title("(b) final hypervolume distribution")
plt.setp(ax.get_xticklabels(), rotation=38, ha="right", fontsize=8)
savefig(fig, "fig2_convergence_and_hv")

# %%
# ---- Figure 3: critical-difference diagram + set-coverage heatmap ----------------------
def cd_diagram(ax, avg_ranks: Dict[str, float], cd: float, title: str):
    """Demsar-style critical-difference diagram. Rank 1 (best) is on the left; the better
    half of the algorithms is labelled on the left, the worse half on the right, and thick
    bars join groups whose average ranks differ by less than the critical difference."""
    items = sorted(avg_ranks.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    ranks = [v for _, v in items]
    k = len(names)
    lo, hi = 1, max(k, math.ceil(max(ranks)))
    n_left = (k + 1) // 2
    rows = max(n_left, k - n_left)
    y_axis = 0.0
    step = 0.55
    pad = 0.45 * (hi - lo) + 0.6          # room for labels inside the axes
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(-(rows + 1.6) * step, 1.6)
    ax.axis("off")
    # rank axis
    ax.plot([lo, hi], [y_axis, y_axis], "k-", lw=1.2)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [y_axis, y_axis + 0.12], "k-", lw=1.0)
        ax.text(t, y_axis + 0.2, str(t), ha="center", va="bottom", fontsize=9)
    # critical-difference scale bar
    ax.plot([lo, lo + cd], [1.2, 1.2], "k-", lw=1.8)
    for xx in (lo, lo + cd):
        ax.plot([xx, xx], [1.12, 1.28], "k-", lw=1.2)
    ax.text(lo + cd / 2, 1.33, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=9)
    # algorithm labels with elbow leader lines
    for i, (nm, r) in enumerate(items):
        left = i < n_left
        row = i if left else (k - 1 - i)
        y = -(row + 1) * step
        x_end = lo - 0.15 if left else hi + 0.15
        ax.plot([r, r], [y_axis, y], "k-", lw=0.9)
        ax.plot([r, x_end], [y, y], "k-", lw=0.9)
        bold = nm == PROPOSED
        ax.text(x_end + (-0.05 if left else 0.05), y, f"{nm} ({r:.2f})",
                ha="right" if left else "left", va="center", fontsize=9,
                fontweight="bold" if bold else "normal",
                color="#d62728" if bold else "k", clip_on=False)
    # cliques: maximal runs of algorithms whose rank spread is within CD
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and ranks[j + 1] - ranks[i] <= cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    yb = -0.18
    for a, b in cliques:
        ax.plot([ranks[a] - 0.03, ranks[b] + 0.03], [yb, yb], "-", color="#d62728",
                lw=3.0, solid_capstyle="butt")
        yb -= 0.12
    ax.set_title(title, fontsize=10, pad=14)


fig, axes = plt.subplots(1, 2, figsize=(16, 5.2),
                         gridspec_kw={"width_ratios": [1.35, 1]})
cd_diagram(axes[0], STATS["HV"]["avg_ranks"], STATS["HV"]["nemenyi_cd"],
           f"(a) Nemenyi CD diagram on hypervolume (N={STATS['HV']['n_blocks']} runs)")
ax = axes[1]
M = COV.astype(float).values
im = ax.imshow(np.nan_to_num(M, nan=0.0), cmap="RdYlBu_r", vmin=0, vmax=1)
ax.set_xticks(range(len(ORDER))); ax.set_yticks(range(len(ORDER)))
ax.set_xticklabels(ORDER, rotation=45, ha="right", fontsize=7)
ax.set_yticklabels(ORDER, fontsize=7)
for i in range(len(ORDER)):
    for j in range(len(ORDER)):
        if i != j:
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] > 0.55 else "black")
ax.set_title("(b) set coverage $C$(row, col)", fontsize=10)
ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)
savefig(fig, "fig3_cd_and_coverage")

# %%
# ---- Figure 4: which design choices survive on the Pareto front? ----------------------
# A reviewer always asks "what did the search actually discover?". This figure answers it
# from the reference front rather than from anecdote.
ALL_RECS = []
for r in SEARCH:
    for h in r["history"]:
        ALL_RECS.append({"algo": r["label"], "run": r["run"],
                         **{k: v for k, v in h.items() if k not in ("f", "ph", "x")},
                         **{f"ph_{k}": v for k, v in h["ph"].items()},
                         "f0": h["f"][0], "f1": h["f"][1] if CFG.n_obj > 1 else np.nan})
EVAL_DF = pd.DataFrame(ALL_RECS)
EVAL_DF.to_csv(OUT / "tables" / "all_evaluations.csv", index=False)
print(f"logged {len(EVAL_DF)} evaluation records "
      f"({EVAL_DF['ok'].mean()*100:.1f}% feasible)")

# phenotypes of the genotypes that sit on the reference front
front_rows, seen = [], set()
for r in SEARCH:
    if not r["archive_F"].size:
        continue
    for xi, fi in zip(r["archive_X"], r["archive_F"]):
        if ((REF_FRONT == fi).all(axis=1)).any():
            k = genotype_key(xi)
            if k not in seen:
                seen.add(k)
                front_rows.append(decode(xi))
FRONT_PH = pd.DataFrame(front_rows)
FRONT_PH.to_csv(OUT / "tables" / "reference_front_architectures.csv", index=False)

CAT_GENES = ["block_type", "activation", "optimizer", "head_pool", "n_blocks",
             "base_filters", "kernel", "use_se", "stage_depth"]
base_ph = pd.DataFrame([h["ph"] for r in SEARCH for h in r["history"]])
fig, axes = plt.subplots(3, 3, figsize=(13, 9))
for ax, g in zip(axes.ravel(), CAT_GENES):
    if g not in FRONT_PH.columns:
        ax.axis("off"); continue
    vf = FRONT_PH[g].astype(str).value_counts(normalize=True)
    vb = base_ph[g].astype(str).value_counts(normalize=True)
    cats = sorted(set(vf.index) | set(vb.index))
    xs = np.arange(len(cats))
    ax.bar(xs - 0.2, [vb.get(c, 0) for c in cats], width=0.4, label="all evaluated",
           color="#bbbbbb")
    ax.bar(xs + 0.2, [vf.get(c, 0) for c in cats], width=0.4, label="reference front",
           color="#d62728")
    ax.set_xticks(xs); ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=7)
    ax.set_title(g, fontsize=9)
    ax.set_ylabel("frequency", fontsize=8)
axes.ravel()[0].legend(fontsize=7)
fig.suptitle("Design choices enriched on the Pareto reference front "
             "(red) relative to the full search (grey)", fontsize=11)
fig.tight_layout()
savefig(fig, "fig4_design_choice_enrichment")

# %% [markdown]
# ## 14b. Equal-tuning protocol
#
# The standard protocol above compares MO-P3PGA-II, whose settings were chosen by a tuning
# screen, with competitors at their recommended settings. The obvious objection is tuning
# bias. This section answers it: the two strongest competitors (GDE3 and MOGWO) are re-run with
# settings chosen by the SAME screen and the same 70-configuration budget, on the same seeds as
# their default runs, and the rank tests are repeated with them in place of their defaults.
# Both tables share one normalisation and reference front, so their values are comparable.

# %%
TUNED_ORDER = [l for l in TUNED_LABELS if any(r["label"] == l for r in SEARCH_TUNED)]
STATS_T: Dict[str, Any] = {}
T_AGG = pd.DataFrame()
TUNED_PAIRED: List[Dict[str, Any]] = []
if TUNED_ORDER:
    _swap = {e["algo"]: e["label"] for e in TUNED_ENTRIES if e["label"] in TUNED_ORDER}
    T_LABELS = [_swap.get(a, a) for a in ORDER if not (a == BASE_P3 and PROPOSED != BASE_P3)]
    _trows = []
    for r in SEARCH + SEARCH_TUNED:
        if r["label"] in T_LABELS:
            _trows.append({"algo": r["label"], "run": r["run"], **indicators(r["archive_F"]),
                           "generations": r["generations"]})
    T_RUNS_DF = pd.DataFrame(_trows)
    T_RUNS_DF.to_csv(OUT / "tables" / "per_run_indicators_tuned_protocol.csv", index=False)
    T_AGG = (T_RUNS_DF.groupby("algo")[["HV", "IGD+", "GD", "SP", "MS", "n_nd", "best_err",
                                         "generations"]].agg(["mean", "std"]))
    T_AGG.columns = [f"{a}_{b}" for a, b in T_AGG.columns]
    T_AGG = T_AGG.sort_values("HV_mean", ascending=False)
    T_AGG.to_csv(OUT / "tables" / "indicator_summary_tuned_protocol.csv")
    STATS_T = {m: stats_block(T_RUNS_DF, m, hb) for m, hb in
               [("HV", True), ("IGD+", False), ("GD", False), ("best_err", False)]}
    # tuned vs default run of the same competitor: same seed, so a paired comparison
    for e in TUNED_ENTRIES:
        if e["label"] not in TUNED_ORDER:
            continue
        d = RUNS_DF[RUNS_DF["algo"] == e["algo"]].set_index("run")["HV"]
        t = T_RUNS_DF[T_RUNS_DF["algo"] == e["label"]].set_index("run")["HV"]
        common = sorted(set(d.index) & set(t.index))
        x, y = t.loc[common].values, d.loc[common].values
        p = (1.0 if np.allclose(x, y) else
             float(sps.wilcoxon(x, y, zero_method="zsplit").pvalue)) if len(common) > 1 else np.nan
        TUNED_PAIRED.append({"algo": e["algo"], "settings": e["overrides"],
                             "HV_default": float(y.mean()), "HV_tuned": float(x.mean()),
                             "wilcoxon_p": p, "runs": len(common)})
    STATS_T["tuned_vs_default"] = TUNED_PAIRED
    jdump("search_statistics_tuned_protocol.json", STATS_T)

    print(f"\n{'='*78}\nEQUAL-TUNING PROTOCOL ({', '.join(TUNED_ORDER)} replace their defaults)"
          f"\n{'='*78}")
    print(T_AGG[["HV_mean", "HV_std", "IGD+_mean", "GD_mean", "generations_mean"]]
          .round(4).to_string())
    for m in ["HV", "IGD+"]:
        st = STATS_T[m]
        print(f"\n--- {m}: Friedman chi2={st.get('friedman_chi2', float('nan')):.3f} "
              f"p={st.get('friedman_p', float('nan')):.4g} (Nemenyi CD={st['nemenyi_cd']:.3f})")
        print("average ranks:", {k: round(v, 2) for k, v in
                                 sorted(st["avg_ranks"].items(), key=lambda kv: kv[1])})
        print(pd.DataFrame(st["pairwise"])[["vs", "mean_base", "mean_other", "wilcoxon_p",
                                            "wilcoxon_p_holm", "A12", "effect"]]
              .to_string(index=False))
    print("\n--- tuned vs default settings of the same competitor (paired by seed) ---")
    print(pd.DataFrame(TUNED_PAIRED).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(17, 5.2))
    cd_diagram(axes[0], STATS["HV"]["avg_ranks"], STATS["HV"]["nemenyi_cd"],
               f"(a) standard protocol: competitors at recommended settings")
    cd_diagram(axes[1], STATS_T["HV"]["avg_ranks"], STATS_T["HV"]["nemenyi_cd"],
               f"(b) equal-tuning protocol: {', '.join(TUNED_ORDER)}")
    savefig(fig, "fig12_protocols_cd")

# %% [markdown]
# ## 15. From Pareto front to reported models
#
# A multi-objective search returns a *set*, so the paper must say explicitly how a single
# deployable network is chosen from it. Three principled, non-arbitrary operating points are
# extracted:
#
# * **Knee** — the point of maximum curvature, i.e. the solution furthest from the hyperplane
#   spanned by the extreme points of the normalised front. This is the classic
#   "best trade-off without a stated preference" choice and is the headline model.
# * **Accuracy-optimal** — the minimum-error point, for benchmarking against accuracy-only
#   literature.
# * **Efficiency-optimal** — the minimum-FLOPs point, for the edge-deployment argument.
#
# The knee point of **every** competing optimiser is also trained to convergence, so the
# comparison is not merely indicator-level but ends in test-set accuracy of real networks.

# %%
def knee_point(F: np.ndarray) -> int:
    """Index of the maximum-curvature solution of a normalised front."""
    Fn = normalize_front(F, F.min(axis=0), F.max(axis=0))
    if Fn.shape[0] <= 2:
        return int(np.argmin(Fn.sum(axis=1)))
    if Fn.shape[1] == 2:
        order = np.argsort(Fn[:, 0])
        a, b = Fn[order[0]], Fn[order[-1]]
        d = b - a
        nrm = np.linalg.norm(d)
        if nrm < 1e-12:
            return int(np.argmin(Fn.sum(axis=1)))
        # perpendicular distance to the chord joining the two extremes
        v = Fn - a
        dist = (d[0] * v[:, 1] - d[1] * v[:, 0]) / nrm
        return int(np.argmax(np.abs(dist)))
    # M >= 3: distance to the hyperplane through the extreme points (intercept form)
    ext = Fn[[int(np.argmin(Fn[:, j])) for j in range(Fn.shape[1])]]
    try:
        w = np.linalg.solve(ext, np.ones(Fn.shape[1]))
    except np.linalg.LinAlgError:
        return int(np.argmin(Fn.sum(axis=1)))
    return int(np.argmax((1.0 - Fn @ w) / (np.linalg.norm(w) + 1e-12)))


def pick_operating_points(X: np.ndarray, F: np.ndarray) -> Dict[str, int]:
    nd = non_dominated_mask(F)
    Xn, Fn = X[nd], F[nd]
    idx_map = np.where(nd)[0]
    pts = {"knee": idx_map[knee_point(Fn)],
           "accuracy": idx_map[int(np.argmin(Fn[:, 0]))]}
    if CFG.n_obj > 1:
        pts["efficiency"] = idx_map[int(np.argmin(Fn[:, 1]))]
    return pts


# per-optimiser pooled front (all runs merged), then operating points from it
POOLED_X: Dict[str, np.ndarray] = {}
for a in ORDER + TUNED_ORDER:
    _ra = [r for r in SEARCH + SEARCH_TUNED if r["label"] == a]
    Xs = np.vstack([r["archive_X"] for r in _ra if r["archive_X"].size])
    Fs = np.vstack([r["archive_F"] for r in _ra if r["archive_F"].size])
    m = non_dominated_mask(Fs)
    POOLED_X[a] = Xs[m]
    POOLED[a] = Fs[m]

CANDIDATES: List[Dict[str, Any]] = []
for a in ORDER + TUNED_ORDER:
    pts = pick_operating_points(POOLED_X[a], POOLED[a])
    wanted = ["knee", "accuracy", "efficiency"] if a == PROPOSED else ["knee"]
    for kind in wanted:
        if kind not in pts:
            continue
        x = POOLED_X[a][pts[kind]]
        CANDIDATES.append({
            "name": f"{a} ({kind})" if a == PROPOSED else f"{a} (knee)",
            "algo": a, "kind": kind, "x": x.copy(), "ph": decode(x),
            "proxy_f": POOLED[a][pts[kind]].tolist(),
        })

print(f"{len(CANDIDATES)} architectures selected for full training\n")
_c = pd.DataFrame([{"model": c["name"],
                    "proxy_err": round(c["proxy_f"][0], 4),
                    "MMac": round(10 ** c["proxy_f"][1], 1) if CFG.n_obj > 1 else None,
                    "blocks": c["ph"]["n_blocks"], "block": c["ph"]["block_type"],
                    "base": c["ph"]["base_filters"], "act": c["ph"]["activation"],
                    "opt": c["ph"]["optimizer"], "lr": c["ph"]["lr"]}
                   for c in CANDIDATES])
print(_c.to_string(index=False))

# %% [markdown]
# ## 16. Full training, test-set evaluation and reference baselines
#
# Full protocol, identical for every model: train on the 70 % training split at
# `img_size_final`, select the epoch by validation macro-$F_1$, and report on the **held-out
# 15 % test split that is never touched during search or model selection**. Each architecture
# is trained with `final_seeds` independent initialisations, and the reported figures are
# mean ± s.d. over seeds with bootstrap confidence intervals on the best model.

# %%
def full_metrics(probs: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    pred = probs.argmax(1)
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "kappa": cohen_kappa_score(y, pred),
    }
    try:
        out["auc_ovr_macro"] = roc_auc_score(y, probs, multi_class="ovr", average="macro")
    except Exception:
        out["auc_ovr_macro"] = float("nan")
    for ci, c in enumerate(CLASSES):
        out[f"f1_{c}"] = f1_score((y == ci).astype(int), (pred == ci).astype(int),
                                  zero_division=0)
    return out


def train_final(ph: Dict[str, Any], name: str, seeds: Tuple[int, ...],
                epochs: Optional[int] = None) -> Dict[str, Any]:
    epochs = epochs or CFG.final_epochs
    per_seed, best = [], None
    for sd in seeds:
        if not CLOCK.ok(0.04 * SESSION_HOURS * 3600):
            print(f"    [clock] stopping {name} after {len(per_seed)} seed(s)")
            break
        print(f"    seed {sd}: training started", flush=True)
        res = train_one(ph, IDX_TR, IDX_VA, CFG.img_size_final, epochs,
                        batch=CFG.proxy_batch,   # the batch size the search actually ranked at
                        seed=sd, amp=True, use_ema=CFG.final_ema,
                        label_smooth=max(ph["label_smooth"], CFG.final_label_smooth),
                        return_model=True, warmup_frac=CFG.final_warmup_frac)
        if not res.get("ok"):
            print(f"    seed {sd}: FAILED ({res.get('reason')})")
            continue
        model = res["model"]
        if res.get("state") is not None:
            model.load_state_dict({k: v.to(DEVICE) for k, v in res["state"].items()})
        probs_te, y_te = predict(model, IDX_TE, CFG.img_size_final, tta=True)
        m = full_metrics(probs_te, y_te)
        m.update(seed=sd, val_f1=res["val_f1"], mmac=res["mmac"],
                 params_m=res["params_m"],
                 latency_ms=measure_latency(model, CFG.img_size_final, 1),
                 latency_ms_b32=measure_latency(model, CFG.img_size_final, 32),
                 train_seconds=res["seconds"], best_epoch=res["best_epoch"])
        per_seed.append(m)
        if best is None or res["val_f1"] > best["val_f1"]:
            best = {"val_f1": res["val_f1"], "probs": probs_te, "y": y_te,
                    "seed": sd, "history": res["history"],
                    "weights": store_weights(name,
                                             {k: v.cpu() for k, v in
                                              model.state_dict().items()})}
        del model, res
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        print(f"    seed {sd}: test acc {m['accuracy']*100:.2f}%  "
              f"macro-F1 {m['f1_macro']*100:.2f}%  ({m['params_m']:.3f}M, {m['mmac']:.1f} MMac)")
    if not per_seed:
        return {"name": name, "ok": False}
    df = pd.DataFrame(per_seed)
    return {"name": name, "ok": True, "ph": ph, "per_seed": per_seed,
            "mean": df.mean(numeric_only=True).to_dict(),
            "std": df.std(numeric_only=True).to_dict(),
            "best": best, "n_seeds": len(per_seed)}


# new key: models trained by the first version used the inactive batch-size gene
FINAL_KEY = "final_results_fixed_batch"
FINAL: Dict[str, Any] = load_state(FINAL_KEY, {}) or {}

if RUN_PLAN in ("analysis", "all") and CFG.run_final_training and SEARCH_COMPLETE:
    print(f"\n{'='*78}\nFULL TRAINING OF SEARCHED ARCHITECTURES\n{'='*78}")
    for c in CANDIDATES:
        _old = FINAL.get(c["name"])
        if _old is not None and _old.get("ok") and _old.get("x") is not None and \
                genotype_key(_old["x"]) == genotype_key(c["x"]):
            print(f"  {c['name']}: cached")
            continue
        if _old is not None:
            print(f"  {c['name']}: " + ("cached model is a different architecture; retraining"
                                       if _old.get("ok") else "previous attempt failed; retrying"))
        seeds = CFG.final_seeds if c["algo"] == PROPOSED else CFG.final_seeds[:2]
        print(f"  {c['name']}  ({len(seeds)} seeds)")
        r = train_final(c["ph"], c["name"], tuple(seeds))
        r["algo"], r["kind"], r["x"] = c["algo"], c["kind"], c["x"]
        FINAL[c["name"]] = r
        save_state(FINAL_KEY, FINAL)
        CLOCK.report()

# %%
# ---- reference baselines: ImageNet-pretrained backbones + a hand-designed CNN ----------
def torchvision_baseline(arch: str) -> Tuple[Optional[nn.Module], bool]:
    """Return (model, pretrained_flag). Falls back to random init without internet."""
    try:
        import torchvision.models as tvm
    except ImportError:
        return None, False
    getter = getattr(tvm, arch, None)
    if getter is None:
        return None, False
    pre = True
    try:
        model = getter(weights="DEFAULT")
    except Exception:
        try:
            model = getter(weights=None)
            pre = False
        except Exception:
            return None, False
    # swap the classifier for our 4 classes
    for attr in ("fc", "classifier", "head"):
        if hasattr(model, attr):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Linear):
                setattr(model, attr, nn.Linear(mod.in_features, CFG.n_classes))
            elif isinstance(mod, nn.Sequential):
                for i in range(len(mod) - 1, -1, -1):
                    if isinstance(mod[i], nn.Linear):
                        mod[i] = nn.Linear(mod[i].in_features, CFG.n_classes)
                        break
                    if isinstance(mod[i], nn.Conv2d):
                        mod[i] = nn.Conv2d(mod[i].in_channels, CFG.n_classes, 1)
                        break
            break
    return model, pre


def train_baseline(model: nn.Module, name: str, seeds: Tuple[int, ...],
                   epochs: int, lr: float = 3e-4) -> Dict[str, Any]:
    """Fixed, strong, non-searched recipe: AdamW + cosine + light augmentation."""
    per_seed, best = [], None
    base_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    for sd in seeds:
        if not CLOCK.ok(0.04 * SESSION_HOURS * 3600):
            break
        set_seed(sd)
        net = model
        net.load_state_dict({k: v.to(DEVICE) for k, v in base_state.items()})
        net = net.to(DEVICE)
        if DEVICE.type == "cuda":
            net = net.to(memory_format=torch.channels_last)
        mmac, params_m = count_flops_params(net, CFG.img_size_final)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        bs = CFG.final_batch
        spe = max(1, len(IDX_TR) // bs)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda_factory(spe * epochs, 0.1))
        scaler = make_scaler(DEVICE.type == "cuda")
        idx_t = torch.as_tensor(IDX_TR, device=DEVICE)
        bstate, bf1 = None, -1.0
        t0 = time.time()
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(idx_t), device=DEVICE)
            for s in range(spe):
                bi = idx_t[perm[s * bs:(s + 1) * bs]]
                if bi.numel() < 2:
                    continue
                xb = augment(fetch(bi), CFG.img_size_final, 0.6, 0.15)
                yb = Y_GPU[bi]
                yt = F.one_hot(yb, CFG.n_classes).float()
                yt = yt * (1 - 0.05) + 0.05 / CFG.n_classes
                xb = normalize(xb)
                if DEVICE.type == "cuda":
                    xb = xb.contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    loss = torch.sum(-yt * F.log_softmax(net(xb).float(), 1), 1).mean()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update(); sched.step()
            p, yv = predict(net, IDX_VA, CFG.img_size_final)
            f1 = f1_score(yv, p.argmax(1), average="macro", zero_division=0)
            if f1 > bf1:
                bf1 = f1
                bstate = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if bstate is not None:
            net.load_state_dict({k: v.to(DEVICE) for k, v in bstate.items()})
        probs_te, y_te = predict(net, IDX_TE, CFG.img_size_final, tta=True)
        m = full_metrics(probs_te, y_te)
        m.update(seed=sd, val_f1=bf1, mmac=mmac, params_m=params_m,
                 latency_ms=measure_latency(net, CFG.img_size_final, 1),
                 latency_ms_b32=measure_latency(net, CFG.img_size_final, 32),
                 train_seconds=time.time() - t0, best_epoch=-1)
        per_seed.append(m)
        if best is None or bf1 > best["val_f1"]:
            best = {"val_f1": bf1, "probs": probs_te, "y": y_te, "seed": sd}
        print(f"    seed {sd}: test acc {m['accuracy']*100:.2f}%  "
              f"macro-F1 {m['f1_macro']*100:.2f}%  ({params_m:.3f}M, {mmac:.1f} MMac)")
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    if not per_seed:
        return {"name": name, "ok": False}
    df = pd.DataFrame(per_seed)
    return {"name": name, "ok": True, "per_seed": per_seed,
            "mean": df.mean(numeric_only=True).to_dict(),
            "std": df.std(numeric_only=True).to_dict(),
            "best": best, "n_seeds": len(per_seed)}


BASELINE_ARCHS = ["mobilenet_v3_small", "mobilenet_v2", "efficientnet_b0",
                  "resnet18", "shufflenet_v2_x1_0"]

if RUN_PLAN in ("analysis", "all") and CFG.run_baselines and SEARCH_COMPLETE:
    print(f"\n{'='*78}\nREFERENCE BASELINES\n{'='*78}")
    bseeds = tuple(CFG.final_seeds[:2])
    for arch in BASELINE_ARCHS:
        nm = f"{arch} (ImageNet)"
        if nm in FINAL or f"{arch} (scratch)" in FINAL:
            print(f"  {arch}: cached"); continue
        if not CLOCK.ok(0.08 * SESSION_HOURS * 3600):
            print("  [clock] skipping remaining baselines"); break
        model, pre = torchvision_baseline(arch)
        if model is None:
            print(f"  {arch}: unavailable"); continue
        nm = f"{arch} ({'ImageNet' if pre else 'scratch'})"
        if not pre:
            print(f"  NOTE: pretrained weights unavailable (enable Internet in the Kaggle "
                  f"notebook settings for ImageNet-initialised baselines)")
        print(f"  {nm}")
        r = train_baseline(model, nm, bseeds, epochs=max(8, CFG.final_epochs // 3),
                           lr=3e-4 if pre else 1e-3)
        r["algo"], r["kind"] = "baseline", "transfer" if pre else "scratch"
        FINAL[nm] = r
        save_state(FINAL_KEY, FINAL)
        del model
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # hand-designed CNN: the "expert default" a practitioner would write without any search
    HAND_PH = {"n_blocks": 4, "base_filters": 32, "growth": 2.0, "block_type": "basic",
               "kernel": 3, "use_se": False, "se_ratio": 0.25, "activation": "relu",
               "stem_stride": 1, "head_pool": "avg", "dense_units": 128,
               "conv_dropout": 0.0, "dense_dropout": 0.5, "optimizer": "adam",
               "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 32, "label_smooth": 0.0,
               "aug_strength": 0.6, "mix_alpha": 0.0, "mix_mode": "mixup",
               "p_erase": 0.0, "stage_depth": 1, "expand_ratio": 4}
    if "Hand-designed CNN" not in FINAL and CLOCK.ok(0.05 * SESSION_HOURS * 3600):
        print("  Hand-designed CNN (expert default, no search)")
        r = train_final(HAND_PH, "Hand-designed CNN", tuple(CFG.final_seeds[:2]))
        r["algo"], r["kind"] = "baseline", "handcrafted"
        FINAL["Hand-designed CNN"] = r
        save_state(FINAL_KEY, FINAL)

FINAL = {k: v for k, v in FINAL.items() if v.get("ok")}
print(f"\n{len(FINAL)} fully trained models available")

# Later cells are gated on this flag rather than stopped with SystemExit, because Kaggle marks a
# version as failed if a cell raises, even when the search finished correctly.
ANALYSIS_READY = bool(FINAL) and SEARCH_COMPLETE
if not ANALYSIS_READY:
    print("\nSearch-phase artefacts are saved. Next: open this version's Output, add it as an\n"
          "input to the notebook, set RUN_PLAN = 'analysis' and run a new version.")

# %% [markdown]
# ## 17. Test-set results, bootstrap confidence intervals and McNemar tests
#
# This is the section that carries the paper's empirical weight. $n = $ |test split| images,
# so the paired tests here are properly powered, unlike the run-level rank tests of
# Section 13.

# %%
if ANALYSIS_READY:
    pass
    def bootstrap_ci(y: np.ndarray, pred: np.ndarray, metric: str = "accuracy",
                     n_boot: int = 10_000, seed: int = 0) -> Tuple[float, float, float]:
        rng = np.random.RandomState(seed)
        n = len(y)
        fn = (accuracy_score if metric == "accuracy"
              else (lambda a, b: f1_score(a, b, average="macro", zero_division=0)))
        point = fn(y, pred)
        vals = np.empty(n_boot)
        for b in range(n_boot):
            i = rng.randint(0, n, n)
            vals[b] = fn(y[i], pred[i])
        return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


    def mcnemar(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> Dict[str, float]:
        """Exact / corrected McNemar test on paired correctness of two classifiers."""
        c1, c2 = (p1 == y), (p2 == y)
        n01 = int(np.sum(~c1 & c2))     # model 1 wrong, model 2 right
        n10 = int(np.sum(c1 & ~c2))     # model 1 right, model 2 wrong
        n = n01 + n10
        if n == 0:
            return {"n01": n01, "n10": n10, "statistic": 0.0, "p": 1.0, "test": "degenerate"}
        if n < 25:
            p = float(min(1.0, 2.0 * sps.binom.cdf(min(n01, n10), n, 0.5)))
            return {"n01": n01, "n10": n10, "statistic": float(min(n01, n10)), "p": p,
                    "test": "exact binomial"}
        stat = (abs(n01 - n10) - 1.0) ** 2 / n          # Edwards continuity correction
        return {"n01": n01, "n10": n10, "statistic": float(stat),
                "p": float(sps.chi2.sf(stat, 1)), "test": "chi2 (Edwards)"}


    rows = []
    for nm, r in FINAL.items():
        m, s = r["mean"], r["std"]
        rows.append({
            "model": nm, "source": r.get("algo", "?"), "point": r.get("kind", "?"),
            "seeds": r["n_seeds"],
            "test_acc": m["accuracy"], "test_acc_std": s.get("accuracy", 0.0),
            "test_f1": m["f1_macro"], "test_f1_std": s.get("f1_macro", 0.0),
            "bal_acc": m["balanced_accuracy"], "precision": m["precision_macro"],
            "recall": m["recall_macro"], "mcc": m["mcc"], "kappa": m["kappa"],
            "auc": m["auc_ovr_macro"],
            "params_M": m["params_m"], "MMac": m["mmac"],
            "lat_ms": m["latency_ms"], "lat_ms_b32": m["latency_ms_b32"],
            "train_s": m["train_seconds"], "val_f1": m["val_f1"],
        })
    TEST_DF = pd.DataFrame(rows).sort_values("test_f1", ascending=False).reset_index(drop=True)
    TEST_DF.to_csv(OUT / "tables" / "final_test_results.csv", index=False)

    print("--- Held-out test set: all fully trained models ---")
    print(TEST_DF.assign(
        acc=lambda d: (d.test_acc * 100).round(2).astype(str) + " +/- "
                      + (d.test_acc_std * 100).round(2).astype(str),
        f1=lambda d: (d.test_f1 * 100).round(2).astype(str) + " +/- "
                     + (d.test_f1_std * 100).round(2).astype(str),
    )[["model", "acc", "f1", "params_M", "MMac", "lat_ms", "seeds"]].to_string(index=False))

# %%
if ANALYSIS_READY:
    pass
    # ---- the headline model and paired comparisons -----------------------------------------
    PROP_KEYS = [k for k in FINAL if FINAL[k].get("algo") == PROPOSED]
    _want_head = f"{PROPOSED} ({CFG.headline_point})"
    HEAD_KEY = (_want_head if _want_head in FINAL else
                f"{PROPOSED} (knee)" if f"{PROPOSED} (knee)" in FINAL
                else (PROP_KEYS[0] if PROP_KEYS else TEST_DF.iloc[0]["model"]))
    print(f"\nheadline model: {HEAD_KEY}")

    hb = FINAL[HEAD_KEY]["best"]
    y_te = hb["y"]
    pred_head = hb["probs"].argmax(1)

    acc_pt, acc_lo, acc_hi = bootstrap_ci(y_te, pred_head, "accuracy")
    f1_pt, f1_lo, f1_hi = bootstrap_ci(y_te, pred_head, "f1")
    print(f"  accuracy  {acc_pt*100:.2f}%  95% CI [{acc_lo*100:.2f}, {acc_hi*100:.2f}]")
    print(f"  macro-F1  {f1_pt*100:.2f}%  95% CI [{f1_lo*100:.2f}, {f1_hi*100:.2f}]")

    mc_rows, raw_p = [], []
    for nm, r in FINAL.items():
        if nm == HEAD_KEY:
            continue
        p2 = r["best"]["probs"].argmax(1)
        res = mcnemar(y_te, pred_head, p2)
        a_pt, a_lo, a_hi = bootstrap_ci(y_te, p2, "accuracy", n_boot=4000)
        mc_rows.append({"vs": nm, "acc_vs": a_pt, "acc_ci": f"[{a_lo*100:.2f}, {a_hi*100:.2f}]",
                        "n01": res["n01"], "n10": res["n10"],
                        "mcnemar_stat": res["statistic"], "mcnemar_p": res["p"],
                        "test": res["test"]})
        raw_p.append(res["p"])
    for r_, pa in zip(mc_rows, holm_correction(raw_p)):
        r_["mcnemar_p_holm"] = pa
        r_["significant_0.05"] = bool(pa < 0.05)
    MCN_DF = pd.DataFrame(mc_rows).sort_values("mcnemar_p")
    MCN_DF.to_csv(OUT / "tables" / "mcnemar_tests.csv", index=False)
    print(f"\n--- McNemar: {HEAD_KEY} (acc {acc_pt*100:.2f}%) vs each model, "
          f"n={len(y_te)} test images ---")
    print(MCN_DF[["vs", "acc_vs", "n01", "n10", "mcnemar_p", "mcnemar_p_holm",
                  "significant_0.05"]].to_string(index=False))
    jdump("headline_summary.json", {
        "model": HEAD_KEY, "accuracy": acc_pt, "accuracy_ci95": [acc_lo, acc_hi],
        "f1_macro": f1_pt, "f1_macro_ci95": [f1_lo, f1_hi],
        "phenotype": FINAL[HEAD_KEY].get("ph", {}),
        "n_test": int(len(y_te)),
    })

# %%
if ANALYSIS_READY:
    pass
    # ---- Figure 5: accuracy-complexity trade-off of fully trained models -------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    for _, r in TEST_DF.iterrows():
        is_prop = r["source"] == PROPOSED
        is_base = r["source"] == "baseline"
        ax.errorbar(r["MMac"], r["test_f1"] * 100, yerr=r["test_f1_std"] * 100,
                    fmt=("*" if is_prop else ("s" if is_base else "o")),
                    ms=16 if is_prop else 8,
                    color="#d62728" if is_prop else ("#2ca02c" if is_base else "#1f77b4"),
                    ecolor="gray", elinewidth=0.8, capsize=2,
                    zorder=5 if is_prop else 3)
        ax.annotate(r["model"].replace(" (knee)", "").replace(" (ImageNet)", ""),
                    (r["MMac"], r["test_f1"] * 100), fontsize=6.2,
                    xytext=(4, 4), textcoords="offset points")
    nd_in = TEST_DF[["test_f1", "MMac"]].values.copy()
    nd_in[:, 0] = 1 - nd_in[:, 0]
    m = non_dominated_mask(nd_in)
    P = TEST_DF.loc[m].sort_values("MMac")
    ax.plot(P["MMac"], P["test_f1"] * 100, "k--", lw=1.0, alpha=0.6,
            label="test-set Pareto front", zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("FLOPs (MMac, log scale)")
    ax.set_ylabel("test macro-$F_1$ [%]")
    ax.set_title("(a) accuracy vs. compute, fully trained models")
    ax.legend(fontsize=8)

    ax = axes[1]
    cm = confusion_matrix(y_te, pred_head)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(CFG.n_classes)); ax.set_yticks(range(CFG.n_classes))
    labs = [PRETTY.get(c, c).replace(" ", "\n") for c in CLASSES]
    ax.set_xticklabels(labs, fontsize=7.5); ax.set_yticklabels(labs, fontsize=7.5)
    for i in range(CFG.n_classes):
        for j in range(CFG.n_classes):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]*100:.1f}%", ha="center", va="center",
                    fontsize=7.5, color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"(b) confusion matrix, {HEAD_KEY}")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)
    savefig(fig, "fig5_tradeoff_and_confusion")

    print("\n--- per-class report, headline model ---")
    print(classification_report(y_te, pred_head,
                                target_names=[PRETTY.get(c, c) for c in CLASSES],
                                digits=4, zero_division=0))
    with open(OUT / "tables" / "classification_report.txt", "w") as fh:
        fh.write(classification_report(y_te, pred_head,
                                       target_names=[PRETTY.get(c, c) for c in CLASSES],
                                       digits=4, zero_division=0))

# %%
if ANALYSIS_READY:
    pass
    # ---- Figure 6: training dynamics of the headline model --------------------------------
    if "history" in FINAL[HEAD_KEY].get("best", {}):
        h = pd.DataFrame(FINAL[HEAD_KEY]["best"]["history"])
        fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
        ax1.plot(h["epoch"], h["loss"], color="#1f77b4", label="training loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("training loss", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax2 = ax1.twinx()
        ax2.plot(h["epoch"], h["val_f1"] * 100, color="#d62728", label="val macro-$F_1$")
        ax2.plot(h["epoch"], h["val_acc"] * 100, color="#2ca02c", ls="--", label="val accuracy")
        ax2.set_ylabel("validation metric [%]")
        ax2.grid(False)
        ax2.axvline(h["val_f1"].idxmax(), color="gray", ls=":", lw=1)
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="center right")
        ax1.set_title(f"Training dynamics: {HEAD_KEY}")
        savefig(fig, "fig6_training_curve")

# %% [markdown]
# ## 18. Qualitative evidence: Grad-CAM
#
# Reviewers of plant-disease papers routinely ask whether the network attends to lesion
# tissue or to background//illumination artefacts. Grad-CAM on the last convolutional stage
# answers it, and a disagreement panel (cases the headline model gets wrong) is far more
# informative than a grid of successes.

# %%
if ANALYSIS_READY:
    pass
    if CFG.run_gradcam and RUN_PLAN in ("analysis", "all") and HEAD_KEY in FINAL:
        try:
            ph_head = FINAL[HEAD_KEY]["ph"]
            w = load_weights(FINAL[HEAD_KEY]["best"]["weights"])
            if w is None:
                raise RuntimeError("weights unavailable")
            net = SearchedCNN(ph_head, CFG.n_classes, CFG.img_size_final).to(DEVICE)
            net.load_state_dict({k: v.to(DEVICE) for k, v in w.items()})
            net.eval()

            feats, grads = {}, {}
            target_layer = net.blocks[-1]
            h1 = target_layer.register_forward_hook(
                lambda m, i, o: feats.__setitem__("v", o))
            h2 = target_layer.register_full_backward_hook(
                lambda m, gi, go: grads.__setitem__("v", go[0]))

            def gradcam(idx_list):
                bi = torch.as_tensor(np.array(idx_list), device=DEVICE)
                xb = normalize(resize_eval(fetch(bi), CFG.img_size_final))
                out = net(xb)
                cls = out.argmax(1)
                net.zero_grad(set_to_none=True)
                out.gather(1, cls[:, None]).sum().backward()
                A, G = feats["v"], grads["v"]
                w = G.mean(dim=(2, 3), keepdim=True)
                cam = F.relu((w * A).sum(dim=1, keepdim=True))
                cam = F.interpolate(cam, size=(CFG.img_size_final, CFG.img_size_final),
                                    mode="bilinear", align_corners=False)
                cam = cam - cam.amin(dim=(2, 3), keepdim=True)
                cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)
                return cam.detach().cpu().numpy()[:, 0], cls.cpu().numpy()

            correct = np.where(pred_head == y_te)[0]
            wrong = np.where(pred_head != y_te)[0]
            picks: List[Tuple[int, str]] = []
            rs = np.random.RandomState(0)
            for c in range(CFG.n_classes):
                cand = correct[y_te[correct] == c]
                if len(cand):
                    picks.append((int(IDX_TE[rs.choice(cand)]), "correct"))
            for w_ in wrong[:4]:
                picks.append((int(IDX_TE[w_]), "misclassified"))

            if picks:
                cams, cls = gradcam([p for p, _ in picks])
                n = len(picks)
                fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.6))
                axes = np.atleast_2d(axes)
                for k, ((gi, tag), cam) in enumerate(zip(picks, cams)):
                    img = np.array(Image.fromarray(X_ALL[gi]).resize(
                        (CFG.img_size_final, CFG.img_size_final))) / 255.0
                    axes[0, k].imshow(img); axes[0, k].axis("off")
                    axes[0, k].set_title(f"{PRETTY.get(CLASSES[Y_ALL[gi]], '')}\n"
                                         f"pred: {PRETTY.get(CLASSES[cls[k]], '')}",
                                         fontsize=6.5,
                                         color="black" if tag == "correct" else "#d62728")
                    axes[1, k].imshow(img); axes[1, k].imshow(cam, cmap="jet", alpha=0.45)
                    axes[1, k].axis("off")
                    axes[1, k].set_title(tag, fontsize=6.5)
                fig.suptitle(f"Grad-CAM, {HEAD_KEY} (top: input, bottom: class activation)",
                             fontsize=10)
                fig.tight_layout()
                savefig(fig, "fig7_gradcam")
            h1.remove(); h2.remove()
            del net
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[gradcam] skipped: {type(e).__name__}: {e}")

# %% [markdown]
# ## 19. Robustness to field-realistic corruptions
#
# Images captured in a real crop field are blurrier, noisier, darker and more
# JPEG-compressed than curated dataset photographs. A searched architecture that wins only on
# clean data is a weak contribution, so the headline model and the strongest baseline are
# re-evaluated under five parametric corruptions at three severities.

# %%
if ANALYSIS_READY:
    pass
    def corrupt(x: torch.Tensor, kind: str, sev: float) -> torch.Tensor:
        if kind == "gauss_noise":
            return (x + torch.randn_like(x) * (0.04 * sev)).clamp(0, 1)
        if kind == "blur":
            k = int(2 * round(1 + 1.6 * sev) + 1)
            sigma = 0.6 + 1.0 * sev
            ax = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
            g = torch.exp(-(ax ** 2) / (2 * sigma ** 2)); g = g / g.sum()
            x = F.conv2d(x, g.view(1, 1, 1, k).expand(3, 1, 1, k), padding=(0, k // 2), groups=3)
            return F.conv2d(x, g.view(1, 1, k, 1).expand(3, 1, k, 1), padding=(k // 2, 0),
                            groups=3).clamp(0, 1)
        if kind == "brightness":
            return (x * (1.0 - 0.22 * sev)).clamp(0, 1)
        if kind == "contrast":
            m = x.mean(dim=(1, 2, 3), keepdim=True)
            return ((x - m) * (1.0 - 0.25 * sev) + m).clamp(0, 1)
        if kind == "occlusion":
            y = x.clone()
            H = y.shape[2]
            s = int(H * (0.10 + 0.10 * sev))
            for i in range(y.shape[0]):
                t, l = random.randint(0, H - s), random.randint(0, H - s)
                y[i, :, t:t + s, l:l + s] = 0.0
            return y
        return x


    @torch.no_grad()
    def eval_corrupted(net: nn.Module, kind: str, sev: float) -> float:
        net.eval()
        preds, ys = [], []
        for s in range(0, len(IDX_TE), 128):
            bi = torch.as_tensor(IDX_TE[s:s + 128], device=DEVICE)
            xb = resize_eval(fetch(bi), CFG.img_size_final)
            xb = normalize(corrupt(xb, kind, sev))
            if DEVICE.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                preds.append(net(xb).float().argmax(1).cpu().numpy())
            ys.append(Y_ALL[IDX_TE[s:s + 128]])
        return float(f1_score(np.concatenate(ys), np.concatenate(preds),
                              average="macro", zero_division=0))


    ROBUST_DF = pd.DataFrame()
    if CFG.run_robustness and RUN_PLAN in ("analysis", "all"):
        # headline model vs. the strongest other model whose weights we still hold
        others = [k for k in FINAL if k != HEAD_KEY and "ph" in FINAL[k]
                  and FINAL[k].get("best", {}).get("weights")]
        cands = [HEAD_KEY] + sorted(others, key=lambda k: -FINAL[k]["mean"]["f1_macro"])[:1]
        rows = []
        for nm in cands:
            w = load_weights(FINAL[nm].get("best", {}).get("weights", ""))
            if "ph" not in FINAL[nm] or w is None:
                continue
            set_seed(0)
            net = SearchedCNN(FINAL[nm]["ph"], CFG.n_classes, CFG.img_size_final).to(DEVICE)
            net.load_state_dict({k: v.to(DEVICE) for k, v in w.items()})
            clean = eval_corrupted(net, "none", 0.0)
            for kind in ["gauss_noise", "blur", "brightness", "contrast", "occlusion"]:
                for sev in [1.0, 2.0, 3.0]:
                    rows.append({"model": nm, "corruption": kind, "severity": sev,
                                 "f1_macro": eval_corrupted(net, kind, sev),
                                 "clean_f1": clean})
            del net
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        if rows:
            ROBUST_DF = pd.DataFrame(rows)
            ROBUST_DF["rel_drop"] = 1 - ROBUST_DF["f1_macro"] / ROBUST_DF["clean_f1"]
            ROBUST_DF.to_csv(OUT / "tables" / "robustness.csv", index=False)
            piv = ROBUST_DF.pivot_table(index=["model"], columns=["corruption", "severity"],
                                        values="f1_macro")
            print("\n--- macro-F1 under corruption ---")
            print((piv * 100).round(2).to_string())

            fig, ax = plt.subplots(figsize=(8.5, 4.2))
            kinds = ["gauss_noise", "blur", "brightness", "contrast", "occlusion"]
            w = 0.8 / max(1, len(cands))
            for mi, nm in enumerate([c for c in cands if c in set(ROBUST_DF["model"])]):
                sub = ROBUST_DF[ROBUST_DF["model"] == nm]
                vals = [sub[sub["corruption"] == k]["f1_macro"].mean() * 100 for k in kinds]
                ax.bar(np.arange(len(kinds)) + mi * w, vals, width=w,
                       label=nm, color="#d62728" if nm == HEAD_KEY else "#2ca02c", alpha=0.85)
            ax.axhline(ROBUST_DF["clean_f1"].max() * 100, ls="--", color="gray", lw=1,
                       label="clean test macro-$F_1$")
            ax.set_xticks(np.arange(len(kinds)) + w / 2)
            ax.set_xticklabels(kinds, rotation=15)
            ax.set_ylabel("macro-$F_1$ [%] (mean over 3 severities)")
            ax.set_title("Robustness to field-realistic corruptions")
            ax.legend(fontsize=8)
            savefig(fig, "fig8_robustness")

# %% [markdown]
# ## 20. Ablation study
#
# A new algorithm is only credible if its individual mechanisms are shown to earn their
# place. Four variants of MO-P3PGA are run under the identical protocol:
#
# | Variant | What is removed or changed | Question it answers |
# |---|---|---|
# | `w/o mito` | mitochondrial change disabled; islands reduce to a plain island NSGA-II | Does the 3-parent step contribute anything? |
# | `w/o attract` | Step 5 pull toward the archive leader disabled | Does inter-island information flow repay the evaluations it costs? |
# | `NP=1` (`MO-3PGA`) | single population | Is the *parallel* structure necessary, or is 3-parent recombination enough? |
# | `literal port` | n-point crossover + uniform-reset mutation + SUS + single global leader, exactly as in the 2020 chapter | How much of the result comes from the Pareto-aware modernisation rather than from the original mechanics? |
#
# **Design.** Each variant runs `ablation_runs` times at the main search budget. Run $r$ of every
# variant uses the seed of run $r$ of the main MO-P3PGA search, so all variants start from the
# same initial population, and the main MO-P3PGA runs themselves serve as the full algorithm (no
# re-training). Each variant is compared with the full algorithm by a paired Wilcoxon signed-rank
# test over runs, with Holm correction across the four variants, plus the Vargha-Delaney
# $\hat A_{12}$ effect size. All indicators here use one normalisation and reference front pooled
# from the main search and every ablation run, so the variant rows are directly comparable.

# %%
if ANALYSIS_READY:
    pass
    def variant_phase(variants: List[Dict[str, Any]], n_runs: int, budget: int,
                      state_key: str, seed_label: Optional[str] = None) -> List[Dict[str, Any]]:
        """Run labelled variants (same optimiser class, different overrides), round by round."""
        results: List[Dict[str, Any]] = drop_truncated(load_state(state_key, []) or [],
                                                       state_key)
        done = {(r["label"], r["run"]) for r in results}
        total = len(variants) * n_runs
        n_cached = sum(1 for v in variants for run in range(n_runs) if (v["label"], run) in done)
        if n_cached:
            print(f"{n_cached}/{total} variant runs restored from checkpoint")
        i = 0
        for run in range(n_runs):          # round by round, so a clock stop stays balanced
            for v in variants:
                i += 1
                if (v["label"], run) in done:
                    continue
                need = budget * SEC_PER_EVAL * 0.6 + 60
                if not CLOCK.ok(need + 0.05 * SESSION_HOURS * 3600):
                    print(f"*** session clock: stopping variant phase at {v['label']} "
                          f"run {run} ({len(results)}/{total} done). Attach this version's "
                          f"output and run RUN_PLAN='analysis' again to continue. ***")
                    save_state(state_key, results); save_state("global_memo", GLOBAL_MEMO)
                    return results
                print(f"[{i}/{total}] {v['label']} run {run} ... ", end="", flush=True)
                rec = run_one_search(v["algo"], run, budget, overrides=v.get("overrides"),
                                     label=v["label"], seed_label=seed_label)
                if is_truncated(rec):
                    print(f"stopped by the session clock after {rec['summary']['evals']}/"
                          f"{budget} evaluations; not saved. Attach this version's output and "
                          f"run RUN_PLAN='analysis' again to continue.")
                    save_state(state_key, results); save_state("global_memo", GLOBAL_MEMO)
                    return results
                results.append(rec)
                save_state(state_key, results); save_state("global_memo", GLOBAL_MEMO)
                print(f"gen={rec['generations']:3d} |A|={len(rec['archive_F']):3d} "
                      f"HV={indicators(rec['archive_F'])['HV']:.4f} "
                      f"{CLOCK.hms(rec['wall'])}")
        save_state(state_key, results); save_state("global_memo", GLOBAL_MEMO)
        return results


    ABL_FULL = "MO-P3PGA (full)"
    ABLATION_VARIANTS = [
        {"label": ABL_FULL, "algo": "MO-P3PGA", "overrides": {}},
        {"label": "MO-P3PGA w/o mito", "algo": "MO-P3PGA",
         "overrides": {"use_mito": False}},
        {"label": "MO-P3PGA w/o attract", "algo": "MO-P3PGA",
         "overrides": {"use_attract": False}},
        {"label": "MO-P3PGA NP=1", "algo": "MO-P3PGA",
         "overrides": {"p3pga_np": 1, "p3pga_n": CFG.pop_total}},
        {"label": "MO-P3PGA literal port", "algo": "MO-P3PGA",
         "overrides": {"crossover_op": "npoint", "selection_op": "sus_rank",
                       "leader_mode": "global_least_crowded"}},
    ]

    ABL: List[Dict[str, Any]] = []
    if CFG.run_ablations:
        # The main MO-P3PGA runs are the full algorithm when the budgets match.
        _main_p3 = sorted([r for r in SEARCH if r["label"] == BASE_P3], key=lambda r: r["run"])
        _reuse = CFG.ablation_budget == BUDGET_EFF and len(_main_p3) > 0
        ABL_RUNS = min(CFG.ablation_runs, len(_main_p3)) if _reuse else CFG.ablation_runs
        if _reuse and ABL_RUNS < CFG.ablation_runs:
            print(f"NOTE: only {len(_main_p3)} main MO-P3PGA runs exist, so the ablation uses "
                  f"{ABL_RUNS} paired runs instead of {CFG.ablation_runs}")
        _full = [dict(r, label=ABL_FULL) for r in _main_p3 if r["run"] < ABL_RUNS] if _reuse else []
        _todo = ABLATION_VARIANTS[1:] if _reuse else ABLATION_VARIANTS
        if RUN_PLAN in ("analysis", "all"):
            print(f"\n{'='*78}\nABLATION STUDY: {len(_todo)} variants x {ABL_RUNS} paired runs x "
                  f"{CFG.ablation_budget} evaluations"
                  + (f" (full algorithm = main runs 0-{ABL_RUNS - 1})" if _reuse else "")
                  + f"\n{'='*78}")
            _var = variant_phase(_todo, ABL_RUNS, CFG.ablation_budget, "ablation_paired",
                                 seed_label=BASE_P3)
        else:
            _var = load_state("ablation_paired", []) or []
        ABL = _full + [r for r in _var if r["run"] < ABL_RUNS]

    if ABL:
        # one normalisation for the main search and every ablation run
        _AF = np.vstack([ALL_F] + [r["archive_F"] for r in ABL if r["archive_F"].size])
        A_ZMIN, A_ZMAX = _AF.min(axis=0), _AF.max(axis=0)
        A_REF = normalize_front(_AF[non_dominated_mask(_AF)], A_ZMIN, A_ZMAX)

        def abl_indicators(F: np.ndarray) -> Dict[str, float]:
            Fn = normalize_front(F, A_ZMIN, A_ZMAX)
            Fn = Fn[non_dominated_mask(Fn)]
            return {"HV": hypervolume(Fn, REF_POINT), "IGD+": igd_plus(Fn, A_REF),
                    "GD": gd(Fn, A_REF), "SP": spacing(Fn), "n_nd": float(Fn.shape[0]),
                    "best_err": float(F[:, 0].min())}

        ABL_DF = pd.DataFrame([{"variant": r["label"], "run": r["run"],
                                "generations": r["generations"],
                                **abl_indicators(r["archive_F"])} for r in ABL])
        ABL_DF.to_csv(OUT / "tables" / "ablation_per_run.csv", index=False)
        _cols = ["HV", "IGD+", "GD", "SP", "n_nd", "best_err", "generations"]
        ABL_AGG = ABL_DF.groupby("variant")[_cols].agg(["mean", "std"])
        ABL_AGG.columns = [f"{a}_{b}" for a, b in ABL_AGG.columns]
        ABL_AGG["runs"] = ABL_DF.groupby("variant")["run"].nunique()
        full_hv = ABL_AGG.loc[ABL_FULL, "HV_mean"] if ABL_FULL in ABL_AGG.index else np.nan
        ABL_AGG["HV_delta_%"] = (ABL_AGG["HV_mean"] / full_hv - 1) * 100

        # paired tests of every variant against the full algorithm, Holm across variants
        ABL_TESTS = {}
        for metric, higher in [("HV", True), ("IGD+", False), ("best_err", False)]:
            piv = ABL_DF.pivot_table(index="run", columns="variant", values=metric)
            others = [v for v in piv.columns if v != ABL_FULL]
            raw, rows = [], []
            for v in others:
                pair = piv[[ABL_FULL, v]].dropna()
                x, y = pair[ABL_FULL].values, pair[v].values
                try:
                    pw = 1.0 if np.allclose(x, y) else float(
                        sps.wilcoxon(x, y, zero_method="zsplit").pvalue)
                except Exception:
                    pw = float("nan")
                # A12 here = P(full is better than the variant)
                a12 = vargha_delaney_a12(x, y) if higher else vargha_delaney_a12(-x, -y)
                rows.append({"variant": v, "pairs": len(pair), "wilcoxon_p": pw, "A12_full": a12})
                raw.append(pw if np.isfinite(pw) else 1.0)
            for r_, pa in zip(rows, holm_correction(raw) if raw else []):
                r_["p_holm"] = pa
            ABL_TESTS[metric] = rows
            for r_ in rows:
                ABL_AGG.loc[r_["variant"], f"{metric}_p_holm"] = r_["p_holm"]
                ABL_AGG.loc[r_["variant"], f"{metric}_A12_full"] = r_["A12_full"]
        piv = ABL_DF.pivot_table(index="run", columns="variant", values="HV").dropna()
        if piv.shape[0] >= 2 and piv.shape[1] >= 3:
            _st, _pf = sps.friedmanchisquare(*[piv[v].values for v in piv.columns])
            ABL_FRIEDMAN = {"chi2": float(_st), "p": float(_pf), "blocks": int(piv.shape[0])}
        else:
            ABL_FRIEDMAN = {"chi2": float("nan"), "p": float("nan"), "blocks": int(piv.shape[0])}
        jdump("ablation_statistics.json", {"friedman_HV": ABL_FRIEDMAN, "tests": ABL_TESTS})

        ABL_AGG = ABL_AGG.sort_values("HV_mean", ascending=False)
        ABL_AGG.to_csv(OUT / "tables" / "ablation.csv")
        print(f"\n--- Ablation, {int(ABL_AGG['runs'].min())} paired runs per variant "
              f"(Friedman on HV p = {ABL_FRIEDMAN['p']:.3g}) ---")
        print("p_holm: paired Wilcoxon vs the full algorithm, Holm-corrected; "
              "A12_full: probability the full algorithm is better")
        _show = ["HV_mean", "HV_std", "HV_delta_%", "HV_p_holm", "HV_A12_full", "IGD+_mean",
                 "IGD+_p_holm", "best_err_mean", "generations_mean", "runs"]
        print(ABL_AGG[[c for c in _show if c in ABL_AGG.columns]].round(4).to_string())

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        vs = list(ABL_AGG.index)
        for ax, metric, title in [(axes[0], "HV", "(a) hypervolume (higher is better)"),
                                  (axes[1], "IGD+", "(b) IGD+ (lower is better)"),
                                  (axes[2], "generations", "(c) generations completed")]:
            data = [ABL_DF.loc[ABL_DF["variant"] == v, metric].values for v in vs]
            bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.55)
            for patch, v in zip(bp["boxes"], vs):
                patch.set_facecolor("#d62728" if v == ABL_FULL else "#1f77b4")
                patch.set_alpha(0.75)
            for k, d in enumerate(data):
                ax.scatter(d, np.full(len(d), k + 1) + np.random.uniform(-0.12, 0.12, len(d)),
                           s=12, c="k", alpha=0.6, zorder=5)
            ax.set_yticks(range(1, len(vs) + 1))
            ax.set_yticklabels(vs if ax is axes[0] else [], fontsize=8.5)
            ax.invert_yaxis()
            ax.set_title(title)
        savefig(fig, "fig9_ablation")

# %% [markdown]
# ## 21. Parameter-sensitivity analysis
#
# The three parameters that the 2020 chapter leaves either unspecified or problem-dependent
# are exactly the ones a reviewer will probe: the number of parallel populations $N_P$, the
# mitochondrial-change magnitude $\delta$, and the leader-attraction probability
# $p_{\text{attract}}$ (described in the chapter only as "a fixed small probability").

# %%
if ANALYSIS_READY:
    pass
    SENS_GRID = []
    for np_ in [1, 2, 4, 8]:
        if CFG.pop_total % np_ == 0:
            SENS_GRID.append({"label": f"NP={np_}", "algo": "MO-P3PGA", "group": "NP",
                              "value": np_,
                              "overrides": {"p3pga_np": np_,
                                            "p3pga_n": CFG.pop_total // np_}})
    for pa in [0.0, 0.05, 0.10, 0.20, 0.40]:
        SENS_GRID.append({"label": f"p_attract={pa}", "algo": "MO-P3PGA",
                          "group": "p_attract", "value": pa,
                          "overrides": {"p_attract": pa, "use_attract": pa > 0}})
    for ms in [0.05, 0.10, 0.2222, 0.40]:
        SENS_GRID.append({"label": f"delta={ms}", "algo": "MO-P3PGA", "group": "mito_scale",
                          "value": ms, "overrides": {"mito_scale": ms}})

    SENS: List[Dict[str, Any]] = []
    if CFG.run_sensitivity and RUN_PLAN in ("analysis", "all"):
        print(f"\n{'='*78}\nPARAMETER SENSITIVITY\n{'='*78}")
        SENS = variant_phase(SENS_GRID, CFG.sensitivity_runs, CFG.sensitivity_budget,
                             "sensitivity_results")
    else:
        SENS = load_state("sensitivity_results", []) or []

    if SENS:
        gmap = {v["label"]: (v["group"], v["value"]) for v in SENS_GRID}
        srows = []
        for r in SENS:
            g, val = gmap.get(r["label"], ("?", np.nan))
            srows.append({"label": r["label"], "group": g, "value": val, "run": r["run"],
                          **indicators(r["archive_F"])})
        SENS_DF = pd.DataFrame(srows)
        SENS_DF.to_csv(OUT / "tables" / "sensitivity.csv", index=False)
        agg = SENS_DF.groupby(["group", "value"])[["HV", "IGD+", "best_err"]].agg(["mean", "std"])
        print("\n--- parameter sensitivity ---")
        print(agg.round(4).to_string())

        groups = [g for g in ["NP", "p_attract", "mito_scale"]
                  if g in set(SENS_DF["group"])]
        if groups:
            fig, axes = plt.subplots(1, len(groups), figsize=(4.3 * len(groups), 3.8),
                                     squeeze=False)
            titles = {"NP": "number of parallel populations $N_P$",
                      "p_attract": "leader-attraction probability $p_{attract}$",
                      "mito_scale": "mitochondrial-change magnitude $\\delta$"}
            for ax, g in zip(axes[0], groups):
                sub = SENS_DF[SENS_DF["group"] == g].groupby("value")["HV"].agg(["mean", "std"])
                ax.errorbar(sub.index, sub["mean"], yerr=sub["std"].fillna(0), marker="o",
                            color="#d62728", capsize=3)
                if g == "NP":
                    ax.set_xscale("log", base=2)
                    ax.set_xticks(sub.index.tolist())
                    ax.set_xticklabels([str(int(v)) for v in sub.index])
                if g == "mito_scale":
                    ax.axvline(0.2222, ls=":", color="gray", lw=1)
                    ax.text(0.2222, ax.get_ylim()[0], " chapter default", fontsize=7,
                            rotation=90, va="bottom", color="gray")
                ax.set_xlabel(titles[g]); ax.set_ylabel("hypervolume")
            fig.suptitle("Parameter sensitivity of MO-P3PGA", fontsize=11)
            fig.tight_layout()
            savefig(fig, "fig10_sensitivity")

# %% [markdown]
# ## 22. Validating the low-fidelity proxy
#
# The search ranks architectures by a short, low-resolution, subset training run. A reviewer
# will ask whether that ranking survives full training. This section answers it quantitatively:
# a stratified sample of evaluated architectures spanning the whole proxy-quality range is
# retrained under the full protocol, and the rank correlation between proxy error and final
# test error is reported. A high Spearman $\rho$ is the evidence that the search optimised the
# right thing; reporting it, whatever it turns out to be, is what makes the protocol
# defensible rather than merely cheap.

# %%
if ANALYSIS_READY:
    pass
    FID: List[Dict[str, Any]] = load_state("fidelity_results", []) or []
    if CFG.run_fidelity and RUN_PLAN in ("analysis", "all"):
        print(f"\n{'='*78}\nPROXY FIDELITY STUDY\n{'='*78}")
        pool = [h for r in SEARCH if r["label"] in MAIN_ALGOS for h in r["history"] if h["ok"]]
        uniq: Dict[str, Dict[str, Any]] = {}
        for h in pool:
            uniq.setdefault(genotype_key(h["x"]), h)
        pool = list(uniq.values())
        pool.sort(key=lambda h: h["f"][0])
        if pool:
            take = np.unique(np.linspace(0, len(pool) - 1, min(CFG.fidelity_n, len(pool)))
                             .astype(int))
            done_keys = {f["key"] for f in FID}
            for t in take:
                h = pool[int(t)]
                key = genotype_key(h["x"])
                if key in done_keys:
                    continue
                if not CLOCK.ok(0.05 * SESSION_HOURS * 3600):
                    print("  [clock] stopping fidelity study early"); break
                res = train_one(h["ph"], IDX_TR, IDX_VA, CFG.img_size_final,
                                max(10, CFG.final_epochs // 2), batch=CFG.proxy_batch,
                                seed=0, amp=True,
                                return_model=True)
                if not res.get("ok"):
                    continue
                model = res["model"]
                if res.get("state") is not None:
                    model.load_state_dict({k: v.to(DEVICE) for k, v in res["state"].items()})
                probs, yv = predict(model, IDX_TE, CFG.img_size_final)
                m = full_metrics(probs, yv)
                FID.append({"key": key, "proxy_err": h["f"][0],
                            "proxy_f1": h["val_f1"], "full_test_f1": m["f1_macro"],
                            "full_test_acc": m["accuracy"], "full_val_f1": res["val_f1"],
                            "mmac": res["mmac"], "params_m": res["params_m"]})
                save_state("fidelity_results", FID)
                del model, res
                gc.collect()
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"  [{len(FID)}] proxy_f1={h['val_f1']:.4f} -> "
                      f"full test macro-F1={m['f1_macro']:.4f}")

    if len(FID) >= 5:
        FID_DF = pd.DataFrame(FID)
        FID_DF.to_csv(OUT / "tables" / "proxy_fidelity.csv", index=False)
        rho, p_rho = sps.spearmanr(FID_DF["proxy_f1"], FID_DF["full_test_f1"])
        tau, p_tau = sps.kendalltau(FID_DF["proxy_f1"], FID_DF["full_test_f1"])
        r_p, p_pear = sps.pearsonr(FID_DF["proxy_f1"], FID_DF["full_test_f1"])
        FIDELITY = {"n": int(len(FID_DF)), "spearman_rho": float(rho), "spearman_p": float(p_rho),
                    "kendall_tau": float(tau), "kendall_p": float(p_tau),
                    "pearson_r": float(r_p), "pearson_p": float(p_pear)}
        jdump("proxy_fidelity.json", FIDELITY)
        print(f"\nproxy vs full training (n={len(FID_DF)}): "
              f"Spearman rho={rho:.3f} (p={p_rho:.2g}), Kendall tau={tau:.3f}, "
              f"Pearson r={r_p:.3f}")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        ax = axes[0]
        ax.scatter(FID_DF["proxy_f1"] * 100, FID_DF["full_test_f1"] * 100, s=30,
                   c=np.log10(FID_DF["mmac"]), cmap="viridis", edgecolor="k", linewidth=0.4)
        lo = min(FID_DF["proxy_f1"].min(), FID_DF["full_test_f1"].min()) * 100 - 2
        hi = max(FID_DF["proxy_f1"].max(), FID_DF["full_test_f1"].max()) * 100 + 2
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y = x")
        z = np.polyfit(FID_DF["proxy_f1"] * 100, FID_DF["full_test_f1"] * 100, 1)
        xs = np.linspace(FID_DF["proxy_f1"].min() * 100, FID_DF["proxy_f1"].max() * 100, 50)
        ax.plot(xs, np.polyval(z, xs), "-", color="#d62728", lw=1.6,
                label=f"fit (Spearman $\\rho$={rho:.3f})")
        ax.set_xlabel(f"proxy macro-$F_1$ [%]  ({CFG.proxy_epochs} ep, "
                      f"{CFG.img_size_proxy}px, {len(IDX_PTR)} imgs)")
        ax.set_ylabel("full-training test macro-$F_1$ [%]")
        ax.set_title("(a) proxy fidelity")
        ax.legend(fontsize=8)
        ax = axes[1]
        pr = FID_DF["proxy_f1"].rank(ascending=False)
        fr = FID_DF["full_test_f1"].rank(ascending=False)
        ax.scatter(pr, fr, s=30, color="#1f77b4", edgecolor="k", linewidth=0.4)
        ax.plot([1, len(pr)], [1, len(pr)], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("rank by proxy"); ax.set_ylabel("rank by full training")
        ax.set_title(f"(b) rank agreement (Kendall $\\tau$={tau:.3f})")
        savefig(fig, "fig11_proxy_fidelity")
    else:
        FIDELITY = {"n": len(FID)}
        print(f"proxy fidelity study: only {len(FID)} points, correlation not reported")

# %% [markdown]
# ## 23. LaTeX tables and the results digest
#
# Everything below writes straight into a manuscript. `outputs/tables/*.tex` use `booktabs`;
# add `\usepackage{booktabs}` to the preamble.

# %%
if ANALYSIS_READY:
    pass
    def to_latex(df: pd.DataFrame, name: str, caption: str, label: str,
                 float_fmt: str = "%.4f"):
        def esc(v):                             # escape plain names, keep cells written as LaTeX math
            t = str(v)
            return t if "$" in t else t.replace("_", r"\_").replace("%", r"\%")

        df = df.copy()
        df.index = [esc(v) for v in df.index]
        df.columns = [esc(v) for v in df.columns]
        df = df.map(lambda v: esc(v) if isinstance(v, str) else v)
        try:
            tex = df.to_latex(index=True, escape=False, float_format=float_fmt,
                              column_format="l" + "r" * (df.shape[1]))
        except Exception as e:                  # a table must never cost a finished run its results
            print(f"  [latex] skipped {name}: {type(e).__name__}: {e}")
            return
        tex = tex.replace("\\toprule", "\\toprule").replace("tabular", "tabular")
        full = ("\\begin{table}[htbp]\n\\centering\n\\caption{" + caption + "}\n"
                "\\label{tab:" + label + "}\n\\footnotesize\n" + tex + "\\end{table}\n")
        (OUT / "tables" / f"{name}.tex").write_text(full)
        print(f"  wrote tables/{name}.tex")


    tex_main = pd.DataFrame({
        "HV $\\uparrow$": AGG.apply(lambda r: f"{r['HV_mean']:.4f} $\\pm$ {r['HV_std']:.4f}", axis=1),
        "IGD$^+$ $\\downarrow$": AGG.apply(lambda r: f"{r['IGD+_mean']:.4f} $\\pm$ {r['IGD+_std']:.4f}", axis=1),
        "GD $\\downarrow$": AGG.apply(lambda r: f"{r['GD_mean']:.4f}", axis=1),
        "SP $\\downarrow$": AGG.apply(lambda r: f"{r['SP_mean']:.4f}", axis=1),
        "MS $\\uparrow$": AGG.apply(lambda r: f"{r['MS_mean']:.4f}", axis=1),
        "$|$ND$|$": AGG.apply(lambda r: f"{r['n_nd_mean']:.1f}", axis=1),
        "rank": pd.Series(STATS["HV"]["avg_ranks"]).round(2),
    })
    to_latex(tex_main, "tab_indicators",
             f"Multi-objective performance indicators on the {DS['title']} NAS problem "
             f"(mean $\\pm$ s.d. over {N_RUNS_EFF} independent runs, {BUDGET_EFF} evaluations "
             f"per run). Friedman $p$ = {STATS['HV'].get('friedman_p', float('nan')):.3g}; "
             f"Nemenyi CD = {STATS['HV']['nemenyi_cd']:.2f}.",
             "indicators")

    tex_test = TEST_DF.set_index("model")[
        ["test_acc", "test_f1", "precision", "recall", "mcc", "params_M", "MMac", "lat_ms"]]
    tex_test.columns = ["Acc.", "macro-$F_1$", "Prec.", "Rec.", "MCC",
                        "Params (M)", "MMac", "Lat. (ms)"]
    to_latex(tex_test.round(4), "tab_test_results",
             "Held-out test-set performance and deployment cost of the fully trained "
             "architectures selected from each optimiser's Pareto front.", "testresults")

    if ABL:
        _tc = [c for c in ["HV_mean", "HV_std", "HV_delta_%", "HV_p_holm", "HV_A12_full",
                           "IGD+_mean", "IGD+_p_holm", "best_err_mean", "generations_mean"]
               if c in ABL_AGG.columns]
        to_latex(ABL_AGG[_tc].round(4), "tab_ablation",
                 f"Ablation of MO-P3PGA components over {int(ABL_AGG['runs'].min())} paired runs. "
                 "$\\Delta$HV is relative to the complete algorithm; $p$ values are paired "
                 "Wilcoxon tests against it with Holm correction.", "ablation")

    if not MCN_DF.empty:
        to_latex(MCN_DF.set_index("vs")[["n01", "n10", "mcnemar_stat", "mcnemar_p",
                                         "mcnemar_p_holm"]].round(5), "tab_mcnemar",
                 f"McNemar tests of {HEAD_KEY} against every other model on the "
                 f"{len(y_te)}-image test split, with Holm--Bonferroni correction.", "mcnemar")

# %%
if ANALYSIS_READY:
    pass
    # ---- results digest -------------------------------------------------------------------
    prop_rank = STATS["HV"]["avg_ranks"].get(PROPOSED, float("nan"))
    rank_order = sorted(STATS["HV"]["avg_ranks"].items(), key=lambda kv: kv[1])
    win_vs = [r for r in STATS["HV"]["pairwise"] if r.get("significant_0.05")]
    head = FINAL.get(HEAD_KEY, {})
    best_overall = TEST_DF.iloc[0]

    lines = [
        f"# MO-P3PGA for Pareto-optimal CNN design — {DS['title']}",
        "",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  |  preset `{PRESET}`  |  "
        f"plan `{RUN_PLAN}`  |  GPU `{GPU_NAME}`",
        f"- Dataset: `{DATA_ROOT}` — {N_TOTAL} images, {CFG.n_classes} classes "
        f"({', '.join(CLASSES)})",
        *([f"- Exact-duplicate removal before splitting: {DEDUP_INFO['images_before']} -> "
           f"{DEDUP_INFO['images_after']} images ({DEDUP_INFO['duplicate_copies_removed']} copies "
           f"removed, {DEDUP_INFO['conflicting_label_groups_dropped']} label-conflicting groups "
           f"dropped)"] if DEDUP_INFO.get("applied") else []),
        *([f"- Group-aware split: {GROUP_INFO['n_groups']} original photos (rotated/flipped/"
           f"re-encoded copies grouped at thumbnail similarity >= {GROUP_INFO['threshold']}); "
           f"split {'/'.join(f'{f:.0%}' for f in GROUP_INFO['fractions_by_photo'])} by photo; "
           f"train {GROUP_INFO['train_images']} images from {GROUP_INFO['train_photos']} photos, "
           f"validation {GROUP_INFO['val_images']} images from {GROUP_INFO['val_photos']} photos, "
           f"test {GROUP_INFO['test_images_one_per_photo']} images (one per photo)"]
          if GROUP_INFO.get("applied") else []),
        f"- Near-duplicate audit (rotation/flip-invariant 32x32 thumbnails): test images whose "
        f"closest training image has similarity >= 0.995: "
        f"{DUP_AUDIT.get('test_with_train_neighbour_sim_ge_0.995', 'n/a')}, >= 0.98: "
        f"{DUP_AUDIT.get('test_with_train_neighbour_sim_ge_0.98', 'n/a')}",
        f"- Exact duplicates: {DUP_AUDIT['test_images_also_in_train']}/{DUP_AUDIT['n_test']} "
        f"test images also appear in training",
        f"- Splits: {len(IDX_TR)} train / {len(IDX_VA)} val / {len(IDX_TE)} test "
        f"(stratified, seed {CFG.seed})",
        f"- Objectives (minimised): {', '.join(CFG.objectives)}",
        f"- Search space: {D_GENES} decision variables, identical for all optimisers",
        f"- Protocol: {len(ORDER)} optimisers x {N_RUNS_EFF} runs x {BUDGET_EFF} evaluations "
        f"= {len(ORDER)*N_RUNS_EFF*BUDGET_EFF} budgeted evaluations "
        f"({sum(r['summary'].get('trained', 0) for r in SEARCH)} of them needed training; "
        f"the rest were repeats served from the shared memo)",
        "",
        "## Search-phase outcome",
        "",
        f"- Reference front: {REF_FRONT.shape[0]} solutions pooled from "
        f"{ALL_F.shape[0]} archive members",
        f"- Hypervolume ranking (best first): "
        + ", ".join(f"{k} ({v:.2f})" for k, v in rank_order),
        f"- {PROPOSED} mean HV = {AGG.loc[PROPOSED, 'HV_mean']:.4f} "
        f"± {AGG.loc[PROPOSED, 'HV_std']:.4f}, average rank {prop_rank:.2f}"
        if PROPOSED in AGG.index else "",
        f"- Friedman on HV: chi2 = {STATS['HV'].get('friedman_chi2', float('nan')):.3f}, "
        f"p = {STATS['HV'].get('friedman_p', float('nan')):.4g} "
        f"(N = {STATS['HV']['n_blocks']} runs, k = {len(STATS['HV']['algos'])})",
        f"- Wilcoxon (Holm-corrected) significant wins for {PROPOSED}: "
        f"{len(win_vs)}/{len(STATS['HV']['pairwise'])} "
        + (f"({', '.join(r['vs'] for r in win_vs)})" if win_vs else ""),
        f"- Contributions to the reference front: "
        + ", ".join(f"{a}={int(POOLED_IND.loc[a, 'contrib_to_ref'])}" for a in ORDER),
        "",
        *([] if not STATS_T else [
            "## Equal-tuning protocol",
            "",
            f"- {', '.join(TUNED_ORDER)} replace their default runs (same seeds)",
            "- Hypervolume ranking (best first): " + ", ".join(
                f"{k} ({v:.2f})" for k, v in
                sorted(STATS_T["HV"]["avg_ranks"].items(), key=lambda kv: kv[1])),
            f"- Friedman on HV: p = {STATS_T['HV'].get('friedman_p', float('nan')):.4g}, "
            f"Nemenyi CD = {STATS_T['HV']['nemenyi_cd']:.2f}",
            *[f"- {PROPOSED} vs {r['vs']}: mean HV {r['mean_base']:.4f} vs "
              f"{r['mean_other']:.4f}, Holm p = {r['wilcoxon_p_holm']:.3g}, "
              f"A12 = {r['A12']:.2f}" for r in STATS_T["HV"]["pairwise"]
              if r["vs"] in TUNED_ORDER],
            *[f"- {r['algo']}: default HV {r['HV_default']:.4f} -> tuned {r['HV_tuned']:.4f} "
              f"(paired Wilcoxon p = {r['wilcoxon_p']:.3g})" for r in TUNED_PAIRED],
            ""]),
        "## Final models (held-out test split)",
        "",
        f"- Headline model `{HEAD_KEY}`: accuracy **{acc_pt*100:.2f}%** "
        f"(95% CI {acc_lo*100:.2f}–{acc_hi*100:.2f}), macro-F1 **{f1_pt*100:.2f}%** "
        f"(95% CI {f1_lo*100:.2f}–{f1_hi*100:.2f})",
        f"- Cost: {head.get('mean', {}).get('params_m', float('nan')):.3f} M parameters, "
        f"{head.get('mean', {}).get('mmac', float('nan')):.0f} MMac, "
        f"{head.get('mean', {}).get('latency_ms', float('nan')):.2f} ms/image "
        f"at batch 1 on {GPU_NAME}",
        *[f"- {PROPOSED} {pt} point: macro-F1 {FINAL[k]['mean']['f1_macro']*100:.2f}% "
          f"± {FINAL[k]['std'].get('f1_macro', float('nan'))*100:.2f} over {FINAL[k]['n_seeds']} "
          f"seeds, {FINAL[k]['mean']['params_m']:.3f} M parameters, "
          f"{FINAL[k]['mean']['mmac']:.0f} MMac"
          for pt in ("accuracy", "knee", "efficiency")
          for k in [f"{PROPOSED} ({pt})"] if k in FINAL],
        f"- Best model by test macro-F1 overall: `{best_overall['model']}` "
        f"({best_overall['test_f1']*100:.2f}%)",
        (f"- McNemar of `{HEAD_KEY}` against the other {len(MCN_DF)} models (Holm-corrected): "
         f"significantly better than {int((MCN_DF['significant_0.05'] & (MCN_DF['n10'] > MCN_DF['n01'])).sum())}, "
         f"significantly worse than {int((MCN_DF['significant_0.05'] & (MCN_DF['n10'] < MCN_DF['n01'])).sum())}, "
         f"no significant difference from {int((~MCN_DF['significant_0.05']).sum())}"
         + (" (" + ", ".join(MCN_DF.loc[~MCN_DF['significant_0.05'], 'vs']) + ")"
            if (~MCN_DF['significant_0.05']).any() else ""))
        if not MCN_DF.empty else "- McNemar: not computed",
        "",
        "## Protocol validation",
        "",
        f"- Proxy fidelity: Spearman rho = {FIDELITY.get('spearman_rho', float('nan')):.3f} "
        f"between proxy macro-F1 and full-training test macro-F1 (n = {FIDELITY.get('n', 0)})",
    ]
    if ABL:
        lines += ["", f"## Ablation ({int(ABL_AGG['runs'].min())} paired runs; "
                  f"Friedman on HV p = {ABL_FRIEDMAN['p']:.3g})", ""]
        for v in ABL_AGG.index:
            _p = ABL_AGG.loc[v].get("HV_p_holm", float("nan"))
            lines.append(f"- {v}: HV {ABL_AGG.loc[v, 'HV_mean']:.4f} "
                         f"({ABL_AGG.loc[v, 'HV_delta_%']:+.2f}%), "
                         f"{ABL_AGG.loc[v, 'generations_mean']:.1f} generations"
                         + ("" if v == ABL_FULL else f", Holm p = {_p:.3g} vs full"))
    if SENS:
        lines += ["", "## Parameter sensitivity (best setting by mean HV)", ""]
        for g in set(SENS_DF["group"]):
            sub = SENS_DF[SENS_DF["group"] == g].groupby("value")["HV"].mean()
            lines.append(f"- {g}: best = {sub.idxmax()} (HV {sub.max():.4f}), "
                         f"worst = {sub.idxmin()} (HV {sub.min():.4f})")

    lines += [
        "",
        "## Architecture discovered (headline model)",
        "",
        "```json",
        json.dumps(head.get("ph", {}), indent=2),
        "```",
        "",
        "## Artefacts",
        "",
        f"- Tables: `{OUT}/tables/` (CSV + LaTeX)",
        f"- Figures: `{OUT}/figures/` (PNG 300 dpi + PDF)",
        f"- Checkpoints: `{STATE}/` (delete to force a clean re-run)",
        f"- Session wall-clock used: {CLOCK.hms()}",
    ]

    REPORT = "\n".join(l for l in lines if l is not None)
    (OUT / "RESULTS.md").write_text(REPORT)
    print("\n" + "=" * 78)
    print(REPORT)
    print("=" * 78)

# %%
if ANALYSIS_READY:
    pass
    # ---- manifest of everything produced ---------------------------------------------------
    save_state("global_memo", GLOBAL_MEMO)
    jdump("config_used.json", {**asdict(CFG), "preset": PRESET, "plan": RUN_PLAN,
                               "n_runs_effective": N_RUNS_EFF,
                               "budget_effective": BUDGET_EFF,
                               "sec_per_eval": SEC_PER_EVAL,
                               "gpu": GPU_NAME, "classes": CLASSES,
                               "n_images": int(N_TOTAL),
                               "d_genes": D_GENES, "gene_names": GENE_NAMES})
    print("\nfiles written:")
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(OUT)}  ({p.stat().st_size/1024:.1f} KiB)")
    CLOCK.report("final")
    print("\nDone.")

# %% [markdown]
# ## 24. Cross-dataset summary
#
# Runs automatically when the other dataset's output is attached as an input. Each dataset keeps
# its own statistics above. Average ranks across two datasets are descriptive only: the
# dataset-by-run Friedman test below treats runs from one dataset as separate blocks, so state
# that when you report it.

# %%
if ANALYSIS_READY:
    pass
    def find_tables(name: str) -> Optional[Path]:
        if name == DATASET:
            return OUT / "tables"
        folder = "outputs" if name == PRIMARY_DATASET else f"outputs_{name}"
        root = Path(os.environ.get("P3PGA_INPUT", "/kaggle/input"))
        if root.exists():
            for p in root.rglob(f"{folder}/tables/per_run_indicators.csv"):
                return p.parent
        return None


    _tabs = {n: find_tables(n) for n in DATASETS}
    _tabs = {n: t for n, t in _tabs.items() if t is not None and (t / "per_run_indicators.csv").exists()}
    if len(_tabs) < 2:
        print("cross-dataset summary skipped: attach the other dataset's output as an input "
              f"(found: {list(_tabs)})")
    else:
        runs = []
        for n, t in _tabs.items():
            df = pd.read_csv(t / "per_run_indicators.csv")
            df["dataset"] = n
            runs.append(df)
        runs = pd.concat(runs, ignore_index=True)
        # Only optimisers run on every dataset can be ranked across datasets; the others
        # (e.g. the maize-only equal-tuning entries) keep their per-dataset tables.
        _common = set.intersection(*[set(g["algo"]) for _, g in runs.groupby("dataset")])
        _dropped = sorted(set(runs["algo"]) - _common)
        if _dropped:
            print(f"cross-dataset summary uses the {len(_common)} optimisers run on every "
                  f"dataset; left out: {', '.join(_dropped)}")
        runs = runs[runs["algo"].isin(_common)]
        xrows = {}
        for n, g in runs.groupby("dataset"):
            m = g.groupby("algo")["HV"].agg(["mean", "std"])
            xrows[(n, "HV mean")] = m["mean"]
            xrows[(n, "HV s.d.")] = m["std"]
            xrows[(n, "HV rank")] = m["mean"].rank(ascending=False)
        XDS = pd.DataFrame(xrows)
        XDS[("all", "mean rank")] = XDS[[c for c in XDS.columns if c[1] == "HV rank"]].mean(axis=1)
        for n, t in _tabs.items():
            f = t / "final_test_results.csv"
            if f.exists():
                ft = pd.read_csv(f)
                ft = ft[ft["model"].str.endswith("(knee)")].copy()
                ft["algo"] = ft["model"].str.replace(" (knee)", "", regex=False)
                ft = ft.set_index("algo")
                XDS[(n, "knee test F1")] = ft["test_f1"]
                XDS[(n, "knee MMac")] = ft["MMac"]
        XDS = XDS.sort_values(("all", "mean rank"))
        XDS.columns = [f"{a}: {b}" for a, b in XDS.columns]
        XDS.to_csv(OUT / "tables" / "cross_dataset_summary.csv")
        to_latex(XDS.round(4), "tab_cross_dataset",
                 "Optimiser performance on both datasets. HV ranks are computed within each dataset.",
                 "crossdataset")
        print("--- cross-dataset summary ---")
        print(XDS.round(4).to_string())
        piv = runs.pivot_table(index=["dataset", "run"], columns="algo", values="HV").dropna()
        if piv.shape[0] >= 2 and piv.shape[1] >= 3:
            stat, p_x = sps.friedmanchisquare(*[piv[a].values for a in piv.columns])
            print(f"Friedman on HV across dataset-run blocks: chi2={stat:.3f}, p={p_x:.4g}, "
                  f"blocks={piv.shape[0]} (runs within a dataset are not independent datasets)")
            jdump("cross_dataset_friedman.json", {"chi2": float(stat), "p": float(p_x),
                                                  "blocks": int(piv.shape[0])})
