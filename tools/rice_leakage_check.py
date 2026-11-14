"""Sensitivity of the rice test results to residual near-duplicates.

Rebuilds the exact rice split of the notebook from the saved image cache (same code, same seed),
flags test photographs whose closest training image has a rotation/flip-invariant thumbnail
similarity >= 0.98, and recomputes every trained model's test accuracy and macro-F1 with and
without the flagged photographs.

    python3 tools/rice_leakage_check.py <folder with p3pga_state_rice/ and outputs_rice/>

The folder is the Kaggle output of the rice run (it holds the image cache and the trained models).
"""
import sys, re, json, pickle, hashlib
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
res = Path(sys.argv[1]).resolve()
st = res / "p3pga_state_rice"

src = (ROOT / "p3pga_mo_nas_maize.py").read_text()
def grab(start, end):
    return src[src.index(start):src.index(end)]
ns: Dict[str, Any] = {"np": np, "hashlib": hashlib, "Dict": Dict, "Any": Any, "List": List}
exec(grab("def stratified_split(", "GROUP_INFO: Dict[str, Any]"), ns)        # split + grouping helpers

meta = json.loads((st / "imgcache_176.json").read_text())
X = np.load(st / "imgcache_176.npy"); Y = np.asarray(meta["labels"], dtype=np.int64)
CLASSES = meta["classes"]

# exact-duplicate removal, as in the notebook
h = [hashlib.md5(X[i].tobytes()).hexdigest() for i in range(len(X))]
first, labels = {}, {}
for i, k in enumerate(h):
    first.setdefault(k, i); labels.setdefault(k, set()).add(int(Y[i]))
conflict = {k for k, v in labels.items() if len(v) > 1}
keep = np.array(sorted(i for k, i in first.items() if k not in conflict))
X, Y = X[keep], Y[keep]

G = ns["near_duplicate_groups"](X, Y, 0.995)
tr, va, te = ns["stratified_group_split"](Y, G, (0.60, 0.20, 0.20), 42, True)
print(f"split reproduced: train {len(tr)} | val {len(va)} | test {len(te)} | photos {len(np.unique(G))}")

# best train similarity of every test image, any rotation/flip, 32x32 thumbnails
T = ns["_thumbnails"]; U = ns["_unit_rows"]
np.seterr(all='ignore')
A = U(T(X, tr)); B = T(X, te)
best = np.full(len(te), -2.0); same = np.zeros(len(te), bool)
for r in range(4):
    for fl in (False, True):
        Bt = np.rot90(B, r, axes=(1, 2)); Bt = Bt[:, :, ::-1] if fl else Bt
        S = U(np.ascontiguousarray(Bt)) @ A.T
        a = S.argmax(1); v = S[np.arange(len(a)), a]
        upd = v > best; best[upd] = v[upd]; same[upd] = (Y[tr][a] == Y[te])[upd]
flag = best >= 0.98
print(f"test photos with a training neighbour >= 0.98: {flag.sum()} "
      f"(same class: {(flag & same).sum()}); per class: "
      + str({CLASSES[c]: int((flag & (Y[te] == c)).sum()) for c in range(len(CLASSES))}))

def accuracy_score(y, p):
    return float((y == p).mean())

def f1_score(y, p, average="macro"):
    """Macro-F1 over the classes present in y (as scikit-learn with zero_division=0)."""
    f = []
    for c in np.unique(np.concatenate([y, p])):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))
F = pickle.load(open(st / "final_results_fixed_batch.pkl", "rb"))
rows = []
for name, r in F.items():
    if not r.get("ok"):
        continue
    y = r["best"]["y"]; p = r["best"]["probs"].argmax(1)
    assert (y == Y[te]).all(), name
    k = ~flag
    rows.append((name, accuracy_score(y, p), f1_score(y, p, average="macro"),
                 accuracy_score(y[k], p[k]), f1_score(y[k], p[k], average="macro")))
rows.sort(key=lambda t: -t[2])
print(f"\n{'model (best seed)':34s} {'acc all':>8s} {'F1 all':>8s} {'acc clean':>10s} {'F1 clean':>9s}")
for n, a1, f1, a2, f2 in rows:
    print(f"{n:34s} {a1*100:8.2f} {f1*100:8.2f} {a2*100:10.2f} {f2*100:9.2f}")
json.dump({"n_test": int(len(te)), "flagged_ge_0.98": int(flag.sum()),
           "flagged_same_class": int((flag & same).sum()),
           "rows": [dict(model=n, acc_all=a1, f1_all=f1, acc_clean=a2, f1_clean=f2) for n, a1, f1, a2, f2 in rows]},
          open(res / "outputs_rice" / "residual_leakage_check.json", "w"), indent=1)
