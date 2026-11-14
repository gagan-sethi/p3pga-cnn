import sys; sys.argv.append("--cv")
import time
t=time.time()
from harness import *
print("loaded", round(time.time()-t), "s; memo", len(MEMO), "features", Xf.shape)
# A: replay real runs
real = {(r["label"], r["run"]): ind(r["archive_F"])[0] for r in SEARCH}
for algo in ["MO-P3PGA", "GDE3", "MOGWO", "NSGA-II"]:
    for k in (0, 1):
        o = run(algo, k)
        print(f"replay {algo:9s} run {k}: sur HV={o['HV']:.4f} real HV={real[(algo,k)]:.4f} predicted={o['pred_frac']:.2f} gen={o['gen']}")
