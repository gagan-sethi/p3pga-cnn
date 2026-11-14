"""Stage A: equal-effort tuning of GDE3 and MOGWO on the tuning seeds 20-29.
Stage B: every finalist (and every default) re-tested on 20 fresh seeds 30-49."""
import sys, time, itertools, pickle
import multiprocessing as mp
import pandas as pd
from harness import *
for m in (M_ERR, M_FL, M_PA): m.n_jobs = 1
TUNE, VALID = list(range(20, 30)), list(range(30, 50))

def job(a):
    algo, r, ov, lab = a
    o = run(algo, r, overrides=ov, label=lab, seed_label=algo)
    o.pop("F"); o["cfg"] = lab; o["algo"] = algo; return o

def pool(jobs):
    with mp.get_context("fork").Pool(mp.cpu_count()) as p:
        return p.map(job, jobs, chunksize=3)

def lab(algo, c):
    return algo + ("|" + ",".join(f"{k}={v}" for k, v in c.items()) if c else "")

if __name__ == "__main__":
    t = time.time()
    grids = {"GDE3": [{"gde3_F": f, "gde3_CR": cr} for f in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
                      for cr in (0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0)][:70],
             "MOGWO": [{"mogwo_a0": a, "mogwo_mut": m} for a in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
                       for m in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)]}
    A = pool([(alg, r, c, lab(alg, c)) for alg, cs in grids.items() for c in cs for r in TUNE])
    pickle.dump(A, open("screen2A.pkl", "wb"))
    dA = pd.DataFrame(A).groupby(["algo", "cfg"]).HV.mean().reset_index()
    print("stage A done", round(time.time() - t), "s")
    fin = []
    for alg in grids:
        top = dA[dA.algo == alg].sort_values("HV", ascending=False).head(3)
        print(top.to_string())
        for c in top.cfg:
            ov = dict(kv.split("=") for kv in c.split("|")[1].split(","))
            fin.append((alg, {k: float(v) for k, v in ov.items()}, c))
    s1 = pd.DataFrame(pickle.load(open("screen1.pkl", "rb")))
    p3 = s1[s1.cfg.str.startswith("P3|")].groupby("cfg").HV.mean().sort_values(ascending=False)
    for c in p3.index[:8]:
        ov = {}
        for kv in c[3:].split(","):
            k, v = kv.split("=")
            ov[k] = v if k in ("crossover_op", "selection_op") else (int(v) if k in ("p3pga_np", "attract_every") else float(v))
        ov["p3pga_n"] = 24 // ov["p3pga_np"]; ov["use_attract"] = ov["p_attract"] > 0
        fin.append(("MO-P3PGA", ov, c))
    for alg in OPT:
        if alg != "MO-3PGA":
            fin.append((alg, None, alg))
    B = pool([(alg, r, ov, c) for alg, ov, c in fin for r in VALID])
    pickle.dump(B, open("screen2B.pkl", "wb"))
    print("done", round(time.time() - t), "s")
