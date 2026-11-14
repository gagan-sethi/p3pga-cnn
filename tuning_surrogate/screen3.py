"""MO-P3PGA-II: 70 configurations (= GDE3's tuning budget) on tuning seeds 20-29, then the top 5
on the same fresh seeds 30-49 used for tuned GDE3 / tuned MOGWO in screen2B."""
import time, itertools, pickle, random
import multiprocessing as mp
import pandas as pd
from harness import *
for m in (M_ERR, M_FL, M_PA): m.n_jobs = 1
TUNE, VALID = list(range(20, 30)), list(range(30, 50))

def job(a):
    r, ov, lab = a
    o = run("MO-P3PGA-II", r, overrides=ov, label=lab, seed_label="MO-P3PGA")
    o.pop("F"); o["cfg"] = lab; o["algo"] = "MO-P3PGA-II"; o["ov"] = ov; return o

def pool(jobs):
    with mp.get_context("fork").Pool(mp.cpu_count()) as p:
        return p.map(job, jobs, chunksize=3)

if __name__ == "__main__":
    t = time.time()
    space = []
    for npop, mode, frac, sched, pa, cx in itertools.product(
            (2, 3, 4, 6), ("uniform", "differential"), (0.25, 0.5, 0.75), ("const", "linear"),
            (0.05, 0.1, 0.2, 0.3), ("npoint", "sbx")):
        scales = (0.05, 0.1, 0.2222, 0.4) if mode == "uniform" else (0.3, 0.5, 0.8)
        thirds = ("island",) if mode == "uniform" else ("island", "global")
        for sc in scales:
            for th in thirds:
                space.append({"p3pga_np": npop, "p3pga_n": 24 // npop, "p2_mito_mode": mode,
                              "mito_scale": sc, "p2_third": th, "p2_mito_frac": frac,
                              "p2_delta_sched": sched, "p_attract": pa, "crossover_op": cx})
    rnd = random.Random(1)
    cfgs = [{"p3pga_np": 4, "p3pga_n": 6}] + rnd.sample(space, 69)      # default + 69 sampled
    lab = lambda c: "P3II|" + ",".join(f"{k}={v}" for k, v in c.items() if k != "p3pga_n")
    A = pool([(r, c, lab(c)) for c in cfgs for r in TUNE])
    pickle.dump(A, open("screen3A.pkl", "wb"))
    dA = pd.DataFrame(A).groupby("cfg").HV.mean().sort_values(ascending=False)
    print("stage A done", round(time.time() - t), "s"); print(dA.head(8).to_string(), flush=True)
    best = {o["cfg"]: o["ov"] for o in A}
    B = pool([(r, best[c], c) for c in dA.index[:5] for r in VALID])
    pickle.dump(B, open("screen3B.pkl", "wb"))
    print("done", round(time.time() - t), "s", flush=True)
