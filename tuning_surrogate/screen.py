import sys, time, itertools, pickle, random
import multiprocessing as mp
from harness import *
for m in (M_ERR, M_FL, M_PA): m.n_jobs = 1
TUNE_RUNS = list(range(20, 30))        # tuning seeds: never used by the real experiment

def job(a):
    algo, r, ov, lab = a
    o = run(algo, r, overrides=ov, label=lab, seed_label=(algo if lab != algo else None))
    o.pop("F"); o["cfg"] = lab; return o

if __name__ == "__main__":
    jobs = []
    for algo in OPT:
        if algo == "MO-3PGA": continue
        for r in TUNE_RUNS: jobs.append((algo, r, None, algo))
    space = {"p3pga_np": [1, 2, 4, 8], "mito_scale": [0.05, 0.1, 0.2222, 0.4],
             "p_attract": [0.0, 0.05, 0.1, 0.2], "attract_every": [1, 2, 4],
             "crossover_op": ["sbx", "npoint"], "selection_op": ["tournament", "sus_rank"]}
    allc = list(itertools.product(*space.values()))
    random.Random(0).shuffle(allc)
    cfgs = [dict(zip(space, c)) for c in allc[:70]]
    for c in cfgs:
        c["p3pga_n"] = 24 // c["p3pga_np"]; c["use_attract"] = c["p_attract"] > 0
        lab = "P3|" + ",".join(f"{k}={v}" for k, v in c.items() if k not in ("p3pga_n", "use_attract"))
        for r in TUNE_RUNS: jobs.append(("MO-P3PGA", r, c, lab))
    t = time.time()
    with mp.get_context("fork").Pool(mp.cpu_count()) as p:
        out = p.map(job, jobs, chunksize=4)
    pickle.dump(out, open("screen1.pkl", "wb"))
    print("done", len(out), "runs in", round(time.time() - t), "s")
