import time
from harness import *
for m in (M_ERR, M_FL, M_PA): m.n_jobs = 1
for ov in [{}, {"p2_mito_mode": "differential", "mito_scale": 0.5, "p2_third": "global", "p2_delta_sched": "linear", "crossover_op": "npoint"}]:
    t = time.time(); o = run("MO-P3PGA-II", 20, overrides=ov, seed_label="MO-P3PGA")
    print(ov, "HV=%.4f gen=%d %.0fs" % (o["HV"], o["gen"], time.time() - t))
