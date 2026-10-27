import csv
import os
import statistics as st

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
rows = list(csv.DictReader(open(os.path.join(ROOT, "artifacts", "nested_v43", "defog_phase1_repeats.csv"),
                                encoding="utf-8")))


def col(k):
    return [float(r[k]) for r in rows if r.get(k) not in (None, "")]


def wilcoxon_exact_p(d):
    # exact two-sided signed-rank for n=10 via full enumeration
    n = len(d)
    d = [x for x in d if x != 0]
    n = len(d)
    ranks = sorted(range(n), key=lambda i: abs(d[i]))
    w = sum(i + 1 for i, idx in enumerate(ranks) if d[idx] > 0)
    from itertools import product
    total, ge = 0, 0
    for signs in product((0, 1), repeat=n):
        s = sum((i + 1) for i, sg in enumerate(signs) if sg)
        total += 1
        if s >= w:
            ge += 1
    return ge / total


m = col("main_map")
b = col("baseline_map")
d = col("delta")
ef1 = col("event_f1")
farh = col("farh")
sh = col("sh_ap")
brier = col("brier")
ece = col("ece")
rslope = col("reliability_diagram_slope")
lslope = col("logistic_calib_slope")
wf1 = col("window_f1")

mean_d = st.mean(d) if d else float("nan")
sd_d = st.stdev(d) if d else float("nan")
p = wilcoxon_exact_p(d) if d else float("nan")
dz = mean_d / sd_d if d else float("nan")
se = st.stdev(m) / (len(m) ** 0.5)
print(f"main_map      : {st.mean(m):.4f} +- {st.stdev(m):.4f}  range [{min(m):.3f},{max(m):.3f}]")
print(f"  t9 interval : [{st.mean(m) - 2.262 * se:.4f}, {st.mean(m) + 2.262 * se:.4f}]")
print(f"baseline_map  : {st.mean(b):.4f} +- {st.stdev(b):.4f}" if b else "baseline_map  : (not trained in nested arm)")
print(f"delta         : {mean_d:+.4f} +- {sd_d:.4f}, all positive: {all(x > 0 for x in d)}, "
      f"n_pos={sum(1 for x in d if x > 0)}/10, Wilcoxon p={p:.6f}, dz={dz:.2f}" if d else "delta         : (none)")
print(f"sh_ap         : {st.mean(sh):.4f} +- {st.stdev(sh):.4f}")
print(f"event_f1      : {st.mean(ef1):.4f} +- {st.stdev(ef1):.4f}")
print(f"farh          : {st.mean(farh):.2f} +- {st.stdev(farh):.2f}")
print(f"window_f1     : {st.mean(wf1):.4f} +- {st.stdev(wf1):.4f}")
print(f"brier         : {st.mean(brier):.4f} +- {st.stdev(brier):.4f}")
print(f"ece           : {st.mean(ece):.4f} +- {st.stdev(ece):.4f}")
print(f"reliab_slope  : {st.mean(rslope):.4f} +- {st.stdev(rslope):.4f}")
print(f"logistic_slope: {st.mean(lslope):.4f} +- {st.stdev(lslope):.4f}")

# window AUC per repeat from the npz streams
from sklearn.metrics import roc_auc_score

aucs = []
missing = [i for i in range(1, 11)
           if not os.path.exists(os.path.join(ROOT, "artifacts", "nested_v43", f"defog_fold1_r{i}_ens.npz"))]
if missing:
    print(f"window_auc    : SKIPPED - per-repeat prediction arrays absent for repeats {missing}")
    print("                (place the Zenodo .npz files in artifacts/nested_v43/ to run this check)")
else:
    from sklearn.metrics import roc_auc_score
    for i in range(1, 11):
        pth = os.path.join(ROOT, "artifacts", "nested_v43", f"defog_fold1_r{i}_ens.npz")
        d2 = np.load(pth, allow_pickle=True)
        pr = np.asarray(d2["fog_probs"], dtype=np.float32)
        la = (np.asarray(d2["fog_labels"], dtype=np.float32) > 0.5).astype(int)
        aucs.append(roc_auc_score(la, pr))
        del d2, pr, la
    print(f"window_auc    : {st.mean(aucs):.4f} +- {st.stdev(aucs):.4f} (per-repeat OOF)")
