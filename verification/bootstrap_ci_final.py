# bootstrap_ci_final.py -- Round 28bb. Publication-grade bootstrap CIs on the
# FROZEN final-run arrays (audit D5: the launcher's n_boot=50 is development
# screening only; the function's own docstring asks >=5000 for publication).
# Runs TWO bootstraps on the defog member's pooled OOF (r1+r2+r3 concat,
# subject-level resampling via groups):
#   1) mAP CI            -> bootstrap_kaggle_map_ci(n_boot=1000)
#   2) main-vs-baseline diff CI -> kaggle_map_ci_diff(n_boot=1000)
# n_boot=1000 chosen for runtime (~hours); MC error on 95% percentile
# endpoints is small relative to the CI width. Env NEUROMAMBA_BOOTSTRAP_N is
# forced to 1000 INSIDE this script (the functions read it from the env).
#
# Usage (from "v26 final"; CPU, hours -- nohup it):
#   nohup python ../v25/bootstrap_ci_final.py > bootstrap_ci_run.log 2>&1 &
import os, sys, json
os.environ["NEUROMAMBA_BOOTSTRAP_N"] = "1000"
os.environ.setdefault("NEUROMAMBA_TLVMC_ROOT",
                      r"D:\Installed\Datasets\tlvmc-parkinsons-freezing-gait-prediction")
os.environ.setdefault("NEUROMAMBA_SUB_HEAD_MODE", "sigmoid")
os.environ.setdefault("NEUROMAMBA_WINNER_TOKEN_LOSS", "1")
import numpy as np

BASE = os.getcwd()  # run from "v26 final"


def concat(names, groups_fallback=None):
    P, L, O, G = [], [], [], []
    for i, name in enumerate(names):
        z = np.load(name, allow_pickle=True)
        P.append(np.asarray(z["sub_probs"], dtype=np.float32))
        L.append(np.asarray(z["sub_labels"], dtype=np.float32))
        O.append(np.asarray(z["sub_observed"], dtype=bool))
        if "sub_ids" in z.files:
            G.append(np.asarray([f"r{i + 1}:{s}" for s in z["sub_ids"]]))
        elif groups_fallback is not None:
            G.append(groups_fallback[i])  # frame-aligned fallback (baseline sidecars)
        else:
            raise KeyError(f"{name}: no sub_ids and no groups_fallback")
        del z
    return (np.concatenate(P), np.concatenate(L), np.concatenate(O),
            G)  # G returned as the per-repeat LIST (callers concatenate)


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import neuromamba_v25_sota as nm

    main_files = [f"defog_fold1_r{i}_ens.npz" for i in (1, 2, 3)]
    base_files = [f"defog_fold1_r{i}_baseline_ens.npz" for i in (1, 2, 3)]
    P, L, O, G_list = concat(main_files)
    G = np.concatenate(G_list)
    print(f"[boot] pooled main OOF: {len(P)} frames, {len(np.unique(G))} subject-records", flush=True)

    print("[boot] (1/2) mAP CI, 1000 subject-level boots...", flush=True)
    ci = nm.bootstrap_kaggle_map_ci(P, L, O, n_boot=1000, seed=42, groups=G)
    print(f"[boot] mAP CI = {ci}", flush=True)

    PB = None
    if all(os.path.exists(os.path.join(BASE, f)) for f in base_files):
        # baseline sidecars store val_subjects, not per-frame sub_ids; the arrays
        # are frame-aligned with the main OOF, so reuse the main groups
        PB, _, _, _GB_list = concat(base_files, groups_fallback=G_list)
        n = min(len(PB), len(P))
        print(f"[boot] (2/2) main-vs-baseline diff CI, 1000 boots...", flush=True)
        cid = nm.kaggle_map_ci_diff(P[:n], PB[:n], L[:n], O[:n], n_boot=1000, seed=42,
                                    groups=G[:n])
        print(f"[boot] diff CI = {cid}", flush=True)
    else:
        cid = {"status": "baseline arrays missing"}

    out = {"n_boot": 1000, "mAP_ci": ci, "baseline_diff_ci": cid,
           "note": "publication-grade subject-level bootstrap on frozen final-run "
                   "arrays (audit D5 fix); replaces the launcher's n_boot=50 CIs"}
    with open(os.path.join(BASE, "bootstrap_ci_final.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[done] wrote bootstrap_ci_final.json", flush=True)


if __name__ == "__main__":
    main()
