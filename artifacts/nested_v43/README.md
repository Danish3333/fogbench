# _nested_v43 -- READ ME (2026-10-25)

This directory holds the **true fully participant-nested primary arm**
(`ops_nested_v43.sh`, `NEUROMAMBA_WARM_EXCLUDE_VAL=1`): the repeat's
validation patients are removed from the laboratory warm-start pool and its
StartHesitation rehearsal (per-fold banner `[warm-excl v43]` in the run log;
e.g. repeat 1: 937 -> 637 sequences, 20 lab validation patients removed).

**Results (10 repeats, mean +/- SD, from the fold checkpoints):**

| Quantity | Nested (v43, this dir) | Broader exposure (v32) |
|---|---|---|
| Per-class mAP | **0.215 +/- 0.018** | 0.579 +/- 0.064 |
| StartHesitation AP | 0.029 +/- 0.012 | 0.824 +/- 0.067 |
| Turning / Walking AP | 0.550 / 0.068 | -- (pooled) |
| Window AUC | 0.908 +/- 0.013 | 0.928 (ckpt mean) |
| Deployed event F1 @ FA/h | 0.180 +/- 0.082 @ 4.3 +/- 6.1 | 0.443 @ 14.9 (r1) |
| Per-repeat %TF ICC | 0.475 +/- 0.161 | 0.81 (r1) |
| External AUC (D/X/W/F) | 0.884 / 0.869 / 0.795 / 0.788 | within 0.01 |
| Paired delta nested - broader | **-0.363** [-0.452, -0.265], 0/10 positive | -- |

**Reading:** under true participant nesting, StartHesitation is not
learnable; the previously reported warm-start benefit (mAP 0.58/0.58, SH AP
0.83) was evaluation-patient exposure through the warm pool and rehearsal.
The external ranking-transfer result (AUC 0.79-0.89) is unchanged.

Note: the first attempt of this run trained all 10 repeats and was killed
(SIGTERM) during the final pooled-calibration sweep; the relaunch with
`--resume` reused every fold (identity-matched REUSED_RESULT) and completed
the packaging. `v43_run.log` = attempt 1 (authoritative full log);
`v43_run2.log` = resume/packaging pass.
