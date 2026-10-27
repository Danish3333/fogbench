# SELF-CHECK: verify every paper number yourself

Every claim below maps to a file in THIS repository and, where possible, a
one-command check. Paths are relative to the repository root.

## 1. Headline nested results (paper abstract + IV-A)

FILE: artifacts/nested_v43/defog_phase1_repeats.csv (10 rows, one per repeat)
COLUMNS: main_map (mAP), sh_ap, event_f1, farh, brier, ece,
reliability_diagram_slope, window_f1, sens_at_farh10, farh_at_sens85

    python verification/nested_primary_stats.py

Expected output matches the paper: mAP 0.2153 +- 0.0181 | SH AP 0.0286 +-
0.0115 | event F1 0.1803 +- 0.0824 | FA/h 4.31 +- 6.06 | Brier 0.0740 |
ECE 0.0319 | reliability slope 0.6821. The window_auc check prints SKIPPED
until the Zenodo .npz arrays are placed in artifacts/nested_v43/.

## 2. Protocol points (Methods III-F, Table 4)

- 30 held-out patients per repeat, 97 unique over 10 repeats:
  count test_subjects across artifacts/nested_v43/defog_fold1_r*_manifest.json
- fine-tuning pool 23-29 patients:
  count train_subjects in the same manifests
- development side 70, calib 15 / thresh 15 (outer rare-split):
  grep "rare-split" artifacts/nested_v43/gate_v43.log  (lines show
  n_train_subjects 70, n_calib_subjects 15, n_thresh_subjects 15)
- lab pre-training 41-47 of 62 (624-731 sequences):
  grep "warm-excl" artifacts/nested_v43/gate_v43.log
- inner carve 16-20 / 4-5 / 3-4:
  grep "[rare-split] {'n_train_subjects': 1" artifacts/nested_v43/gate_v43_attempt1.log
- SH under-powered warnings (selection side 1-2 subjects):
  grep "SH UNDER POWERED" artifacts/nested_v43/gate_v43.log
- seed sequence 42, 10042, ..., 90042:
  split_seed field in each manifest
- frozen thresholds / deployed release ratios (0.001-0.40 in 9 of 10):
  postproc.trigger / postproc.sustain in the 10 checkpoint JSONs

## 3. Broader-exposure arm (companion, IV-A and IV-E)

- per-repeat mAP mean 0.5786 +- 0.0611 -> the paper's 0.579 +- 0.064:
  artifacts/primary_v32/defog_fold1_r*_manifest.json are the splits;
  the arm's repeats table ships on Zenodo with the arrays
- external frozen AUCs behind Table 5 (gated per-repeat):
  grep "gate cost" artifacts/primary_v32/gate_v32.log  (40 lines, 4 cohorts x 10)
  Daphnet gated mean 0.887 (0.837-0.907), Xuanwu 0.858 (0.846-0.871),
  WearGait 0.786 (0.773-0.807), FoG-STAR 0.791 (0.778-0.804)
- warm-start-removed arm external AUCs (0.917/0.847/0.755/0.783):
  Zenodo _ablation_v33 logs; quoted in Section IV-E(ii)
- 3 never-scored patients:
  artifacts/primary_v32/never_validated_subjects.json

## 4. Published analysis outputs (results/tables/)

- family_split.json      lab/home split: SH AP 0.865/0.001 (broader),
                              0.055/0.004 (nested); %TF ICC 0.902/0.697 and
                              0.748/0.650; 60 lab + 37 home patients
- review_workload.json   22.02 FA/h, 2.36 min/h, median FA 3.08 s,
                              duration strata 0.188 / 0.492 / 0.637,
                              5,549 alarms vs 8,186 episodes
- gate_rate_grid.json    rate-matched gate comparison (OFF dominates)
- matched_merge.json     matched-merge conventions: eF1 -0.007, FA/h -2.4
- chunk_boundary.json    chunk-stitching test: eF1 0.380 -> 0.012
- bench_components_v38.json   component parameters sum 2,957,383; 10.68 GMACs
- pooled_oof_calibration_*    calibration streams behind Section III-H
- claim_verification.json claim-verifier output

## 5. External per-cohort event F1 + FA/h (IV-B headline)

FILE: grep "external]" artifacts/nested_v43/gate_v43.log
  daphnet: eventF1=0.363 FARh=15.86/h
  xuanwu:  eventF1=0.488 FARh=9.36/h
  fogstar: eventF1=0.081 FARh=7.48/h
  weargait:eventF1=0.177 FARh=8.79/h

## 6. What is NOT in this repository (Zenodo)

Per-repeat prediction arrays (10 x ~161 MB .npz), model weights (.ckpt),
ablation-arm run directories. Once placed under artifacts/, these unlock:
the window_auc check in nested_primary_stats.py, and the re-posting
analyses whose outputs ship in results/tables/ (family_split,
review_workload, matched_merge, chunk_boundary, gate_rate_grid).

## 7. Cross-dataset training-design sweep (Section IV-D)

- per-run launch logs: artifacts/design_sweep/gate_gen_*.log (22 runs,
  one per trained cell; each contains the run env, split, and gate lines)
- per-run results: artifacts/design_sweep/results_gen_*.json (AUC, eF1, FA/h
  per cell; the source of every number in the design table)
- aggregated table: artifacts/design_sweep/generalization_table_28cr.json/.md
- budget-restoration check: artifacts/design_sweep/budget_restoration_28dp.json
