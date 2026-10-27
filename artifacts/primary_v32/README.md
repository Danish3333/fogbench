# _primary_v32 -- READ ME (broader-exposure companion arm)

This directory holds the companion broader-exposure (sensitivity) arm of the
10-repeat subject-disjoint protocol. It is NOT the primary participant-nested
arm, despite the directory name: the paper's primary fully participant-nested
numbers come from artifacts/nested_v43/ (run v43, launched with
NEUROMAMBA_WARM_EXCLUDE_VAL=1).

Provenance note (deep-audit finding man-1, verified): this arm was never
participant-nested -- its logs show the full 937-sequence laboratory warm pool
in all 20 folds, so evaluation patients could influence the initialized
encoder. In the corrected paper it therefore serves only as the
broader-exposure companion: the exposure-carrying sensitivity arm used for the
paired exposure audit (mAP 0.579 vs nested 0.215; paired delta -0.363,
negative in all 10 repeats) and for every component and ablation comparison
(ResNet1D baseline 0.136 +/- 0.021 was trained in this arm's baseline_map
column).

Contents:
- defog_phase1_repeats.csv  per-repeat main_map / sh_ap / event_f1 / farh /
  baseline_map (the 0.579 +- 0.064 companion numbers and the 0.136 baseline)
- defog_fold1_r1..r10_manifest.json  split manifests (test_subjects 30 each,
  97 unique across repeats)
- never_validated_subjects.json  the 3 supervised patients never drawn for
  scoring (derived from this arm's manifests)
- gate_v32.log  full run log; its stored gated per-repeat external AUC lines
  are the source of the clinical-agreement table's AUC column and Fig 3(a)
