# FoGBench: A Fully Participant-Nested Benchmark for Wearable Freezing-of-Gait Detection

Code, verification, and per-repeat artifacts for the paper *FoGBench: A Fully Participant-Nested Benchmark for Wearable Freezing-of-Gait Detection*.



The detector is a deliberately conventional CNN-BiGRU network (2,957,383 parameters; 13.2 ms per 120 s window on an RTX 3060) evaluated under a fully participant-nested protocol: 10 repeated subject-disjoint train/evaluate splits over the TLVMC corpus 100-patient supervised pool, with machine-asserted exclusion of evaluation patients from every training stage, plus a frozen four-cohort external evaluation with label-free prior matching.

Headline nested results (10 repeats, two-seed ensembles; see artifacts/nested_v43/defog_phase1_repeats.csv): mAP 0.215 +/- 0.018, StartHesitation AP 0.029 +/- 0.012, frame-level AUC 0.908 +/- 0.013, event F1 0.180 +/- 0.082 at 4.3 +/- 6.1 FA/h. An exposure audit on identical splits shows that admitting evaluation patients' laboratory recordings into pre-training raises mAP to 0.579 +/- 0.064 (paired delta -0.363, negative in all ten repeats).

## Repository layout

```
code/neuromamba_v25_sota.py   single-file detector, training, nested protocol,
                              leakage assertions, external evaluation ("NeuroMamba"
                              is a legacy codename, not an architecture)
verification/                 headline-number checker, component benchmark,
                              subject-level bootstrap
scripts/                      exact environment launchers for the released runs
artifacts/nested_v43/         primary arm: per-repeat manifests, checkpoint metadata,
                              repeats table, run logs
artifacts/primary_v32/        broader-exposure arm: manifests, never-scored list, log
artifacts/design_sweep/       per-run launch logs + result JSONs for all 22
                              single-seed training-design runs (Table: cross-dataset sweep)
results/tables/               published analysis outputs (JSON)
docs/                         Figure-1 <-> code map, artifact layout
```

Large binary artifacts (10 x ~161 MB per-repeat prediction arrays .npz, model weights .ckpt) are distributed via Zenodo (DOI placeholder, added upon acceptance); every JSON/CSV/log needed to check the paper's numbers is in this repository.

## Figure 1 <-> code map

Every block of the framework figure anchors to code/neuromamba_v25_sota.py (line numbers of this release): see docs/figure1_code_map.md for the full table. Summary:

| Figure 1 block | Code anchor | Line |
|---|---|---|
| 13-channel input | N_CHANNELS = int(env NEUROMAMBA_N_CHANNELS, default 13) | 136 |
| stem kernel 7 stride 1 | nn.Conv1d(in_channels, chs[0], 7, 1, 3) | 4294 |
| 4 residual blocks 64->128->128->256->256 | chs = [64, 128, 128, 256, 256], strides = [1, 2, 1, 2, 1] | 4292 |
| 256->128 projection + layer norm | nn.Linear(enc_out, hidden) + LayerNorm | 4495 |
| 2-layer BiGRU hidden 128 | nn.GRU(hidden, hidden, num_layers=n_gru_layers, bidirectional=True) | 4496 |
| local-window attention 4 heads, 128-step | LocalWindowAttention(head_in, window_size=128, n_heads=4) | 4502 |
| medication FiLM | NEUROMAMBA_USE_MED_FILM | grep |
| linear upsample | F.interpolate(h, size=T0, mode="linear") | 4233 |
| FoG + 3 subtype heads | fog_logits, sub_logits | 8544 |
| walking-gate CNN kernel 25 stride 4, 4->16->32->32 | Conv1d(n_phys, base_ch, 25, stride=4) | 4211 |
| event assembly confirm 3 / merge 30 / release 0.4x | causal_event_postprocess | 1892 |
| nested warm-start exclusion | NEUROMAMBA_WARM_EXCLUDE_VAL | 6183 |

The released configuration matches the executed runs (scripts/ops_nested_v43.sh); two documented deviations from library defaults are disclosed in the paper: the event-assembly release point can be stale relative to the final trigger in 9 of 10 released checkpoints (deployed ratios 0.001-0.40), and the merge gap took its default of 30 frames in every released fold.

## Verify the numbers yourself

See [SELF_CHECK.md](SELF_CHECK.md): every headline claim mapped to the file that
carries it and the one-command check that re-derives it.

## Environment

Python 3.11, PyTorch (CUDA), numpy, pandas, scikit-learn, scipy.

```
pip install -r requirements.txt
```

## Quickstart

```
# integrity check of the released program
python -m py_compile code/neuromamba_v25_sota.py

# nested 10-repeat run (exact released environment; WSL2 + single GPU)
bash scripts/ops_nested_v43.sh

# broader-exposure companion arm
bash scripts/ops_primary_v32.sh

# verify the headline nested numbers from the released repeats table
python verification/nested_primary_stats.py

# per-repeat external lines (AUC / event F1 / FA/h per cohort)
grep "external]" artifacts/nested_v43/gate_v43_nested.log
```

Verification scripts read the artifact layout under artifacts/ and the released per-repeat arrays (Zenodo); adjust BASE paths at the top of each script if your layout differs.

## Data

All evaluation cohorts are public: TLVMC (Tel Aviv Sourasky Medical Center ML competition corpus), Daphnet, Xuanwu (Zhang et al. 2022), WearGait-PD (Anderson et al. 2026), FoG-STAR (Borzi et al. 2026). Access routes and loaders are in the dataset configuration block of code/neuromamba_v25_sota.py.

## Notes

- "NeuroMamba" in identifiers and logs is a legacy project codename; the released model is the CNN-BiGRU described above and contains no state-space-model components.
- Several legacy training branches present in the code (e.g., a domain-adversarial head, LODO and contest-mimic tracks) are inactive in the configuration that produced the reported results; the paper discloses this.
- Manifest split_seed sequence: 42, 10042, ..., 90042 (base 42, +10,000 per repeat), matching the released manifests.

## License

MIT (see LICENSE) - please also cite the paper.