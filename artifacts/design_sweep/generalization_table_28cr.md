# Generalization suite (28cr) -- Fog-STAR removed, 1 seed / 1 repeat

event-F1 is printed **with its FARh**; rows are only comparable at a
matched alarm rate (within 0.5 FA/h). `auc` is threshold-free and
always comparable.

TLVMC test rows for the C/D arms were measured post-hoc by the chunked
memory-bounded evaluator (28dc/28dh: in-training TLVMC-as-external
terminated on this 15.9 GB host). Post-hoc rows use each run's own
stored frozen trigger; MLLS prior matching is not applied post-hoc.

| run | protocol | train | test cohort | eF1 | FARh (/h) | AUC | FARh p50/p95 |
|---|---|---|---|---|---|---|---|
| A1_holdD | A  LODO-1 | TLVMC+xuanwu+weargait | daphnet | 0.398 | 7.33 | 0.912 | 11.1/15.7 |
| A2_holdX | A  LODO-1 | TLVMC+daphnet+weargait | xuanwu | 0.600 | 17.68 | 0.845 | 0.0/29.0 |
| A3_holdW | A  LODO-1 | TLVMC+daphnet+xuanwu | weargait | 0.351 | 10.58 | 0.822 | 4.2/68.6 |
| B1_holdXW | B  LODO-2 | TLVMC+daphnet | weargait | 0.214 | 7.32 | 0.812 | 0.0/22.1 |
| B1_holdXW | B  LODO-2 | TLVMC+daphnet | xuanwu | 0.562 | 18.26 | 0.847 | 4.5/20.4 |
| B2_holdDW | B  LODO-2 | TLVMC+xuanwu | daphnet | 0.473 | 13.70 | 0.909 | 30.6/39.6 |
| B2_holdDW | B  LODO-2 | TLVMC+xuanwu | weargait | 0.327 | 12.20 | 0.826 | 0.0/58.8 |
| B3_holdDX | B  LODO-2 | TLVMC+weargait | daphnet | 0.382 | 5.29 | 0.906 | 8.9/12.5 |
| B3_holdDX | B  LODO-2 | TLVMC+weargait | xuanwu | 0.543 | 14.20 | 0.848 | 4.0/22.3 |
| C1_trainT | C  T1TA | TLVMC | daphnet | 0.121 | 1.20 | 0.912 | 1.1/6.0 |
| C1_trainT | C  T1TA | TLVMC | weargait | 0.041 | 0.81 | 0.791 | 0.0/1.9 |
| C1_trainT | C  T1TA | TLVMC | xuanwu | 0.575 | 4.93 | 0.882 | 0.0/4.7 |
| C2_trainD | C  T1TA | daphnet | TLVMC | 0.008 | 22.26 | 0.707 | 25.6/99.3 |
| C2_trainD | C  T1TA | daphnet | weargait | 0.135 | 3.25 | 0.443 | 0.0/0.0 |
| C2_trainD | C  T1TA | daphnet | xuanwu | 0.389 | 7.25 | 0.216 | 0.0/4.7 |
| C3_trainX | C  T1TA | xuanwu | TLVMC | 0.361 | 81.55 | 0.819 | 80.5/132.1 |
| C3_trainX | C  T1TA | xuanwu | daphnet | 0.102 | 21.74 | 0.878 | 12.9/20.1 |
| C3_trainX | C  T1TA | xuanwu | weargait | 0.135 | 3.25 | 0.647 | 0.0/1.9 |
| C4_trainW | C  T1TA | weargait | TLVMC | 0.182 | 109.67 | 0.574 | 112.1/192.2 |
| C4_trainW | C  T1TA | weargait | daphnet | 0.000 | 0.00 | 0.709 | 75.2/104.0 |
| C4_trainW | C  T1TA | weargait | xuanwu | 0.386 | 67.24 | 0.429 | 75.3/150.3 |
| D1_invTrainDXW | D  INVERSE | daphnet+xuanwu+weargait | TLVMC | 0.417 | 88.07 | 0.864 | 88.2/169.4 |
| D2_invTrainDX | D  INVERSE | daphnet+xuanwu | TLVMC | 0.268 | 4.05 | 0.873 | 0.6/29.2 |
| D2_invTrainDX | D  INVERSE | daphnet+xuanwu | weargait | 0.127 | 0.81 | 0.661 | 0.0/0.0 |
| D3_invTrainDW | D  INVERSE | daphnet+weargait | TLVMC | 0.236 | 25.43 | 0.820 | 7.0/68.4 |
| D3_invTrainDW | D  INVERSE | daphnet+weargait | xuanwu | 0.269 | 4.06 | 0.722 | 0.0/0.0 |
| D4_invTrainXW | D  INVERSE | xuanwu+weargait | TLVMC | 0.360 | 410.92 | 0.796 | 421.4/676.2 |
| D4_invTrainXW | D  INVERSE | xuanwu+weargait | daphnet | 0.104 | 20.90 | 0.746 | 25.8/33.4 |

## Post-hoc TLVMC budget points (best event F1 at FARh <= 10/h;
matched-budget convention of the deployment layer)

| run | trigger | eF1 | FARh (/h) |
|---|---|---|---|
| C2_trainD | 0.08 | 0.001 | 6.60 |
| C3_trainX | 0.96 | 0.309 | 7.95 |
| C4_trainW | 0.98 | 0.001 | 1.13 |
| D1_invTrainDXW | 0.80 | 0.096 | 9.55 |
| D2_invTrainDX | 0.28 | 0.512 | 9.24 |
| D3_invTrainDW | 0.56 | 0.124 | 7.98 |
| D4_invTrainXW | 0.28 | 0.263 | 9.26 |
