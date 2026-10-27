# Figure 1 <-> code map (detail)

Figure 1 (framework_v38.pdf) vs `code/neuromamba_v25_sota.py` (this release,
12,088 lines, compiles with Python 3.11). Anchor line numbers refer to this file.

| Claim (figure / paper III-B..III-D) | Code evidence | Line |
|---|---|---|
| 13-channel input | `N_CHANNELS = int(env NEUROMAMBA_N_CHANNELS, default "13")` | 136 |
| stem kernel 7, stride 1 | `nn.Conv1d(in_channels, chs[0], 7, 1, 3, bias=False)` | 4294 |
| residual widths 64->128->128->256->256 | `chs = [64, 128, 128, 256, 256]` | 4292 |
| stride-2 at 64->128 and 128->256 | `strides = [1, 2, 1, 2, 1]` | 4293 |
| 12,000 frames -> 3,000 steps of 40 ms | downsampling 4x from the strides above | -- |
| 256->128 projection + layer norm | `nn.Linear(self.enc_out, hidden), nn.LayerNorm(hidden)`, hidden = 128 | 4495 |
| 2-layer BiGRU, hidden 128/direction | `nn.GRU(hidden, hidden, num_layers=n_gru_layers, bidirectional=True)` | 4496 |
| local-window attention, 4 heads, 128-step | class LocalWindowAttention (window_size=128, n_heads=4) | 4077 / 4502 |
| medication FiLM | NEUROMAMBA_USE_MED_FILM (ON/OFF/unknown=0.5, single linear generator) | grep USE_MED_FILM |
| linear upsample | F.interpolate(h, size=T0, mode="linear") | 4233 |
| FoG logit + 3 subtype sigmoids | fog_logits, sub_logits | 8544 |
| gate CNN kernel 25 / stride 4 / 4->16->32->32 | Conv1d(n_phys, base_ch, 25, stride=4), class in_channels=4, base_ch=16 | 4211 / 4204 |
| gate output interpolated to frame rate | gate head F.interpolate(..., mode="linear") | grep gate |
| event assembly: confirm 3 / merge 30 / release 0.4x | causal_event_postprocess(..., confirm_n=3, cooldown_n=30, merge_gap_n=MERGE_GAP_N_DEFAULT); MERGE_GAP_N_DEFAULT = 30; release thr * 0.4 | 1892 |
| param totals 2,957,383 / 10.68 GMACs | verification/benchmark_components.py (torch required) | -- |
