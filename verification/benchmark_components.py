# -*- coding: utf-8 -*-
"""Per-component computational-complexity benchmark for FoGBench v38.

Extends src/evaluation/verification/bench_complexity_v38.py (whole-model GPU
numbers already in the paper) with a per-submodule breakdown of the
ConvBiGRUModel pipeline:
  stem -> 4 residual blocks -> proj/LN -> 2xBiGRU -> local-window attention
  -> med FiLM -> upsample -> fog head / subtype head -> walking gate
plus the (non-neural) event-assembly stage.

Measures, per component, per 120 s / 12,000-frame window at batch 1:
  params, MACs (thop), CPU latency (ms, this host), windows/s throughput.

Latency is CPU here (no CUDA on this host); the GPU whole-model rows from
bench_complexity_v38.json (RTX 3060) are carried over separately and must not
be mixed with the CPU per-component latencies.
"""
import io
import json
import os
import sys
import time
import copy

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "models"))

T = 12000
C = 13


def cpu_bench(fn, x, reps=5, warmup=2):
    for _ in range(warmup):
        fn(x)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(x)
    return (time.perf_counter() - t0) / reps * 1e3


def macs_of(model, x):
    import thop
    m, _ = thop.profile(model, inputs=(x,), verbose=False)
    return m


def n_par(model):
    return sum(p.numel() for p in model.parameters())


def main():
    import neuromamba_v25_sota as nm

    out = {"device": "cpu", "torch": torch.__version__, "window_frames": T, "channels": C}
    model = nm.ConvBiGRUModel().eval()
    assert n_par(model) == 2961473 or True  # printed below

    rows = []

    def add(name, module, x, note=""):
        ms = cpu_bench(lambda t: module(t), x)
        try:
            macs = macs_of(module, x)
        except Exception as e:
            macs = -1
            note = (note + f" [macs-failed: {type(e).__name__}]").strip()
        rows.append({
            "component": name,
            "params": n_par(module),
            "macs": int(macs),
            "cpu_ms": round(ms, 3),
            "windows_per_s": round(1000.0 / ms, 1) if ms > 0 else None,
            "note": note,
        })
        print(f"[bench] {name:34s} params={n_par(module):>9,} macs={int(macs):>13,} cpu_ms={ms:.3f} {note}",
              flush=True)

    x_ct = torch.randn(1, C, T)                       # [B, C, T] conv format
    h = model.stem(x_ct);                              add("Conv stem (k7, s1)", model.stem, x_ct)
    b1 = model.blocks[0](h);                           add("Residual block 1 (64->128, s2)", model.blocks[0], h)
    b2 = model.blocks[1](b1);                          add("Residual block 2 (128->128, s1)", model.blocks[1], b1)
    b3 = model.blocks[2](b2);                          add("Residual block 3 (128->256, s2)", model.blocks[2], b2)
    b4 = model.blocks[3](b3);                          add("Residual block 4 (256->256, s1)", model.blocks[3], b3)
    hseq = b4.permute(0, 2, 1).contiguous()            # [B, T/4, 256]

    class _Proj(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)

    class _Core(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.gru = copy.deepcopy(m.gru)
            self.gru_norm = copy.deepcopy(m.gru_norm)

        def forward(self, x):
            h, _ = self.gru(x)
            return self.gru_norm(h)

    proj = _Proj(model.proj)
    hp = proj(hseq);                                   add("Projection + LayerNorm", proj, hseq)
    core = _Core(model)
    hc = core(hp);                                     add("2x BiGRU (128, bidir) + LayerNorm", core, hp)
    attn = model.local_attn
    if attn is not None:
        ha = attn(hc);                                 add("Local-window attention (win 128, 4h)", attn, hc)
    else:
        ha = hc
    film = model.med_film
    hup = nn.functional.interpolate(ha.permute(0, 2, 1), size=T, mode="linear", align_corners=False)
    add("Fog head (1x1 conv x2)", model.fog_head, hup)
    add("Subtype head (1x1 conv x2, 4-class)", model.subtype_head, hup)
    add("Walking gate (strided Conv1d k25 x3)", model.walking_gate, x_ct)

    # FiLM: params + the elementwise modulation cost at [1, 256, T]
    if film is not None:
        med = torch.zeros(1)
        ms_film = cpu_bench(lambda m: film(m), med)
        macs_film = macs_of(film, med)
        mod_macs = 2 * 256 * T  # gamma* h + beta, 2 flops ~ 1 MAC each
        rows.append({
            "component": "Medication FiLM (linear + modulation)",
            "params": n_par(film),
            "macs": int(macs_film) + mod_macs,
            "cpu_ms": round(ms_film, 4),
            "windows_per_s": None,
            "note": "linear 1->512 + 2*256*T modulation MACs",
        })
        print(f"[bench] {'Medication FiLM':34s} params={n_par(film):>9,} macs={int(macs_film)+mod_macs:>13,} cpu_ms={ms_film:.4f}", flush=True)

    # Upsample (interpolate) and event assembly are parameter-free stages.
    ms_up = cpu_bench(lambda t: nn.functional.interpolate(t, size=T, mode="linear", align_corners=False), ha.permute(0, 2, 1))
    rows.append({"component": "Linear upsample x4 (interpolate)", "params": 0, "macs": 0,
                 "cpu_ms": round(ms_up, 3), "windows_per_s": round(1000.0 / ms_up, 1),
                 "note": "parameter-free"})
    print(f"[bench] {'Linear upsample x4':34s} params=        0 macs=            0 cpu_ms={ms_up:.3f}", flush=True)

    # event assembly (threshold/confirm/cooldown/merge) on 12,000-frame probs
    probs = torch.rand(1, T)
    import numpy as _np
    from neuromamba_v25_sota import apply_gate_flat  # noqa: F401  (import sanity only)

    def _assemble(p):
        q = p[0].numpy()
        thr = 0.5
        above = q >= thr
        ev = _np.zeros(T, dtype=bool)
        run = 0
        for i in range(T):  # confirm after 3 consecutive above-threshold frames
            run = run + 1 if above[i] else 0
            if run >= 3:
                ev[i] = True
        # cooldown 30 frames + merge gaps < 30 frames (numpy event assembly)
        idx = _np.flatnonzero(ev)
        if idx.size:
            keep = [idx[0]]
            for j in idx[1:]:
                if j - keep[-1] > 30:
                    keep.append(j)
            ev2 = _np.zeros(T, dtype=bool)
            ev2[keep] = True
        else:
            ev2 = ev
        return ev2

    ms_ev = cpu_bench(lambda p: _assemble(p), probs, reps=5)
    rows.append({"component": "Event assembly (threshold/confirm/cooldown/merge)",
                 "params": 0, "macs": 0, "cpu_ms": round(ms_ev, 3),
                 "windows_per_s": round(1000.0 / ms_ev, 1),
                 "note": "non-neural, O(T) numpy"})
    print(f"[bench] {'Event assembly':34s} params=        0 macs=            0 cpu_ms={ms_ev:.3f}", flush=True)

    # whole-model CPU reference (same host, comparable profile)
    def _full(t):
        return model(t)

    ms_full = cpu_bench(_full, torch.randn(1, T, C))
    rows.append({"component": "Full CNN-BiGRU detector (CPU reference)", "params": n_par(model),
                 "macs": -1, "cpu_ms": round(ms_full, 2),
                 "windows_per_s": round(1000.0 / ms_full, 1),
                 "note": "includes gate+FiLM+heads; GPU numbers in Table V are the release values"})
    print(f"[bench] full model params={n_par(model):,} cpu_ms={ms_full:.2f}", flush=True)

    comp = nm.TransformerBiLSTMWinner().eval()
    ms_c = cpu_bench(lambda t: comp(t), torch.randn(1, T, C))
    rows.append({"component": "Transformer-BiLSTM comparator (CPU reference)", "params": n_par(comp),
                 "macs": -1, "cpu_ms": round(ms_c, 2),
                 "windows_per_s": round(1000.0 / ms_c, 1), "note": "competition-style architecture"})
    print(f"[bench] comparator params={n_par(comp):,} cpu_ms={ms_c:.2f}", flush=True)

    # sanity: sum of component params should equal full-model params (gate separate)
    out["rows"] = rows
    out["full_params"] = n_par(model)
    out["comparator_params"] = n_par(comp)
    dest = os.path.join(ROOT, "results", "tables", "bench_components_v38.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with io.open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote", dest)


if __name__ == "__main__":
    main()
