# Protocol notes (post-freeze, documented amendments)

## Note 2 — Mixed-precision fallback adopted for remaining steer cells (2026-07-03)

**What:** Steering cells at L16/L20/L24 are measured with the design review's pre-approved fallback:
bf16 trunk + fp32 readout head (ln_final + answer-token unembed columns + logit softcap computed in
fp32 outside the model graph; `mixed_head.py`). Cells L0–L12 remain as measured under full fp32.

**Why:** Full-fp32 measurement proved computationally infeasible on the study machine: per-forward
times of 4–12 s (vs 0.21–0.25 s under bf16 trunk), traced via stack sampling to Metal-memory
thrash from the 10.4 GB fp32 working set. Multiple runs stalled >2 h on a single cell. The review's
runtime note explicitly sanctioned this fallback ("if memory-tight: bf16 trunk + fp32
unembed/softcap/logit-diff — measured sufficient").

**Validation (before adoption):** exact re-run of the banked fp32 cell steer/comparison/L12 under
mixed mode, same seeds: dose [0.215,0.453,0.660]→[0.209,0.453,0.673]; per-probe shifts corr 0.981,
max |diff| 0.048 on effect scale 0.67; z 1.27→1.41; **gate verdict identical (non-pass)**.
Script: `validate_mixed.py`.

**Comparability note:** each layer's steer gate is null-referenced within its own mode (real effect
vs same-mode permutation nulls), so mode-related scale shifts largely cancel; the validation confirms
gate-level agreement across modes. `results.mode_notes` records the mode of every mixed cell.
med_norm tolerance on resume is relaxed to 2% under --mixed (bf16 residual norms differ ~0.5%).

## Note 1 — see PREREG.md "Protocol correction 1" (steering gate feasibility, pre-data).
