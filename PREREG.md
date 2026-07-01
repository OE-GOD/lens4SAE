# Pre-registration: does "computed concepts read late / steer mid" generalize beyond arithmetic?

*The git commit containing this file freezes the datasets (`computed_concepts.py`, Phase-A decisions in
`phase_a_results.json` + `phase_a2_results.json`) and these criteria. No item, threshold, or
classification edits after this commit. Phases B–D (`phase_bc.py`, then a confirmatory backfire script)
run against this registration. Written after Phase A vetting, before any read/steer forward pass.*

## Claim under test

Prior finding (n=1 computed concept, arithmetic; now under an instrument caveat — see FINDINGS.md):
surface concepts read at L0 but steer late (L15–18); arithmetic correctness reverses — readable only
late (L20–24), steerable mid (L12–18), with steering at the read-committed layers backfiring.
**Test:** does the reversal hold for three new computed concepts (comparison-order, capitals/recall,
parity), measured with the corrected instrument?

## Instrument (corrected)

- **read-depth** — leave-one-cluster-out CV on minimal-pair difference vectors (max-pooled SAE features
  per layer; clusters = number-pair / derangement-cycle / number). Primary statistic: fraction of
  held-out pairs correctly ordered (≡ paired LOPO AUROC). Cluster-sign permutation p; cluster-bootstrap
  CI. read-depth = shallowest layer ≥ 0.5 + 0.95·(peak − 0.5). DEFINED only if peak ≥ 0.80, perm p <
  0.05 at that layer, CI excludes 0.5, embed-layer AUROC ≤ 0.6, and L0 AUROC ≤ 0.65 (violation = LEAKY).
- **steer-depth** — diff-of-means direction steered at per-layer-scaled doses (base = α·median
  per-token resid norm at the layer, α anchored so L12 = the historical norm-12), dose multiples 1/2/4,
  fp32 readout on Phase-A-vetted probes. Primary null: 16 cluster-sign permutation refit directions at
  the 4× dose; empirical rank-p, Holm-corrected across the 7 layers; pass also requires a sustained
  dose-response (effect at 4× ≥ 0.5·peak, > 0). steer-depth = shallowest passing layer
  (first-crossing, same convention as read). Fixed-norm-12 rows recorded only as a robustness line —
  a reversal that appears ONLY under fixed norm is the artifact, not the finding.
- **Controls (gates for the whole run):** sentiment arm must show read ≈ L0 / steer L15–18 / no
  reversal; arithmetic anchor (the original fixed 16 pairs — the only set passing the behavioral gate,
  0.656/0.75, flagged marginal) must reproduce read-late / steer-mid under the corrected instrument or
  the prior finding is retracted as instrument artifact; label-shuffled control ≈ chance; unigram
  harness check ('greater' vs 'less') ≈ 1.0 at L0.

## Phase-A amendments (recorded before freeze)

1. Probe vetting is RELATIVE (readout(TRUE) > readout(paired FALSE)) — absolute-sign vetting rejects
   exactly the FALSE probes via the model's yes-bias floor. Under the amended rule 12/12 comparison and
   12/12 capitals probe pairs survive; parity probes are vetted by the completion screen (24/24 pass).
2. Anchor item set: original fixed 16 pairs (random generators fail the behavioral gate: 2–19 operands
   0.54, small-sum 0.50). The anchor's marginal behavioral reality (0.656/0.75) is itself reported.

## Protocol correction 1 (pre-data; found by the smoke test, before any confirmatory measurement)

The frozen steering gate ("empirical rank-p ≤ 0.05 vs ≥ 32 permutation nulls, Holm-corrected over 7
layers") is mathematically unsatisfiable: the smallest achievable rank-p with K nulls is 1/(K+1)
(0.0303 at K=32), while Holm's strictest step requires ≤ 0.05/7 ≈ 0.0071 — so no layer could ever
pass and every concept would be unevaluable *by construction*. Corrected gate, stricter in spirit and
feasible: a layer passes iff the real 4× effect (a) exceeds **all** permutation nulls (16-null screen,
escalated to 32 for candidates), (b) has robust z ≥ 3 against the null distribution (parametrically
~p < 0.0013, tighter than the Holm bound), and (c) shows a sustained dose-response with positive
effect. Rank-p is still recorded descriptively. The null FAMILY (cluster-sign permutation refits — the
substantive fix over isotropic directions) is unchanged. Also clarified: the read permutation-p gate is
evaluated at the peak layer. No confirmatory data existed when this correction was made.

## Pre-registered predictions

| arm | stage-0 (measured, fp32) | read-depth bin | steer-depth bin | reversal? |
|---|---|---|---|---|
| comparison | 0.892 / paired 1.00 | MID [L8–L16] | EARLY-MID [L4–L12] | **YES** (+ backfire at read-committed layers) |
| capitals | 1.000 / 1.00 | MID [L8–L16] | at-or-below read | flagged in advance as most likely to BREAK the reversal |
| parity | 0.809 / 0.81 | EARLY-MID [L4–L12]; if < L8 → reclassified shallow-computed, unevaluable, replaced | EARLY | only if evaluable |
| arithmetic (anchor, not a vote) | 0.656 / 0.75 (marginal) | [L20–L24] | [L12–L18] | must reproduce or prior finding is retracted |
| sentiment (control, not a vote) | 1.000 / 1.00 | L0 | [L15–L18] | must NOT reverse |

## Scoring (fixed now)

- Per-concept PASS = steer-depth ≤ read-depth − 4 layers, AND no sustained positive steering pass at
  read-committed layers, AND the ordering holds in the matched raw-resid representation.
- Evaluability = stage-0 gates already passed (above) AND read-depth DEFINED and in [L8, L20] AND
  steer-depth defined. Unevaluable concepts are replaced from the frozen backup list (sum-comparison,
  three-number ordering, divisibility-by-3, max-of-three, string-length comparison) — never scored.
- **GENERALIZES** = all three evaluable AND ≥ 2/3 pass. **ARITHMETIC-SPECIFIC** (claim fails) = ≥ 2
  evaluable concepts affirmatively show non-reversal. Anything else = **INDETERMINATE**, published as
  such. Backfire (Phase D, compound criterion: dose-monotone negative, replicated at an adjacent layer,
  negation + cross-concept specificity controls) is reported separately, never required for PASS.
- Cross-concept direction cosines + cross-steering transfer are reported unconditionally; if the
  directions are one shared "computed-truth" direction, the conclusion is rescoped to that, not to
  cross-concept generalization.
