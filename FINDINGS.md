# Findings: read-depth ≠ steer-depth

*A short, honest write-up of what FeatureScope turned up about **where** concepts live in a model.
All on Gemma-2-2b + Gemma Scope SAEs; small curated example sets; a 2B model — treat as suggestive,
not settled. Every claim is reproducible from the scripts named below.*

> **RE-MEASUREMENT OUTCOME (2026-07-03).** The pre-registered re-run (PREREG.md, commit 59d8bdc;
> corrections in PROTOCOL_NOTES.md) is complete. Formal verdict: **INSTRUMENT FAILURE** — the corrected
> steering gate's label-permutation nulls retain concept signal, so no positive steering effect (not
> even sentiment at 3.4 logits) can certify; all steer-depths below are therefore **retracted as
> unmeasurable**, neither confirmed nor refuted. What survived, strengthened: **arithmetic reads late**
> (read-depth L20 in both representations, leave-pairs-out CV, permutation p = 0.000 — the predicted
> bin, confirmed) and the **L20 backfire** (steering "correct" at the committal layer: dose-monotone
> crash to −1.34, z = −5.8, sign-test p = 0.0002, negation and cross-concept controls clean) — though
> the backfire is a one-layer knife edge (L19 +1.16, L20 −1.34, L21 0.00) and thus fails the
> pre-registered two-adjacent-layer replication clause: reported as strong-but-formally-unconfirmed.
> Of the three generalization concepts: parity died of a dataset artifact its 0.00 anti-ordered read
> exposed, capitals of a pre-registered permutation power floor, comparison tripped the L0 leak gate
> (embed-layer clean — likely block-0 computation, not a data leak; unevaluable by the letter).
> Full record: phase_bc_results.json, phase_d_results.json, score_verdict.py output.

## The question

FeatureScope sorts a model's features into **drivers** (steering them changes behaviour — causal) and
**thermometers** (decodable but causally inert). Its founding fact is `read ≠ cause`: a feature can
*correlate* with a concept without *causing* it.

This raises a sharper question across the model's **depth**: at what layer is a concept **readable**
(decodable from the residual stream), and at what layer is it **steerable** (intervening there changes
the output)? Are they the same depth?

## Method

- **read-depth** — at each layer, encode the residual stream with that layer's SAE, max-pool the
  feature activations, and measure how well a difference-of-means direction separates the concept's
  positive vs negative examples (AUROC). The shallowest layer reaching ~95% of peak separability is
  the read-depth. (`close_asterisk.py`; a cruder raw-residual version is `read_depth.py`.)
- **steer-depth** — sweep layers; at each, run FeatureScope's dose-response steering self-test and
  record the synthetic-driver z (signal vs a same-layer random-direction null). The layer where z
  peaks is the steer-depth. (`depth_profiles.py`, `layer_sweep.py`.)

## Result

Across **four concepts** (sentiment, toxicity, formality, certainty):

| concept | read-depth | steer-depth | read ≪ steer? |
|---|---|---|---|
| sentiment | **L0** | L18 | yes |
| toxicity | **L0** | L15 | yes |
| formality | **L0** | L18 | yes |
| certainty | **L0** | L15 | yes |

Every concept is **perfectly readable at layer 0** (the embeddings) yet only **steerable at layers
15–18**. A large, consistent gap.

## Interpretation

- **Detection is lexical and immediate.** These concepts are *in the words* — "rude", "formal",
  "definitely"/"might" are separable right at the embedding layer, before the model computes anything.
- **Control is deep and late.** To change the model's *output*, you must intervene near where it
  *commits to an answer* (late layers). A steer made early gets reworked/washed out by the layers
  above it before reaching the output.
- So **read-depth ≠ steer-depth**: a concept is *readable early but most steerable late.* This is
  `read ≠ cause`, shown across depth — and a caution for interpretability: **decodability at a layer
  does not imply control at that layer.**

## Reasoning concepts read *late*

The four concepts above read at layer 0 because they're **lexical** — the signal is in the words. To
test the converse, I used a **reasoning** concept the surface can't give away: **arithmetic correctness**,
as minimal pairs (`3 + 4 = 7` vs `3 + 4 = 8`; the answer tokens overlap between the correct and incorrect
sets, so there is no lexical cue — you must *compute* the sum).

| concept | read-depth |
|---|---|
| sentiment (surface) | L0 |
| arithmetic (reasoning) | **L20** |

Arithmetic hovers near chance (~0.7) through the early/mid layers and only becomes cleanly decodable at
**layer 20** — once the model has actually done the computation. So **read-depth tracks how much
computation a concept needs**: lexical concepts are available at layer 0; a concept requiring arithmetic
isn't available until deep. "Everything reads at L0" was an artifact of testing only *lexical* concepts.
(`reasoning_read.py`.)

## Reasoning reverses the order: steer *mid*, read *late*

Steering arithmetic completes the surprise. Measuring where it's *steerable* per layer (steer the
raw-residual correctness direction, score vs a same-layer null): it's steerable in the **mid** layers
(L12–18, peak z ≈ 8.4 at L12) and **not** deep (fails/negative at L20+). So for a reasoning concept the
order **reverses**: steer-depth (mid) is *shallower* than read-depth (late) — the opposite of surface
concepts.

To rule out a ruler artifact, I re-measured read **and** steer in the **same** representation
(raw-residual mean-pooled diff-of-means). Read still peaks deep (L24), steer still peaks mid (L12) — the
reversal survives.

| concept type | read-depth | steer-depth |
|---|---|---|
| surface (sentiment, toxicity, …) | early (L0) | late (L15–18) |
| reasoning (arithmetic) | late (L20–24) | mid (L12–18) |

**Reading:** the model *computes* arithmetic in mid-layers — and while it's computing, the correctness
direction is a malleable lever (steerable). The **result** is only readable once the computation is
done, deep. Surface concepts are readable from the input and controllable near the output; reasoning
concepts are controllable *while being computed* and readable *once computed*.

*Caveats:* the raw-residual read is weak/noisy (max AUROC 0.82; the clean read-late signal is the
SAE-feature version at L20). 2B, arithmetic only.

**The L20 backfire is real — a `represent ≠ control` signature (`chase_l20.py`).** Re-run with 40
example-pairs and finer layers: at **L20–21** — exactly where correctness is most *readable* (AUROC
0.79) — steering the correctness direction has a **dose-dependent negative** effect (CI entirely below
0; L20 effect [-0.75, -0.50], z ≈ -8; steering harder backfires more), not an isolated spike or a
straddle-zero fluke. So at the *committed* layers the concept is **represented but not controllable**:
steering *disrupts* the settled computation (the verdict flips toward "incorrect") instead of steering
it. (Why the disruption reads specifically as "incorrect" is interpreted, not proven.)

## A sub-investigation: where does "certainty" live?

Certainty was instructive. At layer 12 it **failed** FeatureScope's self-test (the tool refused to
label it). Diagnosis by elimination:
1. **Data?** The examples confounded content with certainty. Rewriting them as **minimal pairs**
   (same sentence, only the certainty word changes) tripled the signal (z 0.77 → 2.22) — but still
   below the gate.
2. **Layer?** A sweep across depths found certainty only passes around **layer 15** (z 3.37; profile:
   ~0 early → 2.2 @ L12 → 3.4 @ L15 → fades by L21). So confidence *is* causally represented — just
   deeper than the others, and weakly.

A natural guess — *abstraction tracks depth* (surface concepts shallow, abstract ones deep) — turned
out **false**: sentiment (the most "surface" concept) is the *strongest* and peaks the *deepest*. The
real differentiator across concepts is **strength** (certainty maxes at z≈3; the others reach z≈10–11),
not where they peak. (`depth_profiles.py`.)

## Honest caveats

- **Your read-measure matters.** A first pass with a *mean-pooled raw-residual* reader produced a
  spurious "sentiment reads late" result. The proper **max-pooled SAE-feature** reader fixed it
  (sentiment reads at L0 like the rest). Weak probe → artifact.
- **Proximity.** Steering nearer the output can inflate effects by itself. The z-score largely
  controls this (it compares to random directions *at the same layer*, so the proximity factor
  cancels), but a residual effect can't be fully ruled out.
- **Scope.** One model (2B), one SAE family, ~32 examples per concept, four concepts. Suggestive.
- **Predict-then-test.** The "abstraction = depth" claim was written down *before* the run and then
  refuted — which is the point of predicting first.

## Reproduce

```bash
pip install -e ".[gemma]"
python close_asterisk.py     # read-depth (SAE-feature reader) vs steer-depth, 4 concepts
python depth_profiles.py     # steer-depth (driver-z) across layers, 4 concepts
python layer_sweep.py        # certainty across depths
```
