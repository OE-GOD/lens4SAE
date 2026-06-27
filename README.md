# FeatureScope

**Screen SAE features as *drivers* vs *thermometers* — so you don't reward a signal the model can game.**

*Status: research prototype (v0.4) — a mechanistic-interpretability research tool, not a production product. Validated on Gemma-2-2b + Gemma Scope L12 (sentiment & formality); see Scope & limits.*

When you use an interpretability feature as an RL reward or a monitor, only some features actually
*work*. FeatureScope tells the two apart:

- a **driver** is a feature the model *computes with* — steering it changes behaviour (read = write);
- a **thermometer** is decodable/correlated but **causally inert** — steering it does nothing (read ≠ write).

Reward a **driver** and you train the real behaviour. Reward a **thermometer** and the optimizer
**games the gauge** (Goodhart). FeatureScope finds the thermometers so you can rule them out.

> ## ⚠️ Scope & limits (read first)
> FeatureScope is a **one-sided _negative_ screen.** It can confidently tell you a feature is a
> **thermometer (rule it out)**. It does **not** certify that a feature is *safe to optimize* —
> there is deliberately **no `safe_to_optimize`** in the API. Driver-ness is **necessary, not
> sufficient**: a feature that is a driver under gentle steering can still **decouple under hard
> optimization** (off-manifold). Validated on **one model (Gemma-2-2b), one SAE (Gemma Scope L12),
> sentiment**. Cause is a **relative ranking**, magnitude-controlled, with bootstrap CIs — not a
> calibrated absolute. **Concept-general (v0.3):** any concept works via an injectable readout; the
> self-test uses **synthetic anchors** (the concept's difference-of-means direction as a guaranteed
> driver vs a random null), so no per-concept ground-truth features are needed. Concepts that are
> **not linear directions** (e.g. relational ones like comparison) make the self-test **fail by
> design** — the tool refuses rather than emitting junk. Demonstrated on sentiment & formality.
> Treat outputs as *candidates and exclusions*, not guarantees.

## Why trust the labels: it self-tests against ground truth

Before it will report, FeatureScope runs a **self-test** with **synthetic anchors**: it manufactures a
guaranteed driver (the concept's difference-of-means direction) and a guaranteed null (a random
direction) and checks it can tell them apart. If it can't, it **refuses to label** (`report()`
raises). A tool that validates itself before speaking — for *any* concept, with no pre-labeled features.

## How it works

1. **Read** — correlation of each SAE feature's activation with your concept (cheap; finds candidates).
2. **Cause** — steer each feature **in its own activation units** (not a shared norm, so high-firing
   features aren't unfairly favoured — the *magnitude confound*), and measure the shift in the model's
   behaviour, compared to a **random-direction control at matched magnitude** (so the effect must be
   *specific* to the feature, not generic perturbation). Bootstrap CIs over evaluation prompts.
3. **Verdict** — decided on a **unit-free robust z** (= SDs of the feature's effect above its *own*
   random-direction noise floor), so one rule transfers across concepts on different readout scales.
   `RULED_OUT` (z ≤ rule-out gate → thermometer) / `NOT_RULED_OUT` (z ≥ driver gate → driver-like,
   *not* certified) / `INDETERMINATE`. The z gates are **robust, cross-domain defaults pending
   leave-one-concept-out validation** — not claimed optimal (see issue #1).

## Install & run

```bash
pip install -e ".[gemma]"        # needs transformer-lens, sae-lens, transformers
featurescope                       # built-in sentiment demo
featurescope --concept formality   # built-in formality demo
featurescope --concept formality --csv mydata.csv   # your own concept's data (columns: text,label)
```

```python
from featurescope import FeatureScope, FORMALITY
fs = FeatureScope(layer=12, concept=FORMALITY).fit(formal_texts, casual_texts).calibrate()  # or any ReadoutSpec
fs.screen(top_k=20)
print(fs.ruled_out_thermometers())   # the honest output: features to EXCLUDE as rewards
fs.report()
```

## Background

This packages a finding from a series of mech-interp experiments: across ~9,000 sentiment SAE
features, a feature's **readability** and its **causal effect** are nearly **uncorrelated** — so
selecting interpretability signals by *decodability* (the common default) is dominated by causally
inert thermometers. Drivers vs thermometers is the causal-mediator-vs-correlational-probe distinction,
applied to choosing safe reward/monitor signals. Write-ups: https://oe-god.github.io

## License

MIT.
