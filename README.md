# FeatureScope

**Screen SAE features as *drivers* vs *thermometers* — so you don't reward a signal the model can game.**

*Status: research prototype (v0.2) — a mechanistic-interpretability research tool, not a production product. Validated on one model / SAE / concept; see Scope & limits.*

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
> calibrated absolute. **Sentiment-locked (v0.2):** read-scores are concept-general, but the *cause*
> step (a hard-coded sentiment readout) and the self-test (known *sentiment* anchors) are
> sentiment-specific — so `--csv` today means *your own sentiment data*, **not** an arbitrary concept.
> True arbitrary-concept support (an injectable readout + per-concept ground-truth anchors) is on the
> roadmap (v0.3). Treat outputs as *candidates and exclusions*, not guarantees.

## Why trust the labels: it self-tests against ground truth

Before it will report, FeatureScope runs a **self-test**: it checks that a *known driver* out-causes a
*known thermometer* on your corpus. If it can't separate them, it **refuses to label** (`report()`
raises). A tool that validates itself against ground truth before speaking.

## How it works

1. **Read** — correlation of each SAE feature's activation with your concept (cheap; finds candidates).
2. **Cause** — steer each feature **in its own activation units** (not a shared norm, so high-firing
   features aren't unfairly favoured — the *magnitude confound*), and measure the shift in the model's
   behaviour, compared to a **random-direction control at matched magnitude** (so the effect must be
   *specific* to the feature, not generic perturbation). Bootstrap CIs over evaluation prompts.
3. **Verdict** — `RULED_OUT` (thermometer) / `NOT_RULED_OUT` (driver-like, *not* certified) /
   `INDETERMINATE`. The threshold is **calibrated from the known anchors**, never hard-coded.

## Install & run

```bash
pip install -e ".[gemma]"        # needs transformer-lens, sae-lens, transformers
featurescope                     # built-in sentiment demo
featurescope --csv mydata.csv    # your own SENTIMENT data (columns: text,label); arbitrary concepts = v0.3
```

```python
from featurescope import FeatureScope
fs = FeatureScope(layer=12).fit(pos_texts, neg_texts).calibrate()
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
