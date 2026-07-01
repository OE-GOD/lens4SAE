"""Issue #1: validate the driver/thermometer threshold, as far as a laptop honestly can.

A definitive validation needs an independent gold (does the feature get gamed as a reward -> GPU). But a
non-circular laptop check is CONVERGENT VALIDITY across two DIFFERENT interventions:
  - the gate decides via STEERING (add the feature): robust-z >= 3 -> driver.
  - ABLATION (remove it): necessity. A separate causal method the gate doesn't use.
If the gate picks genuinely causal features, its DRIVERS should show higher necessity than its
THERMOMETERS. If driver-necessity ~ thermometer-necessity, the gate isn't picking real causation.
(Caveat: drivers are often sufficient-but-not-necessary due to redundancy, so expect a modest gap.)
"""
import numpy as np, torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from featurescope import FeatureScope
from featurescope.concepts import REGISTRY
from featurescope import data

CONCEPTS = ["sentiment", "formality", "toxicity"]


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=dev, dtype=torch.bfloat16)
    sae = SAE.from_pretrained("gemma-scope-2b-pt-res", "layer_12/width_16k/average_l0_82", device=dev)

    rows = []
    for c in CONCEPTS:
        fs = FeatureScope(layer=12, concept=REGISTRY[c], model=model, sae=sae).fit(*data.examples_for(c)).calibrate()
        if not fs.self_test["passed"]:
            print(f"[{c}] self-test failed — skipped"); continue
        fs.screen(top_k=10)
        ndrv = sum(r.verdict.value == "not_ruled_out" for r in fs.results)
        nthr = sum(r.verdict.value == "ruled_out" for r in fs.results)
        print(f"[{c}] {ndrv} drivers, {nthr} thermometers, {len(fs.results)-ndrv-nthr} indeterminate")
        for r in fs.results:
            rows.append((c, r.z, r.necessity, r.verdict.value))

    drv = [nec for (_, _, nec, v) in rows if v == "not_ruled_out"]
    thr = [nec for (_, _, nec, v) in rows if v == "ruled_out"]
    print("\n=== convergent validity: necessity (ablation) of steering-gate drivers vs thermometers ===")
    print(f"drivers (z>=3):     n={len(drv):2d}  mean necessity={np.mean(drv):.3f}" if drv else "no drivers")
    print(f"thermometers (z<=1): n={len(thr):2d}  mean necessity={np.mean(thr):.3f}" if thr else "no thermometers")
    a = auroc(drv, thr)
    print(f"\nAUROC(necessity separates gate-drivers from gate-thermometers) = {a:.2f}")
    print("=> >0.5 means the STEERING gate's drivers are independently more causal under ABLATION too")
    print("   (convergent validity). ~0.5 means the two interventions disagree. Definitive gold (gaming)")
    print("   still needs a GPU; this is the honest laptop ceiling.")


if __name__ == "__main__":
    main()
