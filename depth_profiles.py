"""Depth profiles for all concepts: does ABSTRACTION track DEPTH?

Hypothesis: surface concepts (sentiment, toxicity) peak at shallower layers; abstract concepts that need
reasoning (certainty) peak deeper. We measure each concept's synthetic-driver z (self-test signal) across
layers and find where each PEAKS. Shares one loaded model across the whole grid; one canonical SAE per layer.
"""
import numpy as np
from featurescope import FeatureScope
from featurescope.concepts import REGISTRY
from featurescope import data

LAYERS = [9, 12, 15, 18]
CONCEPTS = ["sentiment", "toxicity", "formality", "certainty"]


def main():
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=dev, dtype=torch.bfloat16)

    grid = {c: {} for c in CONCEPTS}
    for L in LAYERS:
        sae = SAE.from_pretrained("gemma-scope-2b-pt-res-canonical", f"layer_{L}/width_16k/canonical", device=dev)
        for c in CONCEPTS:
            pos, neg = data.examples_for(c)
            try:
                fs = FeatureScope(layer=L, concept=REGISTRY[c], model=model, sae=sae).fit(pos, neg).calibrate()
                grid[c][L] = fs.self_test["driver_z"]
            except Exception as e:
                grid[c][L] = float("nan"); print(f"  [{c}@{L}] ERROR {str(e)[:50]}")

    print("\n=== depth profiles: synthetic-driver z per concept per layer (PASS if z>=3) ===")
    print(f"{'concept':>10}" + "".join(f"{('L'+str(L)):>8}" for L in LAYERS) + "   peak")
    for c in CONCEPTS:
        zs = [grid[c][L] for L in LAYERS]
        peakL = LAYERS[int(np.nanargmax(zs))]
        print(f"{c:>10}" + "".join(f"{z:>8.2f}" for z in zs) + f"   L{peakL} ({max(zs):.2f})")
    print("\nIf abstraction tracks depth: sentiment/toxicity peak shallower, certainty deepest.")


if __name__ == "__main__":
    main()
