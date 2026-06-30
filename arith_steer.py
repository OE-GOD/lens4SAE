"""Arithmetic's STEER-depth: where can you steer arithmetic-correctness to move the verdict?

Completes the depth 2x2 for a reasoning concept. read-depth was L20 (must compute first). Prediction:
steer-depth is also LATE (>=~L20) and FAILS early -- you can't steer a concept the model hasn't computed
yet. Contrast: surface concepts steer at L15-18 despite reading at L0.
"""
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from featurescope import FeatureScope
from featurescope.concepts import ReadoutSpec

ARITH_POS = ["3 + 4 = 7", "8 + 5 = 13", "6 + 2 = 8", "9 + 7 = 16", "5 + 5 = 10", "12 + 3 = 15",
             "7 + 8 = 15", "4 + 9 = 13", "11 + 6 = 17", "2 + 8 = 10", "14 + 5 = 19", "6 + 6 = 12",
             "10 + 7 = 17", "3 + 9 = 12", "8 + 8 = 16", "5 + 7 = 12"]
ARITH_NEG = ["3 + 4 = 8", "8 + 5 = 12", "6 + 2 = 9", "9 + 7 = 15", "5 + 5 = 11", "12 + 3 = 14",
             "7 + 8 = 16", "4 + 9 = 14", "11 + 6 = 18", "2 + 8 = 11", "14 + 5 = 20", "6 + 6 = 13",
             "10 + 7 = 18", "3 + 9 = 14", "8 + 8 = 15", "5 + 7 = 13"]

ARITH = ReadoutSpec(
    name="arithmetic",
    few_shot="Statement: 2 + 2 = 4. Verdict: correct\nStatement: 2 + 2 = 5. Verdict: incorrect\n",
    template="Statement: {text}. Verdict:",
    pos_word=" correct", neg_word=" incorrect",
    probes=["7 + 6 = 13", "9 + 4 = 12", "5 + 3 = 8", "8 + 7 = 16", "4 + 4 = 9", "10 + 2 = 12",
            "6 + 5 = 12", "3 + 8 = 11", "9 + 9 = 19", "7 + 7 = 14", "2 + 6 = 9", "5 + 8 = 13"],
)

LAYERS = [8, 12, 16, 18, 20, 22, 24]


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=dev, dtype=torch.bfloat16)
    rows = {}
    for L in LAYERS:
        sae = SAE.from_pretrained("gemma-scope-2b-pt-res-canonical", f"layer_{L}/width_16k/canonical", device=dev)
        fs = FeatureScope(layer=L, concept=ARITH, model=model, sae=sae).fit(ARITH_POS, ARITH_NEG).calibrate()
        rows[L] = (fs.self_test["driver_z"], fs.self_test["passed"]); del sae

    print("\n=== arithmetic STEER-depth (synthetic-driver z per layer; gate z>=3) ===")
    print(f"{'layer':>6}{'driver_z':>10}   steerable?")
    for L, (z, p) in rows.items():
        print(f"{L:>6}{z:>10.2f}   {'YES' if p else 'no'}")
    passes = [L for L, (z, p) in rows.items() if p]
    print(f"\nlayers where arithmetic is steerable: {passes if passes else 'none'}")
    print("read-depth was L20. If steer-depth is also late (and early layers fail): reasoning is gated by")
    print("computation on BOTH read and steer -- unlike surface concepts (read L0, steer L15-18).")


if __name__ == "__main__":
    main()
