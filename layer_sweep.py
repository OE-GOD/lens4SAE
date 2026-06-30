"""Layer sweep: WHERE does 'certainty' live? It's weak at layer 12 (synthetic-driver z~2.2, refused).
Does the certainty direction get strong enough to PASS the self-test at a different depth?

Uses the Gemma Scope CANONICAL SAEs (one width_16k SAE per layer). For each layer we run certainty's
self-test (synthetic difference-of-means driver vs random null) and report the driver z. A layer where
z >= 3 means certainty IS a usable causal direction there -- i.e. confidence is represented at that depth.
"""
from featurescope import FeatureScope
from featurescope.concepts import CERTAINTY
from featurescope import data

LAYERS = [3, 6, 9, 12, 15, 18, 21]
pos, neg = data.examples_for("certainty")


def main():
    results = {}
    for L in LAYERS:
        try:
            fs = FeatureScope(layer=L, concept=CERTAINTY,
                              sae_release="gemma-scope-2b-pt-res-canonical",
                              sae_id=f"layer_{L}/width_16k/canonical").fit(pos, neg).calibrate()
            results[L] = (fs.self_test["driver_z"], fs.self_test["null_z"], fs.self_test["passed"])
        except Exception as e:
            results[L] = ("ERR", str(e)[:60], False)
    print("\n=== certainty across layers (does confidence become a usable direction at some depth?) ===")
    print(f"{'layer':>6}{'driver_z':>10}{'null_z':>9}   self-test (gate z>=3)")
    for L, (dz, nz, p) in results.items():
        if dz == "ERR":
            print(f"{L:>6}   ERROR: {nz}")
        else:
            print(f"{L:>6}{dz:>10.2f}{nz:>9.2f}   {'PASS' if p else 'fail'}")
    ok = [L for L, (dz, nz, p) in results.items() if p]
    print(f"\nlayers where certainty PASSES (is a usable causal direction): {ok if ok else 'none'}")
    print("If some layer passes -> confidence is causally represented there (just not at 12).")
    print("If none -> certainty isn't a clean linear/causal direction in this 2B model (needs bigger model, or it's relational).")


if __name__ == "__main__":
    main()
