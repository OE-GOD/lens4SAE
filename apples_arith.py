"""Apples-to-apples: arithmetic read-depth AND steer-depth in the SAME representation (raw residual).

The earlier surprise (read L20 via SAE features, steer L12-18 via raw residual) was confounded -- two
different rulers. Here BOTH use the raw-residual mean-pooled difference-of-means direction:
  read  = AUROC of that direction separating correct vs incorrect examples (decoding).
  steer = driver-z from steering that SAME direction and watching the verdict (FeatureScope.calibrate).
If arithmetic still reads late but steers mid -> the 'compute-mid, read-late' story is real. If read now
also comes out mid -> the earlier read-L20 was an SAE-feature-space artifact and the story collapses.
"""
import numpy as np, torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from featurescope import FeatureScope
from featurescope.concepts import ReadoutSpec
from arith_steer import ARITH_POS, ARITH_NEG, ARITH   # reuse the exact concept + data

LAYERS = [8, 12, 16, 18, 20, 22, 24]


def auroc(s, y):
    s = np.array(s); y = np.array(y); pos, neg = s[y == 1], s[y == 0]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=dev, dtype=torch.bfloat16)
    y = np.array([1] * len(ARITH_POS) + [0] * len(ARITH_NEG))
    rows = {}
    for L in LAYERS:
        hook = f"blocks.{L}.hook_resid_post"
        # READ: mean-pooled raw-residual diff-of-means separability (same pooling fit() uses)
        means = []
        for t in ARITH_POS + ARITH_NEG:
            with torch.no_grad():
                _, cache = model.run_with_cache(model.to_tokens(t), names_filter=lambda n: n == hook)
            means.append(cache[hook][0].float().mean(0).cpu().numpy())
        X = np.stack(means); d = X[y == 1].mean(0) - X[y == 0].mean(0)
        read = auroc(X @ d, y)
        # STEER: steer that same raw-residual diff-of-means (FeatureScope synthetic driver)
        sae = SAE.from_pretrained("gemma-scope-2b-pt-res-canonical", f"layer_{L}/width_16k/canonical", device=dev)
        fs = FeatureScope(layer=L, concept=ARITH, model=model, sae=sae).fit(ARITH_POS, ARITH_NEG).calibrate()
        rows[L] = (read, fs.self_test["driver_z"]); del sae

    print("\n=== arithmetic: READ vs STEER, SAME representation (raw-residual diff-of-means) ===")
    print(f"{'layer':>6}{'read-AUROC':>12}{'steer-z':>10}")
    for L, (r, z) in rows.items():
        print(f"{L:>6}{r:>12.2f}{z:>10.2f}")
    rd = max(rows, key=lambda L: rows[L][0]); sd = max(rows, key=lambda L: rows[L][1])
    print(f"\nread peaks at L{rd} (AUROC {rows[rd][0]:.2f}); steer peaks at L{sd} (z {rows[sd][1]:.2f})")
    print("Same ruler now. If read-peak is DEEPER than steer-peak -> 'compute-mid, read-late' is real.")


if __name__ == "__main__":
    main()
