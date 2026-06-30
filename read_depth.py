"""READ-depth vs STEER-depth — turning the lesson into a result.

READ-depth = where a concept becomes linearly DECODABLE in the residual stream (separability of pos vs
neg by the difference-of-means direction, per layer). Cheap: one cached forward per example gives all layers.
STEER-depth = where the concept is most causally steerable (driver-z peak), from depth_profiles.py.

Hypothesis (from the depth-profiles refutation): READ saturates EARLY, STEER peaks LATE -> read-depth << steer-depth.
"""
import numpy as np, torch
from transformer_lens import HookedTransformer
from featurescope import data

LAYERS = list(range(0, 26, 2))
CONCEPTS = ["sentiment", "toxicity", "formality", "certainty"]
STEER_PEAK = {"sentiment": 18, "toxicity": 15, "formality": 18, "certainty": 15}  # driver-z peaks (depth_profiles)


def auroc(scores, y):
    s = np.array(scores); y = np.array(y)
    pos, neg = s[y == 1], s[y == 0]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=dev, dtype=torch.bfloat16)
    hooks = {f"blocks.{L}.hook_resid_post" for L in LAYERS}

    print(f"\n{'concept':>10} | read-AUROC by layer (decodability)" )
    summary = {}
    for c in CONCEPTS:
        pos, neg = data.examples_for(c)
        per_layer = {L: [] for L in LAYERS}
        for t in pos + neg:
            with torch.no_grad():
                _, cache = model.run_with_cache(model.to_tokens(t[:300]), names_filter=lambda n: n in hooks)
            for L in LAYERS:
                per_layer[L].append(cache[f"blocks.{L}.hook_resid_post"][0].float().mean(0).cpu().numpy())
        y = np.array([1] * len(pos) + [0] * len(neg))
        aurocs = {}
        for L in LAYERS:
            X = np.stack(per_layer[L]).astype(np.float64)
            d = X[y == 1].mean(0) - X[y == 0].mean(0)
            aurocs[L] = auroc(X @ d, y)
        # read-depth = first layer reaching 95% of the max AUROC-above-chance
        mx = max(aurocs.values()); thresh = 0.5 + 0.95 * (mx - 0.5)
        read_depth = next(L for L in LAYERS if aurocs[L] >= thresh)
        summary[c] = (read_depth, mx)
        print(f"{c:>10} | " + " ".join(f"{aurocs[L]:.2f}" for L in LAYERS))

    print(f"\n{'layers:':>10}   " + " ".join(f"L{L}" for L in LAYERS))
    print(f"\n=== READ-depth vs STEER-depth ===")
    print(f"{'concept':>10}{'read-depth':>12}{'steer-depth':>13}   read << steer?")
    for c in CONCEPTS:
        rd = summary[c][0]; sd = STEER_PEAK[c]
        print(f"{c:>10}{('L'+str(rd)):>12}{('L'+str(sd)):>13}   {'YES' if rd < sd else 'no'}")
    print("\nread-depth = first layer at 95% of peak decodability; steer-depth = driver-z peak (depth_profiles).")
    print("If read-depth < steer-depth across concepts: a concept is READABLE early but most STEERABLE late.")


if __name__ == "__main__":
    main()
