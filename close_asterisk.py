"""Close the sentiment asterisk: re-measure READ-depth with the tool's PROPER reader (max-pooled SAE
features per layer), for all 4 concepts, so the read vs steer comparison is apples-to-apples.

read_depth.py used mean-pooled RAW residual (too weak for sentiment). The tool's read-score is the
max-pooled SAE-feature signal -- use that here. Expect a clean 4/4: read-depth << steer-depth.
"""
import numpy as np, torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from featurescope import data

LAYERS = [0, 4, 8, 12, 16, 20, 24]
CONCEPTS = ["sentiment", "toxicity", "formality", "certainty"]
STEER_PEAK = {"sentiment": 18, "toxicity": 15, "formality": 18, "certainty": 15}


def auroc(s, y):
    s = np.array(s); y = np.array(y); pos, neg = s[y == 1], s[y == 0]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=dev, dtype=torch.bfloat16)
    hooks = {f"blocks.{L}.hook_resid_post" for L in LAYERS}

    # cache full-seq residuals (on CPU) for every concept, all layers, in one pass each
    resids = {c: {L: [] for L in LAYERS} for c in CONCEPTS}
    ys = {}
    for c in CONCEPTS:
        pos, neg = data.examples_for(c); ys[c] = np.array([1] * len(pos) + [0] * len(neg))
        for t in pos + neg:
            with torch.no_grad():
                _, cache = model.run_with_cache(model.to_tokens(t[:300]), names_filter=lambda n: n in hooks)
            for L in LAYERS:
                resids[c][L].append(cache[f"blocks.{L}.hook_resid_post"][0].float().cpu())

    read = {c: {} for c in CONCEPTS}
    for L in LAYERS:
        sae = SAE.from_pretrained("gemma-scope-2b-pt-res-canonical", f"layer_{L}/width_16k/canonical", device=dev)
        for c in CONCEPTS:
            feats = []
            for r in resids[c][L]:
                with torch.no_grad():
                    feats.append(sae.encode(r.to(dev)).max(0).values.float().cpu().numpy())
            X = np.stack(feats); d = X[ys[c] == 1].mean(0) - X[ys[c] == 0].mean(0)
            read[c][L] = auroc(X @ d, ys[c])
        del sae

    print(f"\n=== READ-depth via SAE-feature reader (max-pooled) ===")
    print(f"{'concept':>10} | " + " ".join(f"L{L:<2}" for L in LAYERS))
    for c in CONCEPTS:
        print(f"{c:>10} | " + " ".join(f"{read[c][L]:.2f}" for L in LAYERS))

    print(f"\n=== READ-depth vs STEER-depth (clean) ===")
    print(f"{'concept':>10}{'read-depth':>12}{'steer-depth':>13}   read << steer?")
    for c in CONCEPTS:
        mx = max(read[c].values()); thr = 0.5 + 0.95 * (mx - 0.5)
        rd = next(L for L in LAYERS if read[c][L] >= thr); sd = STEER_PEAK[c]
        print(f"{c:>10}{('L' + str(rd)):>12}{('L' + str(sd)):>13}   {'YES' if rd < sd else 'no'}")


if __name__ == "__main__":
    main()
