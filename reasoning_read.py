"""Strongest test of read-depth: a REASONING concept can't be read at layer 0.

arithmetic correctness = minimal pairs (same equation, answer right vs wrong). The answer tokens overlap
between correct and incorrect sets, so there's NO lexical cue -> you must COMPUTE the sum to know
correctness. Prediction: sentiment reads at L0 (surface); arithmetic only becomes readable DEEP (after
the model does the math). If so: surface concepts read early, reasoning concepts read late.
"""
import numpy as np, torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from featurescope import data

# minimal pairs: identical LHS in both sets; only the RHS (correct vs off-by-small) differs.
ARITH_POS = ["3 + 4 = 7", "8 + 5 = 13", "6 + 2 = 8", "9 + 7 = 16", "5 + 5 = 10", "12 + 3 = 15",
             "7 + 8 = 15", "4 + 9 = 13", "11 + 6 = 17", "2 + 8 = 10", "14 + 5 = 19", "6 + 6 = 12",
             "10 + 7 = 17", "3 + 9 = 12", "8 + 8 = 16", "5 + 7 = 12"]
ARITH_NEG = ["3 + 4 = 8", "8 + 5 = 12", "6 + 2 = 9", "9 + 7 = 15", "5 + 5 = 11", "12 + 3 = 14",
             "7 + 8 = 16", "4 + 9 = 14", "11 + 6 = 18", "2 + 8 = 11", "14 + 5 = 20", "6 + 6 = 13",
             "10 + 7 = 18", "3 + 9 = 14", "8 + 8 = 15", "5 + 7 = 13"]

LAYERS = [0, 4, 8, 12, 16, 20, 24]
CONCEPTS = {"arithmetic": (ARITH_POS, ARITH_NEG), "sentiment": data.examples_for("sentiment")}


def auroc(s, y):
    s = np.array(s); y = np.array(y); pos, neg = s[y == 1], s[y == 0]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=dev, dtype=torch.bfloat16)
    hooks = {f"blocks.{L}.hook_resid_post" for L in LAYERS}

    resids = {c: {L: [] for L in LAYERS} for c in CONCEPTS}
    ys = {}
    for c, (pos, neg) in CONCEPTS.items():
        ys[c] = np.array([1] * len(pos) + [0] * len(neg))
        for t in pos + neg:
            with torch.no_grad():
                _, cache = model.run_with_cache(model.to_tokens(t[:300]), names_filter=lambda n: n in hooks)
            for L in LAYERS:
                resids[c][L].append(cache[f"blocks.{L}.hook_resid_post"][0].float().cpu())

    read = {c: {} for c in CONCEPTS}
    for L in LAYERS:
        sae = SAE.from_pretrained("gemma-scope-2b-pt-res-canonical", f"layer_{L}/width_16k/canonical", device=dev)
        for c in CONCEPTS:
            feats = [sae.encode(r.to(dev)).detach().max(0).values.float().cpu().numpy() for r in resids[c][L]]
            X = np.stack(feats); d = X[ys[c] == 1].mean(0) - X[ys[c] == 0].mean(0)
            read[c][L] = auroc(X @ d, ys[c])
        del sae

    print(f"\n=== read-AUROC by layer (decodability) — surface vs reasoning ===")
    print(f"{'concept':>11} | " + " ".join(f"L{L:<2}" for L in LAYERS))
    for c in CONCEPTS:
        print(f"{c:>11} | " + " ".join(f"{read[c][L]:.2f}" for L in LAYERS))
    print()
    for c in CONCEPTS:
        mx = max(read[c].values()); thr = 0.5 + 0.95 * (mx - 0.5)
        rd = next(L for L in LAYERS if read[c][L] >= thr)
        print(f"{c:>11}: read-depth = L{rd}  (peak AUROC {mx:.2f})")
    print("\nPrediction: sentiment readable at L0 (surface); arithmetic only readable deep (must compute).")


if __name__ == "__main__":
    main()
