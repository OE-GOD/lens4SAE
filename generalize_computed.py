"""Generalization test: does 'computed concepts read LATE but steer MID (+ backfire at committed
layers)' hold beyond arithmetic?

Three new COMPUTED concepts, each built as leakage-controlled minimal pairs (bag-of-words identical
across POS/NEG, only the binding/order differs -> no lexical cue; L0 read-AUROC ~0.5 is the built-in
leakage check):
  comparison  "17 is greater than 9"  vs the SAME pair swapped
  capitals    "The capital of France is Paris" vs a derangement (every city+country in BOTH sets)
  parity      "14 is an even number"  vs the same numbers with labels flipped

PRE-REGISTERED PREDICTIONS (written before the run):
  all three: L0 read ~ 0.5 (else the dataset leaks -> fix, don't interpret)
  comparison: read-depth ~L12-16 (binding), steer-depth ~L8-12  -> REVERSAL yes
  capitals:   read-depth mid ~L8-12 (MLP lookup), steer-depth mid -> reversal UNCLEAR (may coincide)
  parity:     stage-0 at risk (2B may not know 2-digit parity); if it passes: read late, steer mid -> yes
  claim generalizes if >=2/3 valid concepts show steer-depth < read-depth; backfire at read-committed
  layers is a bonus signature, not required.

Stage 0 gates everything: the readout must track truth behaviorally (AUROC >= 0.65) or the concept
isn't in the model and its depth profile is moot.
"""
import numpy as np, torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from featurescope import FeatureScope
from featurescope.concepts import ReadoutSpec

# ---------------- datasets (leakage-controlled minimal pairs) ----------------
CMP_PAIRS = [(17, 9), (23, 6), (31, 12), (8, 3), (45, 27), (14, 5), (62, 38), (29, 11), (7, 2),
             (53, 26), (19, 4), (36, 18), (41, 15), (28, 13), (9, 5), (74, 42), (16, 7), (58, 33),
             (21, 10), (37, 22)]
CMP_POS = [f"{a} is greater than {b}" for a, b in CMP_PAIRS]
CMP_NEG = [f"{b} is greater than {a}" for a, b in CMP_PAIRS]

CAPS = [("France", "Paris"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Germany", "Berlin"),
        ("Japan", "Tokyo"), ("Russia", "Moscow"), ("England", "London"), ("Egypt", "Cairo"),
        ("China", "Beijing"), ("Canada", "Ottawa"), ("Greece", "Athens"), ("Portugal", "Lisbon")]
_derange = CAPS[1:] + CAPS[:1]                       # country i gets city of country i+1
CAP_POS = [f"The capital of {c} is {city}" for c, city in CAPS]
CAP_NEG = [f"The capital of {c} is {wrong[1]}" for (c, _), wrong in zip(CAPS, _derange)]

EVENS = [14, 8, 22, 36, 12, 6, 48, 20]
ODDS = [9, 15, 27, 7, 33, 11, 45, 19]
PAR_POS = [f"{n} is an even number" for n in EVENS] + [f"{n} is an odd number" for n in ODDS]
PAR_NEG = [f"{n} is an odd number" for n in EVENS] + [f"{n} is an even number" for n in ODDS]

# arithmetic: read-only replication anchor (validates the harness against the known result)
AR_POS = ["3 + 4 = 7", "8 + 5 = 13", "6 + 2 = 8", "9 + 7 = 16", "5 + 5 = 10", "12 + 3 = 15",
          "7 + 8 = 15", "4 + 9 = 13", "11 + 6 = 17", "2 + 8 = 10", "14 + 5 = 19", "6 + 6 = 12",
          "10 + 7 = 17", "3 + 9 = 12", "8 + 8 = 16", "5 + 7 = 12"]
AR_NEG = ["3 + 4 = 8", "8 + 5 = 12", "6 + 2 = 9", "9 + 7 = 15", "5 + 5 = 11", "12 + 3 = 14",
          "7 + 8 = 16", "4 + 9 = 14", "11 + 6 = 18", "2 + 8 = 11", "14 + 5 = 20", "6 + 6 = 13",
          "10 + 7 = 18", "3 + 9 = 14", "8 + 8 = 15", "5 + 7 = 13"]

# ---------------- readouts (verdict-style, mirrors the arithmetic one) ----------------
def _spec(name, few_true, few_false, probes):
    return ReadoutSpec(
        name=name,
        few_shot=f"Statement: {few_true}. Verdict: correct\nStatement: {few_false}. Verdict: incorrect\n",
        template="Statement: {text}. Verdict:", pos_word=" correct", neg_word=" incorrect", probes=probes)

COMPARISON = _spec("comparison", "5 is greater than 2", "3 is greater than 8",
                   ["12 is greater than 4", "6 is greater than 15", "33 is greater than 20",
                    "10 is greater than 25", "18 is greater than 3", "5 is greater than 40",
                    "27 is greater than 14", "8 is greater than 30", "44 is greater than 19",
                    "13 is greater than 50", "35 is greater than 16", "7 is greater than 60"])
CAPITALS = _spec("capitals", "The capital of Poland is Warsaw", "The capital of Poland is Dublin",
                 ["The capital of Norway is Oslo", "The capital of Austria is Rome",
                  "The capital of Ireland is Dublin", "The capital of Sweden is Cairo",
                  "The capital of Turkey is Ankara", "The capital of Kenya is Lisbon",
                  "The capital of Peru is Lima", "The capital of Hungary is Madrid",
                  "The capital of Cuba is Havana", "The capital of Chile is Athens",
                  "The capital of Iran is Tehran", "The capital of Iraq is Tokyo"])
PARITY = _spec("parity", "4 is an even number", "5 is an even number",
               ["10 is an even number", "13 is an even number", "17 is an odd number",
                "24 is an odd number", "31 is an odd number", "16 is an even number",
                "25 is an even number", "38 is an odd number", "42 is an even number",
                "29 is an odd number", "50 is an odd number", "21 is an even number"])

CONCEPTS = {"comparison": (COMPARISON, CMP_POS, CMP_NEG),
            "capitals": (CAPITALS, CAP_POS, CAP_NEG),
            "parity": (PARITY, PAR_POS, PAR_NEG)}
LAYERS = [0, 4, 8, 12, 16, 20, 24]


def auroc(s, y):
    s = np.asarray(s, dtype=float); y = np.asarray(y); pos, neg = s[y == 1], s[y == 0]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=dev, dtype=torch.bfloat16)
    hooks = {f"blocks.{L}.hook_resid_post" for L in LAYERS}

    # ---- stage 0: behavioral gate (readout must track truth; no steering, layer-independent) ----
    sae12 = SAE.from_pretrained("gemma-scope-2b-pt-res-canonical", "layer_12/width_16k/canonical", device=dev)
    print("\n=== stage 0: does the model HOLD the concept behaviorally? (readout AUROC vs truth) ===")
    gate = {}
    for name, (spec, pos, neg) in CONCEPTS.items():
        fs = FeatureScope(layer=12, concept=spec, model=model, sae=sae12)
        scores = [fs._readout(t) for t in pos + neg]
        y = np.array([1] * len(pos) + [0] * len(neg))
        a = auroc(scores, y); gate[name] = a
        print(f"{name:>12}: behavioral AUROC {a:.2f}  -> {'PASS' if a >= 0.65 else 'FAIL (concept may not be in the model; depth profile moot)'}")
    del sae12
    if dev == "mps":
        torch.mps.empty_cache()

    # ---- cache residuals once for the read sweep (all concepts + arithmetic anchor) ----
    read_sets = {**{k: (v[1], v[2]) for k, v in CONCEPTS.items()}, "arith(anchor)": (AR_POS, AR_NEG)}
    resids = {c: {L: [] for L in LAYERS} for c in read_sets}
    ys = {}
    for c, (pos, neg) in read_sets.items():
        ys[c] = np.array([1] * len(pos) + [0] * len(neg))
        for t in pos + neg:
            with torch.no_grad():
                _, cache = model.run_with_cache(model.to_tokens(t[:300]), names_filter=lambda n: n in hooks)
            for L in LAYERS:
                resids[c][L].append(cache[f"blocks.{L}.hook_resid_post"][0].float().cpu())

    # ---- per layer: read (all concepts) + steer (gated concepts), one SAE load each ----
    read = {c: {} for c in read_sets}
    steer = {c: {} for c in CONCEPTS}
    for L in LAYERS:
        sae = SAE.from_pretrained("gemma-scope-2b-pt-res-canonical", f"layer_{L}/width_16k/canonical", device=dev)
        for c in read_sets:
            X = np.stack([sae.encode(r.to(dev)).detach().max(0).values.float().cpu().numpy() for r in resids[c][L]])
            d = X[ys[c] == 1].mean(0) - X[ys[c] == 0].mean(0)
            read[c][L] = auroc(X @ d, ys[c])
        for name, (spec, pos, neg) in CONCEPTS.items():
            fs = FeatureScope(layer=L, concept=spec, model=model, sae=sae).fit(pos, neg)
            fs._gen.manual_seed(0)
            du = fs._diffmeans / fs._diffmeans.norm()
            dose, z, ci = fs._measure(du, 12.0, 1.0, n_null=16)
            steer[name][L] = (dose, z, ci)
            print(f"  [L{L:>2}] {name:>12}: read {read[name][L]:.2f}  dose {[round(x,2) for x in dose]}  "
                  f"z {z:>6.1f}  CI [{ci[0]:.2f},{ci[1]:.2f}]", flush=True)
        print(f"  [L{L:>2}] {'arith(anchor)':>12}: read {read['arith(anchor)'][L]:.2f}  (read-only anchor)", flush=True)
        del sae
        if dev == "mps":
            torch.mps.empty_cache()

    # ---- summary ----
    print("\n=== SUMMARY: read-depth vs steer-depth (computed concepts) ===")
    print(f"{'concept':>14} | " + " ".join(f"  L{L:<2}" for L in LAYERS) + "  (read AUROC)")
    for c in read_sets:
        print(f"{c:>14} | " + " ".join(f"{read[c][L]:.2f}" for L in LAYERS))
    print(f"\n{'concept':>14} | " + " ".join(f"  L{L:<2}" for L in LAYERS) + "  (steer z; * = backfire CI<0)")
    for c in CONCEPTS:
        row = []
        for L in LAYERS:
            _, z, ci = steer[c][L]
            row.append(f"{z:5.1f}{'*' if ci[1] < 0 else ' '}")
        print(f"{c:>14} | " + " ".join(row))

    print("\n=== verdicts (pre-registered criteria) ===")
    for c in CONCEPTS:
        peak = max(read[c].values())
        if gate[c] < 0.65:
            print(f"{c:>14}: EXCLUDED (failed stage-0 behavioral gate, AUROC {gate[c]:.2f})"); continue
        if read[c][0] > 0.65:
            print(f"{c:>14}: LEAKY (L0 read {read[c][0]:.2f} > 0.65 — lexical cue; do not interpret)"); continue
        if peak < 0.65:
            print(f"{c:>14}: NEVER READABLE (peak {peak:.2f}) — depth undefined"); continue
        thr = 0.5 + 0.95 * (peak - 0.5)
        rd = next(L for L in LAYERS if read[c][L] >= thr)
        passing = [L for L in LAYERS if steer[c][L][1] >= 3]
        sd = max(passing, key=lambda L: steer[c][L][1]) if passing else None
        bf = [L for L in LAYERS if steer[c][L][2][1] < 0]
        rev = (sd is not None and sd < rd)
        print(f"{c:>14}: read-depth L{rd} (peak {peak:.2f})  steer-depth {('L' + str(sd)) if sd is not None else 'none'}  "
              f"reversal {'YES' if rev else 'no'}  backfire@{bf if bf else 'none'}")
    print("\nclaim generalizes if >=2 of 3 valid concepts show steer-depth < read-depth.")


if __name__ == "__main__":
    main()
