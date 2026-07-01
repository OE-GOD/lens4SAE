"""Phase A amendment pass (pre-freeze; documents two corrections to the vetting rules).

Why (from the first Phase A run):
1. YES-BIAS in probe vetting: every dropped probe was a FALSE-side probe — the model's verdict has a
   'correct' floor, so absolute-sign vetting rejects exactly the items we need for balance. The design
   review itself warned 'interpret shifts relative to the null, never to 0'. Amended rule: probes are
   vetted RELATIVELY — a probe pair survives iff readout(TRUE) > readout(paired FALSE). Parity probes
   have no mirror pairs; they are vetted by the completion screen instead (all 24 passed).
2. ARITHMETIC ANCHOR near chance (paired 0.55) on the 40-pair random generator (operands 2-19): the
   model cannot behaviorally judge hard two-digit sums. The prior finding used 16 SMALL-SUM pairs.
   Amended rule: the anchor uses a small-sum generator (a in 2-12, b in 2-9), accepted only if it
   passes the same stage-0 gates as the vote concepts; otherwise the anchor gate FAILS and the prior
   arithmetic finding is reported as unsupported at the behavioral level.
"""
import json
import numpy as np, torch
from transformer_lens import HookedTransformer
import computed_concepts as cc
from phase_a import stage0, binom_p


def small_sum_arith(n=40, seed=0):
    import random
    rng = random.Random(seed)
    pos, neg, seen = [], [], set()
    while len(pos) < n:
        a, b = rng.randint(2, 12), rng.randint(2, 9)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        delta = rng.choice([-2, -1, 1, 2])
        pos.append(f"{a} + {b} = {a + b}"); neg.append(f"{a} + {b} = {a + b + delta}")
    return pos, neg


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=dev, dtype=torch.float32)
    out = {}

    # ---- arithmetic anchor: original 16 fixed pairs, and the small-sum generator ----
    r_ar = cc.make_readout(model, cc.arith_spec())
    from arith_steer import ARITH_POS, ARITH_NEG
    st16 = stage0(r_ar, ARITH_POS, ARITH_NEG)
    sp, sn = small_sum_arith()
    st40 = stage0(r_ar, sp, sn)
    out["arith_fixed16"] = {k: st16[k] for k in ("auroc", "paired_acc", "binom_p")}
    out["arith_smallsum40"] = {k: st40[k] for k in ("auroc", "paired_acc", "binom_p")}
    anchor_ok = st40["auroc"] >= 0.65 and st40["paired_acc"] >= 0.75
    out["anchor_choice"] = "smallsum40" if anchor_ok else (
        "fixed16" if (st16["auroc"] >= 0.65 and st16["paired_acc"] >= 0.75) else "FAIL")
    print(f"[arith] fixed16: AUROC {st16['auroc']:.3f} paired {st16['paired_acc']:.2f}   "
          f"smallsum40: AUROC {st40['auroc']:.3f} paired {st40['paired_acc']:.2f}  -> anchor={out['anchor_choice']}")

    # ---- comparison probes: RELATIVE vetting on mirror pairs ----
    r_cmp = cc.make_readout(model, cc.cmp_spec(cc.CMP_FEWSHOT_4))
    keep_t, keep_f = [], []
    for t, f in zip(cc.CMP_PROBES_TRUE, cc.CMP_PROBES_FALSE):
        if r_cmp(t) > r_cmp(f):
            keep_t.append(t); keep_f.append(f)
    out["comparison_probes"] = {"kept_pairs": len(keep_t), "of": len(cc.CMP_PROBES_TRUE),
                                "true": keep_t, "false": keep_f}
    print(f"[comparison] probe pairs kept (relative vetting): {len(keep_t)}/{len(cc.CMP_PROBES_TRUE)}")

    # ---- capitals probes: RELATIVE vetting per country ----
    r_cap = cc.make_readout(model, cc.cap_spec())
    false_of = dict((c, x) for c, x in cc.CAP_PROBES_FALSE)
    keep_ct, keep_cf = [], []
    for c, city in cc.CAP_PROBES_TRUE:
        if r_cap(f"The capital of {c} is {city}") > r_cap(f"The capital of {c} is {false_of[c]}"):
            keep_ct.append((c, city)); keep_cf.append((c, false_of[c]))
    out["capitals_probes"] = {"kept_pairs": len(keep_ct), "of": len(cc.CAP_PROBES_TRUE),
                              "true": keep_ct, "false": keep_cf}
    print(f"[capitals] probe pairs kept (relative vetting): {len(keep_ct)}/{len(cc.CAP_PROBES_TRUE)}")

    # ---- parity probes: completion screen already passed all 24 -> keep all (recorded decision) ----
    out["parity_probes"] = {"rule": "completion-screen (no mirror pairs exist)", "kept": len(cc.PAR_PROBES)}

    with open("phase_a2_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nwrote phase_a2_results.json — next: PREREG.md + freeze commit, then Phase B.")


if __name__ == "__main__":
    main()
