"""Phase A: behavioral vetting only — NO read/steer measurements happen here.

Per the pre-registered two-phase protocol from the design review: item changes are allowed ONLY in this
phase, via pre-stated rules (drop a fit pair only if the model misjudges BOTH sides; abort a concept if
>2 pairs drop; probes must be judged correctly on both sides; comparison few-shot chosen by paired
accuracy; capitals foil block needs >=4 survivors). After Phase A, datasets + PREREG.md are frozen by
git commit; Phases B-D measure depth with no further item/threshold edits.

Everything runs fp32 (bf16's 0.125 ulp quantizes verdict logit-diffs whose true gaps are ~0.16).
"""
import json, math
import numpy as np, torch
from transformer_lens import HookedTransformer
import computed_concepts as cc


def binom_p(k, n):
    """one-sided exact P(X >= k) under p=0.5"""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def auroc(s, y):
    s = np.asarray(s, dtype=float); y = np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def stage0(readout, pos, neg):
    dp = [readout(t) for t in pos]; dn = [readout(t) for t in neg]
    wins = sum(p > n for p, n in zip(dp, dn))
    return {"auroc": auroc(dp + dn, [1] * len(dp) + [0] * len(dn)),
            "paired_acc": wins / len(pos), "binom_p": binom_p(wins, len(pos)),
            "pos_scores": dp, "neg_scores": dn}


def drops_both_sides(st):
    return [i for i, (p, n) in enumerate(zip(st["pos_scores"], st["neg_scores"])) if p <= 0 and n >= 0]


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=dev, dtype=torch.float32)
    out = {"dtype": "float32", "device": dev}

    fails = cc.check_all(model)
    if fails:
        print("DATASET ASSERTIONS FAILED — aborting Phase A:")
        for f in fails:
            print("  -", f)
        raise SystemExit(1)
    print("[assertions] all dataset invariants hold (token multisets, vocab disjointness, no trailing '.')")

    # ---- comparison: few-shot A/B, then stage-0 + item vetting under the winner ----
    st4 = stage0(cc.make_readout(model, cc.cmp_spec(cc.CMP_FEWSHOT_4)), cc.CMP_POS, cc.CMP_NEG)
    st2 = stage0(cc.make_readout(model, cc.cmp_spec(cc.CMP_FEWSHOT_2)), cc.CMP_POS, cc.CMP_NEG)
    use4 = st4["paired_acc"] >= st2["paired_acc"]
    st_cmp, few = (st4, "4shot") if use4 else (st2, "2shot")
    print(f"[comparison] few-shot A/B: 4shot paired {st4['paired_acc']:.2f} vs 2shot {st2['paired_acc']:.2f} -> {few}")
    r_cmp = cc.make_readout(model, cc.cmp_spec(cc.CMP_FEWSHOT_4 if use4 else cc.CMP_FEWSHOT_2))
    cmp_drops = drops_both_sides(st_cmp)
    cmp_probe_bad = [s for s, truth in zip(cc.CMP_PROBES_TRUE + cc.CMP_PROBES_FALSE,
                                           [True] * 12 + [False] * 12)
                     if (r_cmp(s) > 0) != truth]
    out["comparison"] = {"few_shot": few, "stage0": {k: st_cmp[k] for k in ("auroc", "paired_acc", "binom_p")},
                         "fit_pair_drops": sorted({cc.CMP_CLUSTER[i] for i in cmp_drops}),
                         "probe_drops": cmp_probe_bad,
                         "gate": st_cmp["auroc"] >= 0.65 and st_cmp["paired_acc"] >= 0.75}

    # ---- capitals: stage-0, item vetting, few-shot sanity, foils, probe filter ----
    r_cap = cc.make_readout(model, cc.cap_spec())
    st_cap = stage0(r_cap, cc.CAP_POS, cc.CAP_NEG)
    cap_drops = drops_both_sides(st_cap)
    # few-shot anchor sanity: bare completion must strongly prefer ' Buenos' over ' Kabul'
    with torch.no_grad():
        lg = model(model.to_tokens("The capital of Argentina is"))[0, -1]
    tb = model.to_tokens(" Buenos", prepend_bos=False)[0, 0]; tk = model.to_tokens(" Kabul", prepend_bos=False)[0, 0]
    kabul_ok = float(lg[tb] - lg[tk]) > 0
    foil_surv = [c for c, cap, foil in cc.CAP_FOILS
                 if r_cap(f"The capital of {c} is {cap}") > 0 and r_cap(f"The capital of {c} is {foil}") < 0]
    bad_countries = [c for (c, x), (c2, x2) in zip(cc.CAP_PROBES_TRUE, cc.CAP_PROBES_FALSE)
                     if r_cap(f"The capital of {c} is {x}") <= 0] \
        + [c2 for (c2, x2) in cc.CAP_PROBES_FALSE if r_cap(f"The capital of {c2} is {x2}") >= 0]
    bad_countries = sorted(set(bad_countries))
    probes_t = [p for p in cc.CAP_PROBES_TRUE if p[0] not in bad_countries]
    probes_f = [p for p in cc.CAP_PROBES_FALSE if p[0] not in bad_countries]
    if len(probes_t) + len(probes_f) < 16:
        bc, bcity = cc.CAP_PROBE_FALLBACK
        if r_cap(f"The capital of {bc} is {bcity}") > 0:
            probes_t.append((bc, bcity))
    out["capitals"] = {"stage0": {k: st_cap[k] for k in ("auroc", "paired_acc", "binom_p")},
                       "fit_pair_drops": [cc.CAP_FIT[i][0] for i in cap_drops],
                       "kabul_anchor_ok": kabul_ok, "foil_survivors": foil_surv,
                       "foil_transfer_enabled": len(foil_surv) >= 4,
                       "probe_countries_dropped": bad_countries,
                       "final_probes_true": probes_t, "final_probes_false": probes_f,
                       "gate": st_cap["auroc"] >= 0.90 and st_cap["paired_acc"] >= 0.75}

    # ---- parity: stage-0, item vetting, completion screen, probe filter ----
    r_par = cc.make_readout(model, cc.par_spec())
    st_par = stage0(r_par, cc.PAR_POS, cc.PAR_NEG)
    par_drops = drops_both_sides(st_par)
    te = model.to_tokens(" even", prepend_bos=False)[0, 0]; to = model.to_tokens(" odd", prepend_bos=False)[0, 0]
    def completes_even(n):
        with torch.no_grad():
            lg = model(model.to_tokens(f"Statement: {n} is an"))[0, -1]
        return float(lg[te] - lg[to]) > 0
    fit_screen_bad = [n for n in cc.PAR_EVENS if not completes_even(n)] + [n for n in cc.PAR_ODDS if completes_even(n)]
    par_probe_bad = [s for s, truth in zip(cc.PAR_PROBES, cc.PAR_PROBE_TRUTH) if (r_par(s) > 0) != truth]
    out["parity"] = {"stage0": {k: st_par[k] for k in ("auroc", "paired_acc", "binom_p")},
                     "fit_pair_drops": [int(cc.PAR_POS[i].split()[0]) for i in par_drops],
                     "completion_screen_flags": fit_screen_bad, "probe_drops": par_probe_bad,
                     "gate": st_par["auroc"] >= 0.65 and st_par["paired_acc"] >= 0.75}

    # ---- anchor + control stage-0 (validity gates for the whole instrument) ----
    ar_pos, ar_neg = cc.arith_sets()
    st_ar = stage0(cc.make_readout(model, cc.arith_spec()), ar_pos, ar_neg)
    s_pos, s_neg, s_spec = cc.sentiment_sets_and_spec()
    st_se = stage0(cc.make_readout(model, s_spec), s_pos, s_neg)
    out["arith_anchor"] = {"stage0": {k: st_ar[k] for k in ("auroc", "paired_acc", "binom_p")}}
    out["sentiment_control"] = {"stage0": {k: st_se[k] for k in ("auroc", "paired_acc", "binom_p")}}

    print("\n=== Phase A report ===")
    for c in ("comparison", "capitals", "parity"):
        o = out[c]; s = o["stage0"]
        drops = o["fit_pair_drops"]
        verdict = "ABORT->replace (>2 drops)" if len(drops) > 2 else ("PASS" if o["gate"] else "FAIL stage-0 gate")
        print(f"{c:>12}: AUROC {s['auroc']:.3f}  paired {s['paired_acc']:.2f} (p={s['binom_p']:.4f})  "
              f"drops {drops}  -> {verdict}")
    print(f"{'arithmetic':>12}: AUROC {out['arith_anchor']['stage0']['auroc']:.3f}  "
          f"paired {out['arith_anchor']['stage0']['paired_acc']:.2f}   (anchor)")
    print(f"{'sentiment':>12}: AUROC {out['sentiment_control']['stage0']['auroc']:.3f}  "
          f"paired {out['sentiment_control']['stage0']['paired_acc']:.2f}   (control)")
    print(f"\ncapitals: kabul_anchor_ok={out['capitals']['kabul_anchor_ok']}  "
          f"foil survivors={out['capitals']['foil_survivors']} "
          f"(transfer test {'ON' if out['capitals']['foil_transfer_enabled'] else 'OFF -> bare-pair control only'})")
    print(f"comparison probe drops: {out['comparison']['probe_drops']}")
    print(f"capitals probe countries dropped: {out['capitals']['probe_countries_dropped']}")
    print(f"parity probe drops: {out['parity']['probe_drops']}  completion flags: {out['parity']['completion_screen_flags']}")

    with open("phase_a_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nwrote phase_a_results.json — next: apply drops, write PREREG.md, freeze by commit, run Phase B.")


if __name__ == "__main__":
    main()
