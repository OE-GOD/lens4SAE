"""Phase D: confirmatory backfire test on densified layers.

Runs only on Phase B/C arms with coarse-grid negative 4x mean-CI upper bounds at/after read-depth.
This is intentionally separate from phase_bc.py so Phase B/C remains the frozen exploratory sweep.
"""
import argparse, json, math, sys
import numpy as np, torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from featurescope import FeatureScope
from featurescope.core import _robust_z
from featurescope.concepts import ReadoutSpec
import computed_concepts as cc
import phase_bc

STRENGTHS = (1.0, 2.0, 4.0)
VOTE_ARMS = ("comparison", "capitals", "parity")
TEST_ARMS = VOTE_ARMS + ("arith",)
RESULTS_IN = "phase_bc_results.json"
RESULTS_OUT = "phase_d_results.json"
RNG = np.random.RandomState(0)


def build_arms():
    """Build arms exactly as needed here; mirrors phase_bc.main()'s ARMS construction."""
    a1 = json.load(open("phase_a_results.json"))
    a2 = json.load(open("phase_a2_results.json"))
    few = cc.CMP_FEWSHOT_4 if a1["comparison"]["few_shot"] == "4shot" else cc.CMP_FEWSHOT_2
    cmp_probes = a2["comparison_probes"]["true"] + a2["comparison_probes"]["false"]
    cap_probes = cc.cap_probe_texts([tuple(x) for x in a2["capitals_probes"]["true"]],
                                    [tuple(x) for x in a2["capitals_probes"]["false"]])
    if a2["anchor_choice"] == "FAIL":
        print("ANCHOR GATE FAILED at Phase A2 -- instrument failure per prereg. Stopping.", flush=True)
        sys.exit(1)
    from phase_a2 import small_sum_arith
    from arith_steer import ARITH_POS, ARITH_NEG, ARITH
    ar_pos, ar_neg = (small_sum_arith() if a2["anchor_choice"] == "smallsum40" else (ARITH_POS, ARITH_NEG))
    s_pos, s_neg, s_spec = cc.sentiment_sets_and_spec()

    arms = {
        "comparison": dict(spec=cc.cmp_spec(few), pos=cc.CMP_POS, neg=cc.CMP_NEG,
                           cluster=cc.CMP_CLUSTER, vote=True),
        "capitals": dict(spec=cc.cap_spec(cap_probes), pos=cc.CAP_POS, neg=cc.CAP_NEG,
                         cluster=cc.CAP_CLUSTER, vote=True),
        "parity": dict(spec=cc.par_spec(), pos=cc.PAR_POS, neg=cc.PAR_NEG,
                       cluster=cc.PAR_CLUSTER, vote=True),
        "arith": dict(spec=ARITH, pos=ar_pos, neg=ar_neg, cluster=list(range(len(ar_pos))), vote=False),
        "sentiment": dict(spec=s_spec, pos=s_pos, neg=s_neg, cluster=list(range(len(s_pos))), vote=False),
    }
    sp = arms["comparison"]["spec"]
    arms["comparison"]["spec"] = ReadoutSpec(name=sp.name, few_shot=sp.few_shot, template=sp.template,
                                             pos_word=sp.pos_word, neg_word=sp.neg_word, probes=cmp_probes)
    return arms


def probe_limited_spec(spec, n_probe):
    if n_probe is None or len(spec.probes) <= n_probe:
        return spec
    return ReadoutSpec(name=spec.name, few_shot=spec.few_shot, template=spec.template,
                       pos_word=spec.pos_word, neg_word=spec.neg_word, probes=spec.probes[:n_probe])


def candidate_layers(results, arms):
    out = {}
    for name in TEST_ARMS:
        if name not in results["read"] or name not in results["steer"]:
            continue
        layers = sorted(results["read"][name]["sae"])
        rd, peak = phase_bc.read_depth_from(results["read"][name]["sae"], layers)
        if rd is None:
            print(f"[phase D] {name:>10}: read-depth UNDEFINED (peak {peak:.2f}); skipping", flush=True)
            continue
        cand = [L for L, row in sorted(results["steer"][name].items())
                if L >= rd and row.get("ci4_mean", [0.0, 1.0])[1] < 0]
        out[name] = {"read_depth": rd, "peak": peak, "candidates": cand,
                     "grid": list(range(max(0, rd - 2), min(rd + 2, 25) + 1))}
        print(f"[phase D] {name:>10}: read-depth L{rd} (peak {peak:.2f}); "
              f"coarse candidates {cand if cand else 'none'}", flush=True)
    return {k: v for k, v in out.items() if v["candidates"]}


def cache_fit_resids(model, arms, layers, arm_names):
    hooks = {f"blocks.{L}.hook_resid_post" for L in layers}
    cache_raw = {}
    tok_norms = {L: [] for L in layers}
    names = list(arms)                                      # match phase_bc.py's all-arm med_norm anchor
    for name in names:
        A = arms[name]
        for t in A["pos"] + A["neg"]:
            with torch.no_grad():
                _, ch = model.run_with_cache(model.to_tokens(t[:300]), names_filter=lambda n: n in hooks)
            entry = {L: ch[f"blocks.{L}.hook_resid_post"][0].float().cpu() for L in layers}
            cache_raw[(name, t)] = entry
            for L in layers:
                tok_norms[L] += entry[L].norm(dim=-1).tolist()
    med_norm = {L: float(np.median(tok_norms[L])) for L in layers}
    print("[phase D] cached resids; median per-token norms: " +
          " ".join(f"L{L}:{med_norm[L]:.0f}" for L in layers), flush=True)
    return cache_raw, med_norm


def raw_pair_diff(cache_raw, name, A, L):
    return (np.stack([cache_raw[(name, t)][L].mean(0).numpy() for t in A["pos"]])
            - np.stack([cache_raw[(name, t)][L].mean(0).numpy() for t in A["neg"]])).astype(np.float64)


def bootstrap_mean_ci(shifts, n_boot):
    shifts = np.asarray(shifts, dtype=float)
    boot = [np.mean(shifts[RNG.randint(0, len(shifts), len(shifts))]) for _ in range(n_boot)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def sign_test_p(shifts):
    shifts = np.asarray(shifts, dtype=float)
    neg = int(np.sum(shifts < 0)); pos = int(np.sum(shifts > 0))
    n = neg + pos
    if n == 0:
        return 1.0
    p = sum(float(math.comb(n, k)) for k in range(neg, n + 1)) / (2.0 ** n)
    return float(min(1.0, p))


def first_failed(row):
    order = ("probe_count", "dose_monotone_negative", "ci4_mean_negative", "robust_z",
             "negation_control", "sentiment_specificity")
    for k in order:
        if not row["criteria"].get(k, False):
            return k
    return None


def has_adjacent_confirmed(rows):
    good = sorted(int(L) for L, row in rows.items() if row.get("confirmed_layer"))
    return good, any(b == a + 1 for a, b in zip(good, good[1:]))


def save_checkpoint(results):
    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=2, default=float)


def measure_direction(fs, unit, base_scale, n_boot):
    fs._ensure_base()
    dose_mean, dose_median, per_probe = [], [], {}
    for m in STRENGTHS:
        pp = fs._dir_shift(unit * (base_scale * m))
        dose_mean.append(float(np.mean(pp)))
        dose_median.append(float(np.median(pp)))
        per_probe[str(int(m))] = [float(x) for x in pp]
    shifts4 = np.asarray(per_probe["4"], dtype=float)
    ci4 = bootstrap_mean_ci(shifts4, n_boot)
    return dose_mean, dose_median, per_probe, ci4, sign_test_p(shifts4)


def null_effects(fs, Vr, cluster, base_scale, n_null, seed):
    cl = np.asarray(cluster); cids = np.unique(cl)
    rs = np.random.RandomState(seed)
    nulls = []
    for _ in range(n_null):
        sgn = rs.choice([-1, 1], size=len(cids))[np.searchsorted(cids, cl)]
        nd = torch.tensor((Vr * sgn[:, None]).mean(0), dtype=torch.float32)
        nd = (nd / nd.norm()).to(fs._diffmeans.device)
        nulls.append(float(np.mean(fs._dir_shift(nd * (base_scale * STRENGTHS[-1])))))
    return nulls


def run_layer(model, arms, cache_raw, med_norm, alpha, arm_name, L, args):
    A0 = arms[arm_name]
    probe_shortfall = len(A0["spec"].probes) < 24
    A = dict(A0)
    A["spec"] = probe_limited_spec(A0["spec"], 8 if args.smoke else None)
    n_probe = len(A["spec"].probes)
    base_scale = alpha * med_norm[L]
    sae = SAE.from_pretrained("gemma-scope-2b-pt-res-canonical", f"layer_{L}/width_16k/canonical",
                              device=model.cfg.device)
    try:
        fs = FeatureScope(layer=L, concept=A["spec"], model=model, sae=sae).fit(A["pos"], A["neg"])
        fs._gen.manual_seed(0)
        du = fs._diffmeans / fs._diffmeans.norm()
        dose, dose_median, per_probe, ci4, sign_p = measure_direction(fs, du, base_scale, args.n_boot)

        Vr = raw_pair_diff(cache_raw, arm_name, A, L)
        nulls = null_effects(fs, Vr, A["cluster"], base_scale, args.n_null, 100 + L)
        z = _robust_z(float(np.mean(per_probe["4"])), np.array(nulls))

        neg_dose, neg_median, neg_probe, neg_ci4, neg_sign_p = measure_direction(
            fs, -du, base_scale * STRENGTHS[-1] / 4.0, args.n_boot)
        neg_sig_negative = neg_ci4[1] < 0

        sent_row = {"skipped": arm_name == "sentiment"}
        sent_sig_negative = False
        if arm_name != "sentiment":
            S = arms["sentiment"]
            sent_spec = probe_limited_spec(S["spec"], 8 if args.smoke else None)
            sent_fs = FeatureScope(layer=L, concept=sent_spec, model=model, sae=sae).fit(S["pos"], S["neg"])
            sent_fs._ensure_base()
            sent_shifts4 = sent_fs._dir_shift(du * (base_scale * STRENGTHS[-1]))
            sent_ci4 = bootstrap_mean_ci(sent_shifts4, args.n_boot)
            sent_p = sign_test_p(sent_shifts4)
            sent_sig_negative = sent_ci4[1] < 0
            sent_row = {"skipped": False, "ci4_mean": sent_ci4, "sign_p_negative": sent_p,
                        "mean4": float(np.mean(sent_shifts4)),
                        "n_probes": len(sent_spec.probes)}

        criteria = {
            "probe_count": n_probe >= 24 or probe_shortfall,
            "dose_monotone_negative": dose[1] < 0 and dose[2] < dose[1],
            "ci4_mean_negative": ci4[1] < 0,
            "robust_z": z <= -3,
            "negation_control": not neg_sig_negative,
            "sentiment_specificity": not sent_sig_negative,
        }
        row = {
            "layer": L, "base_scale": base_scale, "n_probes": n_probe,
            "probe_shortfall": probe_shortfall, "dose": dose, "dose_median": dose_median,
            "per_probe": per_probe,
            "mean4": float(np.mean(per_probe["4"])), "ci4_mean": ci4,
            "sign_p_negative": sign_p, "nulls": nulls, "z": float(z),
            "negation_control": {"dose": neg_dose, "dose_median": neg_median,
                                 "mean4": float(np.mean(neg_probe["4"])),
                                 "ci4_mean": neg_ci4, "sign_p_negative": neg_sign_p,
                                 "significantly_negative": neg_sig_negative},
            "sentiment_specificity": sent_row,
            "criteria": criteria,
        }
        row["confirmed_layer"] = all(criteria.values())
        row["first_failed_criterion"] = first_failed(row)
        return row
    finally:
        del sae
        if model.cfg.device == "mps":
            torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.n_null = 4 if args.smoke else 16
    args.n_boot = 100 if args.smoke else 400

    results_bc = phase_bc.normalize_loaded_results(json.load(open(RESULTS_IN)))
    arms = build_arms()
    candidates = candidate_layers(results_bc, arms)
    if not candidates:
        if not args.smoke:
            print("[phase D] no candidate backfire layers on the coarse grid; exiting cleanly", flush=True)
            return
        rd, peak = phase_bc.read_depth_from(results_bc["read"]["comparison"]["sae"],
                                            sorted(results_bc["read"]["comparison"]["sae"]))
        rd = rd or 12
        candidates = {"comparison": {"read_depth": rd, "peak": peak, "candidates": [],
                                     "grid": [max(0, rd - 2), max(0, rd - 1)]}}

    if args.smoke:
        name = next(iter(candidates), "comparison")
        if name not in candidates:
            rd, peak = phase_bc.read_depth_from(results_bc["read"][name]["sae"],
                                                sorted(results_bc["read"][name]["sae"]))
            candidates = {name: {"read_depth": rd or 12, "peak": peak, "candidates": [],
                                 "grid": [max(0, (rd or 12) - 2), max(0, (rd or 12) - 1)]}}
        else:
            candidates = {name: dict(candidates[name], grid=candidates[name]["grid"][:2])}
        print(f"[smoke] {name} only, layers {candidates[name]['grid']}, 8 probes, 4 nulls, n_boot=100",
              flush=True)

    layers = sorted({12} | {L for c in candidates.values() for L in c["grid"]})
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=dev, dtype=torch.float32)
    cache_raw, med_norm = cache_fit_resids(model, arms, layers, candidates.keys())
    alpha = 12.0 / med_norm[12]

    out = {"source": RESULTS_IN, "arms": {}, "med_norm": med_norm, "alpha": alpha,
           "smoke": bool(args.smoke)}
    for name, meta in candidates.items():
        out["arms"][name] = {k: meta[k] for k in ("read_depth", "peak", "candidates", "grid")}
        out["arms"][name]["layers"] = {}
        print(f"\n[phase D] {name}: densified grid {meta['grid']}", flush=True)
        for L in meta["grid"]:
            row = run_layer(model, arms, cache_raw, med_norm, alpha, name, L, args)
            out["arms"][name]["layers"][L] = row
            save_checkpoint(out)
            flag = " [probe<24]" if row["probe_shortfall"] else ""
            print(f"  [backfire L{L:>2}] {name:>10}: dose {[round(d, 2) for d in row['dose']]}  "
                  f"z {row['z']:5.1f}  ci4 [{row['ci4_mean'][0]:.2f},{row['ci4_mean'][1]:.2f}]  "
                  f"sign-p {row['sign_p_negative']:.3g}  "
                  f"{'PASS' if row['confirmed_layer'] else 'fail:' + row['first_failed_criterion']}{flag}",
                  flush=True)

    print("\n=== PHASE D BACKFIRE VERDICTS ===", flush=True)
    for name, arm_out in out["arms"].items():
        good, ok = has_adjacent_confirmed(arm_out["layers"])
        arm_out["confirmed_layers"] = good
        arm_out["confirmed"] = ok
        if ok:
            print(f"{name:>10}: CONFIRMED at layers {good}", flush=True)
        else:
            failures = [row["first_failed_criterion"] for _, row in sorted(arm_out["layers"].items())
                        if row["first_failed_criterion"]]
            first = failures[0] if failures else "adjacent_replication"
            print(f"{name:>10}: NOT CONFIRMED, first failed criterion: {first}", flush=True)
    save_checkpoint(out)
    print(f"\nsaved {RESULTS_OUT}", flush=True)


if __name__ == "__main__":
    main()
