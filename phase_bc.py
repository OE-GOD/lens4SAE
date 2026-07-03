"""Phase B (read sweep) + Phase C (steer sweep) under the corrected instrument.

Corrections vs the old pipeline (all from the adversarial design review):
  READ  — leave-one-cluster-out CV (never in-sample); primary statistic = fraction of held-out minimal
          pairs correctly ordered, with a cluster-sign permutation p; LOPO AUROC secondary; embed-layer
          leakage gate; matched raw-resid cross-check; unigram harness check; label-shuffle control.
  STEER — dose scaled per layer (alpha x median per-token resid norm, alpha anchored at L12 = old
          norm-12); pass gate = beat all permutation nulls + robust z >= 3 + sustained effect;
          fixed-norm-12 row kept only as a secondary robustness line;
          steer-depth = shallowest passing layer (first-crossing, same convention as read-depth).

Run `--smoke` first (~10 min): 1 concept, 2 layers, truncated items/probes/nulls, all code paths.
"""
import argparse, json, os, sys
import numpy as np, torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from featurescope import FeatureScope
from featurescope.core import _robust_z
import computed_concepts as cc

LAYERS = [0, 4, 8, 12, 16, 20, 24]
STRENGTHS = (1.0, 2.0, 4.0)
RNG = np.random.RandomState(0)


def auroc(s, y):
    s = np.asarray(s, dtype=float); y = np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


# ---------------- Phase B machinery: LOPO reads on pair-difference vectors ----------------
def lopo_stats(V, cluster):
    """V: [K, d] minimal-pair difference vectors. Returns (frac_ordered, lopo_auroc, t_scores)."""
    K = len(V); cluster = np.asarray(cluster)
    G = V @ V.T
    mask = cluster[:, None] != cluster[None, :]              # exclude own cluster from the refit
    t = (G * mask).sum(1)
    frac = float(np.mean(t > 0) + 0.5 * np.mean(t == 0))
    # secondary: pooled LOPO AUROC via per-fold direction projections of pos and neg separately
    sumV = V.sum(0)
    projs, ys = [], []
    for i in range(K):
        d = sumV - V[cluster == cluster[i]].sum(0)
        projs += [float(V[i] @ d)]                            # (pos - neg) proj > 0 iff ordered
    a = float(np.mean(np.array(projs) > 0) + 0.5 * np.mean(np.array(projs) == 0))
    return frac, a, t


def perm_p(V, cluster, frac_real, n_perm=4000):
    """cluster-sign permutation p for the ordered-fraction statistic."""
    cluster = np.asarray(cluster); cids = np.unique(cluster)
    G = V @ V.T; mask = cluster[:, None] != cluster[None, :]
    if len(cids) <= 12:                                       # exact enumeration
        configs = [(np.array([(c >> k) & 1 for k in range(len(cids))]) * 2 - 1) for c in range(2 ** len(cids))]
    else:
        configs = [RNG.choice([-1, 1], size=len(cids)) for _ in range(n_perm)]
    hits = 0
    for s in configs:
        sig = s[np.searchsorted(cids, cluster)]
        t = sig * ((G * mask) @ sig)
        f = float(np.mean(t > 0) + 0.5 * np.mean(t == 0))
        hits += (f >= frac_real)
    return (1 + hits) / (1 + len(configs))


def cluster_boot_ci(V, cluster, n_boot=2000):
    cluster = np.asarray(cluster); cids = np.unique(cluster)
    G = V @ V.T; mask = cluster[:, None] != cluster[None, :]
    t = (G * mask).sum(1); win = (t > 0).astype(float) + 0.5 * (t == 0)
    per_cl = [win[cluster == c].mean() for c in cids]
    boots = [np.mean([per_cl[i] for i in RNG.randint(0, len(per_cl), len(per_cl))]) for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def read_depth_from(aurocs, layers):
    peak = max(aurocs.values())
    if peak < 0.80:
        return None, peak
    thr = 0.5 + 0.95 * (peak - 0.5)
    return next(L for L in layers if aurocs[L] >= thr), peak


def normalize_loaded_results(results):
    results["med_norm"] = {int(L): v for L, v in results["med_norm"].items()}
    for name in results["read"]:
        R = results["read"][name]
        for k in ("sae", "raw", "frac", "perm_p", "boot_ci"):
            R[k] = {int(L): v for L, v in R[k].items()}
    for name in results["steer"]:
        results["steer"][name] = {int(L): v for L, v in results["steer"][name].items()}
    return results


def controls_complete(results, arms, layers):
    controls = results.get("controls", {})
    mid = layers[len(layers) // 2]
    expected = ["unigram_L0_auroc"] + [f"shuffled_{name}_L{mid}" for name in arms]
    return all(k in controls for k in expected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--mixed", action="store_true",
                    help="review-sanctioned fallback: bf16 trunk + fp32 readout head (validated vs banked L12 cell)")
    args = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=dev,
                                              dtype=torch.bfloat16 if args.mixed else torch.float32)
    if args.mixed:
        import mixed_head
        print("[mode] MIXED precision: bf16 trunk + fp32 readout head (see PROTOCOL_NOTES.md)", flush=True)

    fails = cc.check_all(model)
    if fails:
        print("DATASET ASSERTIONS FAILED — aborting run:")
        for f in fails:
            print("  -", f)
        raise SystemExit(1)
    print("[assertions] all dataset invariants hold (token multisets, vocab disjointness, no trailing '.')")

    a1 = json.load(open("phase_a_results.json")); a2 = json.load(open("phase_a2_results.json"))
    few = cc.CMP_FEWSHOT_4 if a1["comparison"]["few_shot"] == "4shot" else cc.CMP_FEWSHOT_2
    cmp_probes = a2["comparison_probes"]["true"] + a2["comparison_probes"]["false"]
    cap_probes = cc.cap_probe_texts([tuple(x) for x in a2["capitals_probes"]["true"]],
                                    [tuple(x) for x in a2["capitals_probes"]["false"]])
    if a2["anchor_choice"] == "FAIL":
        print("ANCHOR GATE FAILED at Phase A2 — instrument failure per prereg. Stopping."); sys.exit(1)
    from phase_a2 import small_sum_arith
    from arith_steer import ARITH_POS, ARITH_NEG, ARITH
    ar_pos, ar_neg = (small_sum_arith() if a2["anchor_choice"] == "smallsum40" else (ARITH_POS, ARITH_NEG))
    s_pos, s_neg, s_spec = cc.sentiment_sets_and_spec()

    ARMS = {
        "comparison": dict(spec=cc.cmp_spec(few), pos=cc.CMP_POS, neg=cc.CMP_NEG, cluster=cc.CMP_CLUSTER, vote=True),
        "capitals": dict(spec=cc.cap_spec(cap_probes), pos=cc.CAP_POS, neg=cc.CAP_NEG, cluster=cc.CAP_CLUSTER, vote=True),
        "parity": dict(spec=cc.par_spec(), pos=cc.PAR_POS, neg=cc.PAR_NEG, cluster=cc.PAR_CLUSTER, vote=True),
        "arith": dict(spec=ARITH, pos=ar_pos, neg=ar_neg, cluster=list(range(len(ar_pos))), vote=False),
        "sentiment": dict(spec=s_spec, pos=s_pos, neg=s_neg, cluster=list(range(len(s_pos))), vote=False),
    }
    # comparison probes live on the spec; rebuild spec with vetted probes explicitly
    from featurescope.concepts import ReadoutSpec
    sp = ARMS["comparison"]["spec"]
    ARMS["comparison"]["spec"] = ReadoutSpec(name=sp.name, few_shot=sp.few_shot, template=sp.template,
                                             pos_word=sp.pos_word, neg_word=sp.neg_word, probes=cmp_probes)

    layers = LAYERS
    if args.smoke:
        layers = [8, 16]
        ARMS = {"comparison": ARMS["comparison"]}
        a = ARMS["comparison"]
        keep = [i for i, c in enumerate(a["cluster"]) if c < 4]
        a["pos"] = [a["pos"][i] for i in keep]; a["neg"] = [a["neg"][i] for i in keep]
        a["cluster"] = [a["cluster"][i] for i in keep]
        sp = a["spec"]
        a["spec"] = ReadoutSpec(name=sp.name, few_shot=sp.few_shot, template=sp.template,
                                pos_word=sp.pos_word, neg_word=sp.neg_word,
                                probes=sp.probes[:4] + sp.probes[len(sp.probes) // 2:len(sp.probes) // 2 + 4])
        print("[smoke] comparison only, layers [8,16], 4 clusters, 8 probes, 4 nulls")
    n_null = 4 if args.smoke else 16

    # ============ PHASE B: cache resids, LOPO reads ============
    hooks = {f"blocks.{L}.hook_resid_post" for L in layers} | {"hook_embed"}
    cache_raw = {}                                            # (arm, text) -> {layer: [T,d] resid, 'embed': [T,d]}
    tok_norms = {L: [] for L in layers}
    for name, A in ARMS.items():
        for t in A["pos"] + A["neg"]:
            with torch.no_grad():
                _, ch = model.run_with_cache(model.to_tokens(t[:300]), names_filter=lambda n: n in hooks)
            entry = {L: ch[f"blocks.{L}.hook_resid_post"][0].float().cpu() for L in layers}
            entry["embed"] = ch["hook_embed"][0].float().cpu()
            cache_raw[(name, t)] = entry
            for L in layers:
                tok_norms[L] += entry[L].norm(dim=-1).tolist()
    med_norm = {L: float(np.median(tok_norms[L])) for L in layers}
    print("[phase B] cached resids; median per-token norms: " +
          " ".join(f"L{L}:{med_norm[L]:.0f}" for L in layers), flush=True)

    if args.resume and os.path.exists("phase_bc_results.json"):
        results = normalize_loaded_results(json.load(open("phase_bc_results.json")))
        for L in sorted(set(med_norm) & set(results["med_norm"])):
            tol = 0.02 * abs(results["med_norm"][L]) if args.mixed else 1e-6   # bf16 resids shift norms ~0.5%
            if abs(med_norm[L] - results["med_norm"][L]) > tol:
                raise ValueError(
                    f"Resume med_norm mismatch at L{L}: fresh {med_norm[L]:.12g} vs checkpoint "
                    f"{results['med_norm'][L]:.12g}. Refusing to resume because this risks dose-scale "
                    "mixing; resumed runs must not silently merge incompatible normalization scales."
                )
    else:
        results = {"med_norm": med_norm, "read": {}, "steer": {}, "controls": {}}
    if "controls" not in results:
        results["controls"] = {}
    controls_done = args.resume and controls_complete(results, ARMS, layers)
    raw_pair_diff = {}                                        # (arm, layer) -> V_raw for steering nulls
    for name, A in ARMS.items():
        if name not in results["read"]:
            results["read"][name] = {"sae": {}, "raw": {}, "frac": {}, "perm_p": {}, "boot_ci": {}}
        for k in ("sae", "raw", "frac", "perm_p", "boot_ci"):
            if k not in results["read"][name]:
                results["read"][name][k] = {}
    have_embed = args.resume and all("embed_auroc" in results["read"][name] for name in ARMS)
    if not have_embed:
        # embed-layer leakage gate (raw mean-pooled)
        for name, A in ARMS.items():
            Vp = np.stack([cache_raw[(name, t)]["embed"].mean(0).numpy() for t in A["pos"]])
            Vn = np.stack([cache_raw[(name, t)]["embed"].mean(0).numpy() for t in A["neg"]])
            frac_e, auroc_e, _ = lopo_stats((Vp - Vn).astype(np.float64), A["cluster"])
            results["read"][name]["embed_auroc"] = auroc_e
    for L in layers:
        for name, A in ARMS.items():
            Vr = (np.stack([cache_raw[(name, t)][L].mean(0).numpy() for t in A["pos"]])
                  - np.stack([cache_raw[(name, t)][L].mean(0).numpy() for t in A["neg"]])).astype(np.float64)
            raw_pair_diff[(name, L)] = Vr
        read_done = args.resume and all(L in results["read"][name]["sae"] for name in ARMS)
        sae = None
        if not read_done:
            sae = SAE.from_pretrained("gemma-scope-2b-pt-res-canonical", f"layer_{L}/width_16k/canonical", device=dev)
            for name, A in ARMS.items():
                def enc(t):
                    return sae.encode(cache_raw[(name, t)][L].to(dev)).detach().max(0).values.float().cpu().numpy()
                V = np.stack([enc(p) for p in A["pos"]]) - np.stack([enc(n) for n in A["neg"]])
                V = V.astype(np.float64)
                frac, a, _ = lopo_stats(V, A["cluster"])
                R = results["read"][name]
                R["sae"][L] = a; R["frac"][L] = frac
                R["perm_p"][L] = perm_p(V, A["cluster"], frac, n_perm=500 if args.smoke else 4000)
                R["boot_ci"][L] = cluster_boot_ci(V, A["cluster"], n_boot=300 if args.smoke else 2000)
                # matched raw-resid (mean-pooled) cross-check
                Vr = raw_pair_diff[(name, L)]
                R["raw"][L] = lopo_stats(Vr, A["cluster"])[1]
                print(f"  [read L{L:>2}] {name:>10}: LOPO-AUROC {a:.2f} (frac {frac:.2f}, p {R['perm_p'][L]:.3f}, "
                      f"raw {R['raw'][L]:.2f})", flush=True)
        if sae is not None:
            del sae
            if dev == "mps":
                torch.mps.empty_cache()

    # unigram harness check at L0: 'greater' vs 'less' must be near-perfectly readable
    if "comparison" in ARMS and "unigram_L0_auroc" not in results["controls"]:
        sae = SAE.from_pretrained("gemma-scope-2b-pt-res-canonical", "layer_0/width_16k/canonical", device=dev)
        A = ARMS["comparison"]
        g = [t for t in A["pos"] + A["neg"] if "greater" in t]
        l = [t for t in A["pos"] + A["neg"] if "less" in t]
        def encc(t):
            return sae.encode(cache_raw[("comparison", t)][0].to(dev)).detach().max(0).values.float().cpu().numpy()
        Vg = np.stack([encc(t) for t in g[:len(l)]]) - np.stack([encc(t) for t in l[:len(g)]])
        hfrac, hauroc, _ = lopo_stats(Vg.astype(np.float64), list(range(len(Vg))))
        results["controls"]["unigram_L0_auroc"] = hauroc
        print(f"  [harness] unigram 'greater vs less' @L0: LOPO-AUROC {hauroc:.2f} (must be ~1.0)", flush=True)
        del sae
        if dev == "mps":
            torch.mps.empty_cache()
    controls_done = args.resume and controls_complete(results, ARMS, layers)

    # label-shuffle negative control (one seeded cluster-sign flip, full pipeline, SAE read at mid layer)
    if not controls_done:
        for name, A in ARMS.items():
            cl = np.asarray(A["cluster"]); cids = np.unique(cl)
            s = np.random.RandomState(7).choice([-1, 1], size=len(cids))
            sig = s[np.searchsorted(cids, cl)]
            L = layers[len(layers) // 2]
            Vr = raw_pair_diff[(name, L)] * sig[:, None]
            results["controls"][f"shuffled_{name}_L{L}"] = lopo_stats(Vr, A["cluster"])[1]

    with open("phase_bc_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("[phase B] done; checkpoint saved", flush=True)

    # ============ PHASE C: steer sweep ============
    alpha = 12.0 / med_norm[12] if 12 in [*med_norm] else 12.0 / med_norm[layers[len(layers) // 2]]
    for name, A in ARMS.items():
        if name not in results["steer"]:
            results["steer"][name] = {}
    for L in layers:
        steer_done = args.resume and all(L in results["steer"][name] for name in ARMS)
        if steer_done:
            continue
        sae = SAE.from_pretrained("gemma-scope-2b-pt-res-canonical", f"layer_{L}/width_16k/canonical", device=dev)
        base_scale = alpha * med_norm[L]
        for name, A in ARMS.items():
            if args.resume and L in results["steer"][name]:
                continue
            fs = FeatureScope(layer=L, concept=A["spec"], model=model, sae=sae).fit(A["pos"], A["neg"])
            if args.mixed:
                mixed_head.attach(fs)
                results.setdefault("mode_notes", []).append(f"steer {name} L{L}: mixed(bf16 trunk+fp32 head)")
            fs._gen.manual_seed(0)
            du = fs._diffmeans / fs._diffmeans.norm()
            fs._ensure_base()
            dose, shifts4 = [], None
            for m in STRENGTHS:
                pp = fs._dir_shift(du * (base_scale * m))
                dose.append(float(np.median(pp))); shifts4 = pp
            fixed = [float(np.median(fs._dir_shift(du * (12.0 * m)))) for m in STRENGTHS]
            # permutation-null directions: cluster-sign refits of the raw pair-diff mean.
            # Adaptive: 16-null screen, escalate to 32 only for candidate layers (fleet cost lever).
            cl = np.asarray(A["cluster"]); cids = np.unique(cl)
            Vr = raw_pair_diff[(name, L)]
            rs = np.random.RandomState(100 + L)
            def null_effect():
                sgn = rs.choice([-1, 1], size=len(cids))[np.searchsorted(cids, cl)]
                nd = torch.tensor((Vr * sgn[:, None]).mean(0), dtype=torch.float32)
                nd = (nd / nd.norm()).to(fs._diffmeans.device)
                return float(np.median(fs._dir_shift(nd * (base_scale * STRENGTHS[-1]))))
            nulls = [null_effect() for _ in range(n_null)]
            z = _robust_z(dose[-1], np.array(nulls))
            if dose[-1] > max(nulls) and z >= 2.5:            # candidate: escalate for a stable z
                nulls += [null_effect() for _ in range(n_null)]
                z = _robust_z(dose[-1], np.array(nulls))
            p = (1 + sum(nv >= dose[-1] for nv in nulls)) / (1 + len(nulls))
            peak = max(dose)
            sustained = peak > 1e-6 and dose[-1] >= 0.5 * peak
            boot = [np.mean(shifts4[RNG.randint(0, len(shifts4), len(shifts4))]) for _ in range(400)]
            ci4 = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
            results["steer"][name][L] = {"dose": dose, "fixed12": fixed, "rank_p": p, "z": float(z),
                                         "sustained": sustained, "ci4_mean": ci4,
                                         "shifts4": [float(x) for x in shifts4],
                                         "nulls": nulls, "base_scale": base_scale}
            print(f"  [steer L{L:>2}] {name:>10}: dose {[round(d,2) for d in dose]}  z {z:5.1f}  p {p:.3f}  "
                  f"sust {sustained}  ci4 [{ci4[0]:.2f},{ci4[1]:.2f}]  (fixed12 {[round(d,2) for d in fixed]})", flush=True)
            with open("phase_bc_results.json", "w") as f:      # per-cell checkpoint: a kill costs one cell, not a layer
                json.dump(results, f, indent=2, default=float)
        del sae
        if dev == "mps":
            torch.mps.empty_cache()
        with open("phase_bc_results.json", "w") as f:
            json.dump(results, f, indent=2, default=float)

    # ============ verdicts per prereg ============
    print("\n=== PRE-REGISTERED VERDICTS ===")
    for name, A in ARMS.items():
        R = results["read"][name]; S = results["steer"][name]
        rd, peak = read_depth_from(R["sae"], layers)
        # steer pass (protocol correction 1): beat ALL permutation nulls AND robust z >= 3 AND sustained.
        # (rank-p+Holm was unsatisfiable at any feasible null count: min p 1/33 > 0.05/7; z>=3 is the
        #  stricter-than-Holm feasible analog and is recorded alongside the descriptive rank-p.)
        passing = [L for L in layers
                   if S[L]["dose"][-1] > max(S[L]["nulls"]) and S[L]["z"] >= 3 and S[L]["sustained"]
                   and S[L]["dose"][-1] > 0]
        sd = passing[0] if passing else None
        leaky = R["sae"].get(0, 0.5) > 0.65 or R["embed_auroc"] > 0.6
        peak_layer = max(R["sae"], key=lambda k: R["sae"][k])
        defined = rd is not None and R["perm_p"].get(peak_layer, 1) < 0.05 and not leaky
        rev = (sd is not None and rd is not None and sd <= rd - 4)
        print(f"{name:>10}: read-depth {('L' + str(rd)) if rd else 'UNDEFINED'} (peak {peak:.2f}, "
              f"embed {R['embed_auroc']:.2f}{', LEAKY' if leaky else ''})  "
              f"steer-depth {('L' + str(sd)) if sd is not None else 'none'}  "
              f"reversal {'YES' if rev else 'no'}{'' if defined else '  [gates not met]'}")
    with open("phase_bc_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nsaved phase_bc_results.json — Phase D (confirmatory backfire) runs only on qualifying concepts.")


if __name__ == "__main__":
    main()
