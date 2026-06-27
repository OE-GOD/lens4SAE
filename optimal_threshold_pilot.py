"""Pilot toward an OPTIMAL driver/thermometer threshold (issue #1, Step 4 — laptop version).

Question: is the cheap single-strength robust-z a good predictor of a STRICTER causal gold, what
z-threshold is optimal, and does one threshold TRANSFER across concepts?

  SCORE  = robust z (single-strength steering vs random-direction null)  -- the cheap statistic we ship.
  GOLD   = dose-response sufficiency: steer at 1x/2x/4x own-units; label DRIVER iff the effect is
           monotonic in strength AND reaches >=50% of the synthetic-driver's effect at 4x.
           (A stricter, anchor-relative, multi-strength test than the single-strength z.)

Honest limits: GOLD is still STEERING-based (sufficiency), not the true gold (does it get gamed under
real optimization = the GPU step). Only 2 concepts here (real leave-one-concept-out needs >=4). Few
features -> noisy ROC. This is a methodology pilot, not the final optimal gate.
"""
import numpy as np
from featurescope import FeatureScope
from featurescope.concepts import SENTIMENT, FORMALITY
from featurescope import data

TOPK, STRENGTHS, N_NULL = 16, [1.0, 2.0, 4.0], 12


def auroc(scores, labels):
    s, y = np.array(scores, float), np.array(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def youden(scores, labels):
    s, y = np.array(scores, float), np.array(labels, int)
    best = (-2.0, None)
    for t in np.unique(s):
        pred = s >= t
        tpr = pred[y == 1].mean() if (y == 1).any() else 0.0
        fpr = pred[y == 0].mean() if (y == 0).any() else 0.0
        if tpr - fpr > best[0]:
            best = (tpr - fpr, float(t))
    return best[1], best[0]


def run_concept(name, spec, pos, neg):
    fs = FeatureScope(concept=spec).fit(pos, neg)
    du = fs._diffmeans / fs._diffmeans.norm()
    drv4 = fs._cause_dir(du, 12.0 * 4.0, 1.0, n_null=0)[0]            # synthetic-driver effect at 4x (gold ref)
    top = fs._alive[np.argsort(-np.abs(fs.read[fs._alive]))[:TOPK]]
    rows = []
    for f in top:
        c = fs._cause_feature(int(f), n_null=N_NULL)
        if c is None:
            continue
        z = c[2]
        Wf = fs.Wdec[int(f)]; unit = Wf / Wf.norm()
        col = fs._X[:, int(f)]; a_hi = float(np.percentile(col[col > 0], 95))
        sign = float(np.sign(fs.read[int(f)]) or 1.0)
        dose = [fs._cause_dir(unit, a_hi * float(Wf.norm()) * m, sign, n_null=0)[0] for m in STRENGTHS]
        gold = int(dose[-1] >= dose[0] and dose[-1] >= 0.5 * drv4)    # monotone & strong vs driver anchor
        rows.append({"f": int(f), "z": z, "dose": dose, "gold": gold})
    return rows


def main():
    concepts = {"sentiment": (SENTIMENT, data.examples_for("sentiment")),
                "formality": (FORMALITY, data.examples_for("formality"))}
    res = {}
    for name, (spec, (pos, neg)) in concepts.items():
        rows = run_concept(name, spec, pos, neg)
        res[name] = rows
        zs = [r["z"] for r in rows]; gs = [r["gold"] for r in rows]
        a = auroc(zs, gs); thr, j = youden(zs, gs)
        print(f"\n[{name}] n={len(rows)} gold-drivers={sum(gs)}  AUROC(z->gold)={a:.2f}  "
              f"optimal z* (Youden)={thr}  J={j:.2f}   (shipped gates: driver z>=3.0, rule-out z<=1.0)")
        for r in sorted(rows, key=lambda r: -r["z"]):
            print(f"   feat {r['f']:>6}  z={r['z']:>6.1f}  dose={[round(d,2) for d in r['dose']]}  gold={'DRIVER' if r['gold'] else 'thermo'}")

    print("\n--- cross-concept transfer (does one optimal threshold hold on the other concept?) ---")
    for tr in res:
        thr, _ = youden([r["z"] for r in res[tr]], [r["gold"] for r in res[tr]])
        if thr is None:
            continue
        for te in res:
            if te == tr:
                continue
            zt = np.array([r["z"] for r in res[te]]); gt = np.array([r["gold"] for r in res[te]])
            acc = float(np.mean((zt >= thr).astype(int) == gt))
            print(f"   threshold from {tr} (z*={thr:.1f}) -> {te}: accuracy={acc:.2f}")
    print("\nNote: gold is steering-based (sufficiency), 2 concepts only, small n -> pilot, not the final gate. "
          "True gold = gaming under real optimization (GPU).")


if __name__ == "__main__":
    main()
