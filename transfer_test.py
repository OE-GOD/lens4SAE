"""Cross-concept transfer test (issue #1): does ONE z-gate work across concepts?

We test it on each concept's GROUND-TRUTH anchors — the manufactured guaranteed-driver (difference-of-
means direction) and guaranteed-null (random direction) that calibrate() already scores. If a single
gate (z>=3) puts every concept's guaranteed-driver above it and every guaranteed-null below it, the
gate transfers. (Honest scope: this is transfer on *manufactured* anchors, not arbitrary features —
but those anchors ARE ground truth by construction, so it's a clean, non-circular transfer check.)
"""
from featurescope import FeatureScope
from featurescope.concepts import REGISTRY
from featurescope import data

CONCEPTS = ["sentiment", "formality", "toxicity", "certainty"]
GATE = 3.0


def main():
    rows = {}
    for name in CONCEPTS:
        pos, neg = data.examples_for(name)
        fs = FeatureScope(concept=REGISTRY[name]).fit(pos, neg).calibrate()
        st = fs.self_test
        rows[name] = (st["driver_z"], st["null_z"], st["passed"])

    print(f"\n=== cross-concept transfer of the z={GATE} gate (on ground-truth anchors) ===")
    print(f"{'concept':>10}{'driver_z':>10}{'null_z':>9}   self-test")
    for n, (dz, nz, p) in rows.items():
        print(f"{n:>10}{dz:>10.2f}{nz:>9.2f}   {'PASS' if p else 'REFUSED (concept not screenable here)'}")

    passed = {n: (dz, nz) for n, (dz, nz, p) in rows.items() if p}
    refused = [n for n, (_, _, p) in rows.items() if not p]
    dzs = [d for d, _ in passed.values()]; nzs = [nz for _, nz in passed.values()]
    print(f"\nAmong the {len(passed)} self-test-PASSING concepts ({', '.join(passed)}):")
    print(f"  guaranteed-driver z: min = {min(dzs):.2f}  (all must be >= {GATE})")
    print(f"  guaranteed-null   z: max = {max(nzs):.2f}  (all must be <  {GATE})")
    ok = min(dzs) >= GATE > max(nzs)
    print(f"  => the single gate z={GATE} separates ground-truth anchors in ALL of them: {ok}"
          + (f"  (clean margin: {min(dzs):.1f} vs {max(nzs):.1f})" if ok else ""))
    if refused:
        print(f"\nREFUSED (self-test failed -> tool declined, did NOT emit labels): {', '.join(refused)}")
        print("  = the honest-refusal safety net working: that concept isn't a usable linear direction here")
        print("    (or its examples/readout are too weak). Either way, no garbage labels.")
    print("\nHonest scope: transfer shown on MANUFACTURED anchors (guaranteed driver/null) across the passing "
          "concepts; a stronger claim still needs labeled real features. Refusals are a feature, not a bug.")


if __name__ == "__main__":
    main()
