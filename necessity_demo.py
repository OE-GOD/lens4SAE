"""Demo: the necessity x sufficiency 2x2. Sufficiency = steering (does adding it work?); necessity =
ablation (does removing it break it?). Most drivers are expected sufficient-but-NOT-necessary (redundancy)."""
from featurescope import FeatureScope
from featurescope.concepts import SENTIMENT
from featurescope import data


def main():
    pos, neg = data.examples_for("sentiment")
    fs = FeatureScope(concept=SENTIMENT).fit(pos, neg).calibrate()
    fs.screen(top_k=8)
    print("\nfeature   z(suff)   verdict        necessity   2x2")
    for r in fs.results:
        nec = fs.necessity(r.feature)
        suff = r.verdict.value == "not_ruled_out"
        necc = nec > 0.3                                  # rough bar: removing it clearly weakens the concept
        box = ("sufficient+necessary (THE lever)" if suff and necc else
               "sufficient, NOT necessary (redundant)" if suff and not necc else
               "necessary, not sufficient (circuit part)" if necc else
               "neither (bystander)")
        print(f"{r.feature:>7}{r.z:>9.1f}   {r.verdict.value:<14}{nec:>9.2f}   {box}")
    print("\nReading: sufficiency = can adding it CAUSE the concept; necessity = does removing it BREAK it.")
    print("Lots of 'sufficient, not necessary' = the concept is spread across redundant features (hydra effect).")


if __name__ == "__main__":
    main()
