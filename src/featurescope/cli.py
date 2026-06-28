"""featurescope CLI — run the driver/thermometer screen out-of-the-box, or on your own concept.

  featurescope                                 # built-in sentiment demo (Gemma-2-2b, Gemma Scope L12)
  featurescope --concept formality             # built-in formality demo
  featurescope --concept formality --csv mydata.csv --top-k 20
  featurescope --csv mydata.csv                # columns: text,label (1 = concept present, 0 = absent)
"""
import argparse, csv


def _load_csv(path):
    pos, neg = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            (pos if str(row["label"]).strip() in ("1", "pos", "positive", "true") else neg).append(row["text"])
    return pos, neg


def main(argv=None):
    ap = argparse.ArgumentParser(description="Screen SAE features as drivers vs thermometers.")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--concept", type=str, default="sentiment", help="built-in concept (sentiment | formality)")
    ap.add_argument("--csv", type=str, default=None, help="text,label CSV (data for the chosen concept's readout)")
    ap.add_argument("--output", type=str, default=None, help="write structured results to a JSON file")
    args = ap.parse_args(argv)

    from .core import FeatureScope          # lazy: heavy imports only when actually running
    from . import data
    from .concepts import REGISTRY
    if args.concept not in REGISTRY:
        ap.error(f"unknown concept '{args.concept}' (have: {', '.join(REGISTRY)})")
    spec = REGISTRY[args.concept]
    pos, neg = _load_csv(args.csv) if args.csv else data.examples_for(args.concept)

    fs = FeatureScope(layer=args.layer, concept=spec).fit(pos, neg).calibrate()
    fs.screen(top_k=args.top_k)
    fs.report()
    if args.output:
        import json
        with open(args.output, "w") as fh:
            json.dump(fs.to_dict(), fh, indent=2)
        print(f"\nwrote structured results to {args.output}")


if __name__ == "__main__":
    main()
