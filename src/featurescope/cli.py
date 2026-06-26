"""featurescope CLI — run the driver/thermometer screen out-of-the-box, or on your own concept.

  featurescope                       # built-in sentiment demo (Gemma-2-2b, Gemma Scope L12)
  featurescope --top-k 20
  featurescope --csv mydata.csv      # columns: text,label   (label 1 = concept present, 0 = absent)
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
    ap.add_argument("--csv", type=str, default=None, help="text,label CSV for your own concept")
    args = ap.parse_args(argv)

    from .core import FeatureScope          # lazy: heavy imports only when actually running
    from . import data
    pos, neg = _load_csv(args.csv) if args.csv else (data.POS, data.NEG)

    fs = FeatureScope(layer=args.layer).fit(pos, neg).calibrate()
    fs.screen(top_k=args.top_k)
    fs.report()


if __name__ == "__main__":
    main()
