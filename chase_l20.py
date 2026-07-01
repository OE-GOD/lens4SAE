"""Chase the L20 puzzle: arithmetic's steer-z went strongly NEGATIVE at L20 (steering 'correct' pushed
the verdict toward 'incorrect') -- exactly where correctness reads cleanest. Real ('represent != control'
at the committed layer) or noise (16-example fluke)?

Test with more power + resolution: 40 equation-pairs, finer layers (16-22), and the bootstrap CI on the
driver's effect. CI entirely below 0 + a smooth trend => real backfire. CI straddling 0 / isolated spike => noise.
"""
import random, numpy as np, torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from featurescope import FeatureScope
from arith_steer import ARITH

random.seed(0)
POS, NEG = [], []
for _ in range(40):
    a, b = random.randint(2, 19), random.randint(2, 19)
    delta = random.choice([-2, -1, 1, 2])
    POS.append(f"{a} + {b} = {a + b}"); NEG.append(f"{a} + {b} = {a + b + delta}")

LAYERS = [16, 18, 19, 20, 21, 22]


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=dev, dtype=torch.bfloat16)
    print(f"\n{'layer':>6}{'dose 1x/2x/4x':>22}{'z':>8}{'  effect 95% CI':>18}   read")
    for L in LAYERS:
        sae = SAE.from_pretrained("gemma-scope-2b-pt-res-canonical", f"layer_{L}/width_16k/canonical", device=dev)
        fs = FeatureScope(layer=L, concept=ARITH, model=model, sae=sae).fit(POS, NEG)
        fs._gen.manual_seed(0)
        du = fs._diffmeans / fs._diffmeans.norm()
        dose, z, ci = fs._measure(du, 12.0, 1.0, n_null=16)
        # read: does the SAME direction separate correct/incorrect at this layer? (mean-pooled raw resid)
        y = np.array([1] * len(POS) + [0] * len(NEG))
        Xr = np.stack([r.numpy() for r in [
            (fs.model.run_with_cache(fs.model.to_tokens(t), names_filter=lambda n: n == fs.hook)[1][fs.hook][0].float().mean(0).cpu())
            for t in POS + NEG]])
        d = Xr[y == 1].mean(0) - Xr[y == 0].mean(0); proj = Xr @ d
        pos, neg = proj[y == 1], proj[y == 0]
        read = float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))
        verdict = "REAL(neg)" if ci[1] < 0 else ("noise/none" if ci[0] < 0 < ci[1] else "pos")
        print(f"{L:>6}{str([round(x,2) for x in dose]):>22}{z:>8.1f}  [{ci[0]:.2f},{ci[1]:.2f}]{read:>8.2f}   {verdict}")
        del sae
    print("\nCI entirely < 0 at L20 (+ smooth trend) => real backfire (represent != control). CI straddles 0 => noise.")


if __name__ == "__main__":
    main()
