"""FeatureScope core — driver/thermometer screening for SAE features (concept-general, v0.5).

A DRIVER is a feature the model *computes with*: steering it changes behaviour (read = write).
A THERMOMETER is decodable/correlated but causally inert: steering does nothing (read != write).
Only DRIVERS are safe to use as an RL reward / monitor; THERMOMETERS get gamed (Goodhart).

The honest, defensible use is a ONE-SIDED NEGATIVE SCREEN: cheaply *rule out* thermometers.
FeatureScope never certifies a feature "safe to optimize" — driver-ness is NECESSARY, not sufficient.

v0.5 — DOSE-RESPONSE (issue #1, pilot-driven). The pilot showed single-strength steering misleads:
features that move behaviour a little at low strength can SATURATE or REVERSE at high strength, and the
single-strength random null was often degenerate (zero spread -> z exploded). v0.5 fixes both:
  * measure a DOSE-RESPONSE: steer at 1x/2x/4x the feature's own units. A real driver's effect is
    SUSTAINED at high strength; a saturating/collapsing feature is not a safe driver.
  * measure the random-direction NULL at the HIGH strength, where random directions actually have
    spread -> no degenerate-null, and it controls off-manifold inflation.
  * decide on a unit-free robust z at the high strength (= SDs above the high-strength noise floor),
    so the rule transfers across concepts. SELF-TEST: a synthetic diff-of-means driver must clear the
    driver gate and a random null must not, else reporting is refused.
"""
from __future__ import annotations
import enum, numpy as np, torch
from dataclasses import dataclass
from .concepts import SENTIMENT, ReadoutSpec

STRENGTHS = (1.0, 2.0, 4.0)            # dose-response steering multiples of a feature's own units


class Verdict(enum.Enum):
    RULED_OUT = "ruled_out"            # high-strength effect indistinguishable from noise -> thermometer
    NOT_RULED_OUT = "not_ruled_out"    # sustained, significant causal effect -> driver-like (NOT certified)
    INDETERMINATE = "indeterminate"    # in between / saturating-collapsing / too noisy


@dataclass(frozen=True)
class FeatureResult:
    feature: int
    read: float
    dose: tuple              # cause at each steering strength (captures saturation)
    z: float                 # robust SDs above the high-strength random-noise floor (unit-free)
    frac: float              # high-strength effect as a fraction of the synthetic driver (anchor-relative)
    sustained: bool          # effect at max strength >= half its peak (didn't collapse)
    verdict: Verdict
    example: str = ""        # the input this feature fires hardest on -> what it detects (its label)
    necessity: float = float("nan")   # ablation: how much removing it drops the concept (> thresh = necessary)
    # deliberately NO `safe_to_optimize` field: the tool issues no safety certificate.


def _robust_z(cause, null):
    if null.size == 0:
        return float("nan")
    med = float(np.median(null))
    mad = float(np.median(np.abs(null - med)))
    if mad < 1e-9:
        return 0.0 if abs(cause - med) < 1e-6 else float(np.sign(cause - med) * 50.0)
    return float(np.clip((cause - med) / (1.4826 * mad), -50.0, 50.0))


class FeatureScope:
    def __init__(self, layer: int = 12, concept: ReadoutSpec = SENTIMENT, device: str | None = None,
                 sae_release: str = "gemma-scope-2b-pt-res", sae_id: str | None = None,
                 model=None, sae=None):
        from transformer_lens import HookedTransformer
        from sae_lens import SAE
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.layer = layer; self.concept = concept
        sae_id = sae_id or f"layer_{layer}/width_16k/average_l0_82"   # default L12 SAE; override for other layers
        self.sae = sae if sae is not None else SAE.from_pretrained(sae_release, sae_id, device=self.device)
        self.hook = self.sae.cfg.metadata["hook_name"]
        self.model = model if model is not None else HookedTransformer.from_pretrained("gemma-2-2b", device=self.device, dtype=torch.bfloat16)
        self.Wdec = self.sae.W_dec.detach().float()
        self.read = None; self._X = None; self._alive = None; self._diffmeans = None; self._base = None; self._texts = None
        self.drv_ref = None; self.z_drv = 3.0; self.z_out = 1.0; self.nec_thresh = 0.3; self.self_test = None; self.results = None
        self._rng = np.random.RandomState(0)
        self._gen = torch.Generator()                 # seeded RNG for random directions -> reproducible verdicts
        self._few = concept.few_shot
        self._pos = self.model.to_tokens(concept.pos_word, prepend_bos=False)[0, 0]
        self._neg = self.model.to_tokens(concept.neg_word, prepend_bos=False)[0, 0]

    # ---------- READ + difference-of-means direction (synthetic-driver anchor) ----------
    def _cache(self, text):
        with torch.no_grad():
            _, c = self.model.run_with_cache(self.model.to_tokens(text[:300]), names_filter=[self.hook])
            resid = c[self.hook][0]
            feats = self.sae.encode(resid).float()
            return feats.max(0).values.cpu().numpy(), resid.float().mean(0)

    def necessity(self, feature, k=6):
        """Necessity (ablation): remove the feature's contribution from the stream on texts where it fires
        hardest, and measure how much the concept readout drops. High = necessary; ~0 = redundant (other
        features carry it). Complements the steering (sufficiency) test -> the necessity x sufficiency 2x2."""
        assert self._X is not None and self._texts is not None, "call .fit() first"
        idx = np.argsort(-self._X[:, feature])[:k]
        sign = float(np.sign(self.read[feature]) or 1.0); Wf = self.Wdec[feature]
        def hook(resid, hook):
            f = self.sae.encode(resid)
            return resid - f[..., feature:feature + 1] * Wf          # subtract this feature's contribution
        drops = []
        for i in idx:
            toks = self.model.to_tokens(self._few + self.concept.template.format(text=self._texts[int(i)]))
            with torch.no_grad():
                base = self.model(toks)
                abl = self.model.run_with_hooks(toks, fwd_hooks=[(self.hook, hook)])
            b = (base[0, -1, self._pos] - base[0, -1, self._neg]).item()
            a = (abl[0, -1, self._pos] - abl[0, -1, self._neg]).item()
            drops.append((b - a) * sign)                             # >0 = removing it weakened the concept
        return float(np.median(drops))

    def fit(self, pos_texts, neg_texts):
        if not pos_texts or not neg_texts:
            raise ValueError("need both positive and negative examples")
        feats, resids = [], []
        for t in pos_texts + neg_texts:
            f, r = self._cache(t); feats.append(f); resids.append(r)
        self._X = np.stack(feats).astype(np.float64)
        self._texts = list(pos_texts) + list(neg_texts)         # kept so we can label each feature
        R = torch.stack(resids); n_pos = len(pos_texts)
        self._diffmeans = (R[:n_pos].mean(0) - R[n_pos:].mean(0))
        y = np.array([1] * n_pos + [0] * len(neg_texts), dtype=np.float64)
        yc = y - y.mean(); zc = self._X - self._X.mean(0)
        self.read = np.nan_to_num((zc * yc[:, None]).mean(0) / (zc.std(0) * yc.std() + 1e-9))
        self._alive = np.where((self._X > 0).mean(0) > 0.02)[0]
        return self

    # ---------- DOSE-RESPONSE measurement of a direction ----------
    def _readout(self, text, steer=None):
        hooks = [(self.hook, lambda r, hook: r + steer)] if steer is not None else []
        with torch.no_grad():
            lg = self.model.run_with_hooks(self.model.to_tokens(self._few + self.concept.template.format(text=text)),
                                           fwd_hooks=hooks)
        return (lg[0, -1, self._pos] - lg[0, -1, self._neg]).item()

    def _ensure_base(self):
        if self._base is None:
            self._base = np.array([self._readout(p) for p in self.concept.probes])

    def _dir_shift(self, vec):
        s = vec.to(self.model.cfg.dtype)
        return np.array([self._readout(p, steer=s) for p in self.concept.probes]) - self._base

    def _rand_unit(self, like):
        r = torch.randn(like.shape, generator=self._gen, dtype=torch.float32).to(like.device, like.dtype)
        return r / r.norm()                           # a proper random UNIT direction, from the seeded RNG

    def _measure(self, unit, base_scale, sign=1.0, n_null=8, n_boot=400):
        """Dose-response over STRENGTHS + a high-strength random-direction null. Returns (dose, z, ci)."""
        self._ensure_base()
        dose, last = [], None
        for m in STRENGTHS:
            pp = self._dir_shift(unit * (base_scale * m)) * sign
            dose.append(float(np.median(pp))); last = pp                      # last = max strength
        boot = [np.median(last[self._rng.randint(0, len(last), len(last))]) for _ in range(n_boot)]
        ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        hi = base_scale * STRENGTHS[-1]
        null = np.array([float(np.median(self._dir_shift(self._rand_unit(unit) * hi)))
                         for _ in range(n_null)])                             # null at HIGH strength (has spread)
        return dose, _robust_z(dose[-1], null), ci

    def _measure_feature(self, feature, n_null=8):
        Wf = self.Wdec[feature]; unit = Wf / Wf.norm()
        col = self._X[:, feature]; active = col[col > 0]
        if active.size < 5:
            return None
        base_scale = float(np.percentile(active, 95)) * float(Wf.norm())
        return self._measure(unit, base_scale, float(np.sign(self.read[feature]) or 1.0), n_null)

    def _top_example(self, feature):
        """The fit text this feature fires hardest on — a human-readable label for what it detects."""
        if self._texts is None:
            return ""
        return self._texts[int(np.argmax(self._X[:, feature]))]

    # ---------- CALIBRATE + SELF-TEST ----------
    def calibrate(self, anchor_norm=12.0, n_null=12):
        self._gen.manual_seed(0)                      # reproducible: same input -> same verdicts
        du = self._diffmeans / self._diffmeans.norm()
        dose_d, zd, _ = self._measure(du, anchor_norm, 1.0, n_null)
        self.drv_ref = max(dose_d) if max(dose_d) > 1e-6 else 1e-6
        ru = self._rand_unit(self._diffmeans)
        _, zr, _ = self._measure(ru, anchor_norm, 1.0, n_null)
        self.self_test = {"driver_z": zd, "null_z": zr, "passed": (zd >= self.z_drv and zr < self.z_drv)}
        print(f"[calibrate:{self.concept.name}] synthetic-driver z={zd:.2f}  random-null z={zr:.2f}  "
              f"(driver gate z>={self.z_drv}, rule-out z<={self.z_out}; dose-response @ {STRENGTHS})")
        print(f"[self-test] {'PASSED' if self.self_test['passed'] else 'FAILED'} "
              f"(synthetic driver must clear the driver gate at high strength; random null must not)")
        if not self.self_test["passed"]:
            print("[self-test] FAILED: cause-machinery can't validate this concept — labels will be REFUSED.")
        return self

    # ---------- SCREEN + REPORT ----------
    def _verdict(self, z, sustained, ci):
        if not np.isfinite(z):
            return Verdict.INDETERMINATE
        if z >= self.z_drv and sustained and ci[0] > 0:       # significant, sustained, direction-specific
            return Verdict.NOT_RULED_OUT
        if z <= self.z_out:                                   # high-strength effect within noise
            return Verdict.RULED_OUT
        return Verdict.INDETERMINATE                          # incl. strong-but-collapsing (saturating) features

    def _role(self, r):
        """The necessity x sufficiency 2x2 role for a result."""
        suff = r.verdict is Verdict.NOT_RULED_OUT
        nec = np.isfinite(r.necessity) and r.necessity > self.nec_thresh
        if suff and nec: return "lever (sufficient+necessary)"
        if suff:         return "redundant (sufficient only)"
        if nec:          return "circuit-part (necessary only)"
        return "bystander"

    def screen(self, top_k=8, n_null=8, with_necessity=True):
        assert self.read is not None and self.self_test is not None, "call .fit() then .calibrate() first"
        self._gen.manual_seed(1)                       # reproducible across reruns (distinct stream from calibrate)
        top = self._alive[np.argsort(-np.abs(self.read[self._alive]))[:top_k]]
        rows = []
        for f in top:
            m = self._measure_feature(int(f), n_null)
            if m is None:
                continue
            dose, z, ci = m
            peak = max(dose)
            sustained = peak > 1e-6 and dose[-1] >= 0.5 * peak
            frac = peak / self.drv_ref
            nec = self.necessity(int(f)) if with_necessity else float("nan")
            rows.append(FeatureResult(int(f), float(self.read[f]), tuple(round(d, 2) for d in dose),
                                      z, frac, sustained, self._verdict(z, sustained, ci),
                                      example=self._top_example(int(f))[:50], necessity=nec))
        self.results = sorted(rows, key=lambda r: -r.z)
        return self.results

    def ruled_out_thermometers(self):
        assert self.results is not None
        return [r.feature for r in self.results if r.verdict is Verdict.RULED_OUT]

    def to_dict(self):
        """Structured, JSON-serialisable results — so the verdicts can feed a downstream pipeline."""
        assert self.results is not None, "call .screen() first"
        by = lambda v: [r.feature for r in self.results if r.verdict is v]
        return {
            "concept": self.concept.name,
            "self_test": self.self_test,
            "drivers": by(Verdict.NOT_RULED_OUT),
            "ruled_out_thermometers": by(Verdict.RULED_OUT),
            "indeterminate": by(Verdict.INDETERMINATE),
            "features": [{"feature": r.feature, "read": round(r.read, 3), "dose": list(r.dose),
                          "z": round(r.z, 2), "frac": round(r.frac, 3), "sustained": r.sustained,
                          "verdict": r.verdict.value, "necessity": round(r.necessity, 3), "role": self._role(r),
                          "fires_on": r.example} for r in self.results],
        }

    def report(self):
        if not (self.self_test and self.self_test["passed"]):
            raise RuntimeError("self-test did not pass — refusing to report untrusted labels (run .calibrate())")
        assert self.results is not None, "call .screen() first"
        print(f"\nconcept: {self.concept.name}   (dose-response @ {STRENGTHS}; z=sufficiency, nec=necessity)")
        print(f"{'feature':>8}{'z':>7}{'nec':>7}   {'role (2x2)':<31} fires hardest on")
        for r in self.results:
            print(f"{r.feature:>8}{r.z:>7.1f}{r.necessity:>7.2f}   {self._role(r):<31} {r.example!r}")
        print(f"\nRULED OUT (do NOT use as rewards): {self.ruled_out_thermometers()}")
        print("Honest scope: ONE-SIDED NEGATIVE screen. Driver = SUSTAINED, significant dose-response (z vs a "
              "high-strength noise floor); saturating/collapsing features are INDETERMINATE, not drivers. "
              "'not_ruled_out' is NOT a safety certificate. z gates are cross-domain defaults pending "
              "leave-one-concept-out validation across more concepts; true gold = gaming under optimization (GPU).")
