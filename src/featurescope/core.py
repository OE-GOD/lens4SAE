"""FeatureScope core — driver/thermometer screening for SAE features (concept-general, v0.4).

A DRIVER is a feature the model *computes with*: steering it changes behaviour (read = write).
A THERMOMETER is decodable/correlated but causally inert: steering does nothing (read != write).
Only DRIVERS are safe to use as an RL reward / monitor; THERMOMETERS get gamed (Goodhart).

The honest, defensible use is a ONE-SIDED NEGATIVE SCREEN: cheaply *rule out* thermometers.
FeatureScope never certifies a feature "safe to optimize" — driver-ness is NECESSARY, not sufficient.

v0.4 — CROSS-DOMAIN decision (issue #1). The verdict is decided on a DIMENSIONLESS quantity so one
rule transfers across concepts on different readout scales: each feature's cause is standardised against
a real per-feature random-direction NULL via a robust z (median / MAD) — "how many SDs above this
direction's own noise floor." Raw cause units are never thresholded.
  * cause = steering the feature in its OWN activation units (controls the magnitude confound).
  * null = many random matched-magnitude directions through the identical pipeline (was: 3 -> a real pool).
  * robust z = (cause - median_null) / (1.4826 * MAD_null); bootstrap CI over probes.
  * SELF-TEST (synthetic anchors): a guaranteed driver (diff-of-means direction) must clear z_drv and a
    random null must not; else reporting is refused. Concept-general, no pre-verified features needed.
"""
from __future__ import annotations
import enum, numpy as np, torch
from dataclasses import dataclass
from .concepts import SENTIMENT, ReadoutSpec


class Verdict(enum.Enum):
    RULED_OUT = "ruled_out"          # causally indistinguishable from noise -> thermometer, exclude
    NOT_RULED_OUT = "not_ruled_out"  # significant causal effect -> driver-like (NOT a safety certificate)
    INDETERMINATE = "indeterminate"  # in between / too noisy / untestable


@dataclass(frozen=True)
class FeatureResult:
    feature: int
    read: float
    cause: float
    cause_ci: tuple
    z: float                 # robust SDs above this direction's random-noise floor (unit-free, cross-domain)
    verdict: Verdict
    # deliberately NO `safe_to_optimize` field: the tool issues no safety certificate.


def _robust_z(cause, null):
    med = float(np.median(null))
    mad = float(np.median(np.abs(null - med)))
    if mad < 1e-9:                         # degenerate null (no spread): can't scale -> fall back to sign, capped
        return 0.0 if abs(cause - med) < 1e-6 else float(np.sign(cause - med) * 50.0)
    return float(np.clip((cause - med) / (1.4826 * mad), -50.0, 50.0))   # cap: beyond ~10 SDs the value is moot


class FeatureScope:
    def __init__(self, layer: int = 12, concept: ReadoutSpec = SENTIMENT, device: str | None = None):
        from transformer_lens import HookedTransformer
        from sae_lens import SAE
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.layer = layer; self.concept = concept
        self.sae = SAE.from_pretrained("gemma-scope-2b-pt-res", f"layer_{layer}/width_16k/average_l0_82", device=self.device)
        self.hook = self.sae.cfg.metadata["hook_name"]
        self.model = HookedTransformer.from_pretrained("gemma-2-2b", device=self.device, dtype=torch.bfloat16)
        self.Wdec = self.sae.W_dec.detach().float()
        self.read = None; self._X = None; self._alive = None; self._diffmeans = None; self._base = None
        self.z_drv = 3.0; self.z_out = 1.0; self.self_test = None; self.results = None
        self._rng = np.random.RandomState(0)
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

    def fit(self, pos_texts, neg_texts):
        if not pos_texts or not neg_texts:
            raise ValueError("need both positive and negative examples")
        feats, resids = [], []
        for t in pos_texts + neg_texts:
            f, r = self._cache(t); feats.append(f); resids.append(r)
        self._X = np.stack(feats).astype(np.float64)
        R = torch.stack(resids); n_pos = len(pos_texts)
        self._diffmeans = (R[:n_pos].mean(0) - R[n_pos:].mean(0))
        y = np.array([1] * n_pos + [0] * len(neg_texts), dtype=np.float64)
        yc = y - y.mean(); zc = self._X - self._X.mean(0)
        self.read = np.nan_to_num((zc * yc[:, None]).mean(0) / (zc.std(0) * yc.std() + 1e-9))
        self._alive = np.where((self._X > 0).mean(0) > 0.02)[0]
        return self

    # ---------- CAUSE of an arbitrary residual direction (+ per-direction null) ----------
    def _readout(self, text, steer=None):
        hooks = [(self.hook, lambda r, hook: r + steer)] if steer is not None else []
        with torch.no_grad():
            lg = self.model.run_with_hooks(self.model.to_tokens(self._few + self.concept.template.format(text=text)),
                                           fwd_hooks=hooks)
        return (lg[0, -1, self._pos] - lg[0, -1, self._neg]).item()

    def _ensure_base(self):
        if self._base is None:                                  # readout with NO steering — same for every direction
            self._base = np.array([self._readout(p) for p in self.concept.probes])

    def _dir_shift(self, vec):
        s = vec.to(self.model.cfg.dtype)
        return np.array([self._readout(p, steer=s) for p in self.concept.probes]) - self._base

    def _cause_dir(self, unit, scale, sign=1.0, n_null=16, n_boot=400):
        self._ensure_base()
        per_probe = self._dir_shift(unit * scale) * sign
        cause = float(np.median(per_probe))
        boot = [np.median(per_probe[self._rng.randint(0, len(per_probe), len(per_probe))]) for _ in range(n_boot)]
        ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        null = np.array([float(np.median(self._dir_shift((torch.randn_like(unit) / torch.randn_like(unit).norm()) * scale)))
                         for _ in range(n_null)])              # random matched-magnitude directions = the noise floor
        return cause, ci, null

    def _cause_feature(self, feature, n_null=16):
        Wf = self.Wdec[feature]; unit = Wf / Wf.norm()
        col = self._X[:, feature]; active = col[col > 0]
        if active.size < 5:
            return None
        scale = float(np.percentile(active, 95)) * float(Wf.norm())
        cause, ci, null = self._cause_dir(unit, scale, float(np.sign(self.read[feature]) or 1.0), n_null)
        return cause, ci, _robust_z(cause, null)

    # ---------- CALIBRATE + SELF-TEST (synthetic anchors, z-scale) ----------
    def calibrate(self, anchor_norm=12.0, n_null=24):
        drv_unit = self._diffmeans / self._diffmeans.norm()
        cd, _, nd = self._cause_dir(drv_unit, anchor_norm, 1.0, n_null)
        zd = _robust_z(cd, nd)                                  # synthetic driver, in SDs above noise
        ru = torch.randn_like(self._diffmeans); ru = ru / ru.norm()
        cr, _, nr = self._cause_dir(ru, anchor_norm, 1.0, n_null)
        zr = _robust_z(cr, nr)                                  # random null, should be ~0
        self.self_test = {"driver_z": zd, "null_z": zr, "passed": (zd >= self.z_drv and zr < self.z_drv)}
        print(f"[calibrate:{self.concept.name}] synthetic-driver z={zd:.2f}  random-null z={zr:.2f}  "
              f"(driver gate z>={self.z_drv}, rule-out z<={self.z_out})")
        print(f"[self-test] {'PASSED' if self.self_test['passed'] else 'FAILED'} "
              f"(synthetic driver must clear the driver gate; random null must not)")
        if not self.self_test["passed"]:
            print("[self-test] FAILED: cause-machinery can't validate this concept (not a linear direction, or "
                  "weak readout/examples) — labels will be REFUSED.")
        return self

    # ---------- SCREEN + REPORT ----------
    def _verdict(self, cause, ci, z):
        if not np.isfinite(z):
            return Verdict.INDETERMINATE
        if z >= self.z_drv and ci[0] > 0:                      # significant, direction-specific causal effect
            return Verdict.NOT_RULED_OUT
        if z <= self.z_out:                                    # indistinguishable from this direction's noise
            return Verdict.RULED_OUT
        return Verdict.INDETERMINATE

    def screen(self, top_k=6, n_null=16):
        assert self.read is not None and self.self_test is not None, "call .fit() then .calibrate() first"
        top = self._alive[np.argsort(-np.abs(self.read[self._alive]))[:top_k]]
        rows = []
        for f in top:
            c = self._cause_feature(int(f), n_null)
            if c is None:
                continue
            cause, ci, z = c
            rows.append(FeatureResult(int(f), float(self.read[f]), cause, ci, z, self._verdict(cause, ci, z)))
        self.results = sorted(rows, key=lambda r: -r.z)
        return self.results

    def ruled_out_thermometers(self):
        assert self.results is not None
        return [r.feature for r in self.results if r.verdict is Verdict.RULED_OUT]

    def report(self):
        if not (self.self_test and self.self_test["passed"]):
            raise RuntimeError("self-test did not pass — refusing to report untrusted labels (run .calibrate())")
        assert self.results is not None, "call .screen() first"
        print(f"\nconcept: {self.concept.name}   (verdict on robust z = SDs above each direction's noise floor)")
        print(f"{'feature':>8}{'read':>8}{'cause':>8}{'  95% CI':>16}{'z':>7}   verdict")
        for r in self.results:
            print(f"{r.feature:>8}{r.read:>8.2f}{r.cause:>8.2f}  [{r.cause_ci[0]:.2f},{r.cause_ci[1]:.2f}]"
                  f"{r.z:>7.1f}   {r.verdict.value}")
        print(f"\nRULED OUT (do NOT use as rewards): {self.ruled_out_thermometers()}")
        print("Honest scope: ONE-SIDED NEGATIVE screen on a unit-free statistic (robust z vs a per-direction "
              "random-noise floor), so the rule transfers across concepts. 'not_ruled_out' is NOT a safety "
              "certificate — driver-ness is necessary, not sufficient. z gates (driver/rule-out) are defaults "
              "pending leave-one-concept-out validation across more concepts.")
