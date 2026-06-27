"""FeatureScope core — driver/thermometer screening for SAE features (concept-general, v0.3).

A DRIVER is a feature the model *computes with*: steering it changes behaviour (read = write).
A THERMOMETER is decodable/correlated but causally inert: steering does nothing (read != write).
Only DRIVERS are safe to use as an RL reward / monitor; THERMOMETERS get gamed (Goodhart).

The honest, defensible use is a ONE-SIDED NEGATIVE SCREEN: cheaply *rule out* thermometers.
FeatureScope never certifies a feature "safe to optimize" — driver-ness is NECESSARY, not sufficient.

Design choices that make it correct (not just runnable):
  * cause is measured by steering each feature in its OWN activation units (a_hi * W_dec) -> controls
    for the magnitude confound ('driver' must mean causal, not high-activation).
  * each effect is compared to a RANDOM-direction control band at matched magnitude -> the effect must
    be SPECIFIC to the feature's direction, not generic perturbation. Bootstrap CIs over probes.
  * SELF-TEST with SYNTHETIC ANCHORS (concept-general): a guaranteed driver = the concept's
    difference-of-means residual direction; a guaranteed null = a random direction. Reporting is
    refused unless the synthetic driver out-causes the null. If the concept is not a linear direction
    (e.g. relational/comparison), this self-test FAILS honestly instead of emitting garbage labels.
"""
from __future__ import annotations
import enum, numpy as np, torch
from dataclasses import dataclass
from .concepts import SENTIMENT, ReadoutSpec


class Verdict(enum.Enum):
    RULED_OUT = "ruled_out"          # confidently a thermometer -> do NOT use as a reward
    NOT_RULED_OUT = "not_ruled_out"  # behaves like a driver -> cannot rule out (NOT a safety certificate)
    INDETERMINATE = "indeterminate"  # untestable / too noisy to decide


@dataclass(frozen=True)
class FeatureResult:
    feature: int
    read: float
    cause: float
    cause_ci: tuple
    control_hi: float
    verdict: Verdict
    # deliberately NO `safe_to_optimize` field: the tool issues no safety certificate.


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
        self.read = None; self._X = None; self._alive = None; self._diffmeans = None
        self.threshold = None; self.self_test = None; self.results = None
        self._rng = np.random.RandomState(0)
        self._few = concept.few_shot
        self._pos = self.model.to_tokens(concept.pos_word, prepend_bos=False)[0, 0]
        self._neg = self.model.to_tokens(concept.neg_word, prepend_bos=False)[0, 0]

    # ---------- READ + difference-of-means direction (the synthetic-driver anchor) ----------
    def _cache(self, text):
        with torch.no_grad():
            _, c = self.model.run_with_cache(self.model.to_tokens(text[:300]), names_filter=[self.hook])
            resid = c[self.hook][0]                              # [seq, d_model]
            feats = self.sae.encode(resid).float()
            return feats.max(0).values.cpu().numpy(), resid.float().mean(0)   # (pooled SAE feats, mean resid)

    def fit(self, pos_texts, neg_texts):
        if not pos_texts or not neg_texts:
            raise ValueError("need both positive and negative examples")
        feats, resids = [], []
        for t in pos_texts + neg_texts:
            f, r = self._cache(t); feats.append(f); resids.append(r)
        self._X = np.stack(feats).astype(np.float64)
        R = torch.stack(resids); n_pos = len(pos_texts)
        self._diffmeans = (R[:n_pos].mean(0) - R[n_pos:].mean(0))   # concept direction in residual space
        y = np.array([1] * n_pos + [0] * len(neg_texts), dtype=np.float64)
        yc = y - y.mean(); zc = self._X - self._X.mean(0)
        self.read = np.nan_to_num((zc * yc[:, None]).mean(0) / (zc.std(0) * yc.std() + 1e-9))
        self._alive = np.where((self._X > 0).mean(0) > 0.02)[0]
        return self

    # ---------- CAUSE of an arbitrary residual direction ----------
    def _readout(self, text, steer=None):
        hooks = [(self.hook, lambda r, hook: r + steer)] if steer is not None else []
        with torch.no_grad():
            lg = self.model.run_with_hooks(self.model.to_tokens(self._few + self.concept.template.format(text=text)),
                                           fwd_hooks=hooks)
        return (lg[0, -1, self._pos] - lg[0, -1, self._neg]).item()

    def _shifts(self, vec):
        s = vec.to(self.model.cfg.dtype)
        base = np.array([self._readout(p) for p in self.concept.probes])
        steer = np.array([self._readout(p, steer=s) for p in self.concept.probes])
        return steer - base

    def _cause_dir(self, unit, scale, sign=1.0, n_controls=3, n_boot=400):
        per_probe = self._shifts(unit * scale) * sign
        cause = float(np.median(per_probe))
        boot = [np.median(per_probe[self._rng.randint(0, len(per_probe), len(per_probe))]) for _ in range(n_boot)]
        ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        ctrl = []
        for _ in range(n_controls):
            r = torch.randn_like(unit); r = r / r.norm()
            ctrl.append(abs(float(np.median(self._shifts(r * scale)))))
        return cause, ci, float(np.percentile(ctrl, 95)) if ctrl else 0.0

    def _cause_feature(self, feature, fallback_norm=None):
        Wf = self.Wdec[feature]; unit = Wf / Wf.norm()
        col = self._X[:, feature]; active = col[col > 0]
        if active.size >= 5:
            scale = float(np.percentile(active, 95)) * float(Wf.norm())   # own units
        elif fallback_norm is not None:
            scale = fallback_norm
        else:
            return None
        return self._cause_dir(unit, scale, float(np.sign(self.read[feature]) or 1.0))

    # ---------- CALIBRATE + SELF-TEST (synthetic anchors) ----------
    def calibrate(self, margin=0.2, anchor_norm=12.0):
        drv_unit = self._diffmeans / self._diffmeans.norm()
        d = self._cause_dir(drv_unit, anchor_norm, 1.0)[0]               # synthetic driver
        null_unit = torch.randn_like(self._diffmeans); null_unit = null_unit / null_unit.norm()
        t = self._cause_dir(null_unit, anchor_norm, 1.0)[0]              # synthetic null
        self.threshold = t + 0.5 * (d - t)
        self.self_test = {"driver_cause": d, "null_cause": t, "margin": d - t, "passed": (d > t + margin)}
        print(f"[calibrate:{self.concept.name}] synthetic-driver={d:.2f}  random-null={t:.2f}  threshold={self.threshold:.2f}")
        print(f"[self-test] {'PASSED' if self.self_test['passed'] else 'FAILED'} (driver must out-cause null + {margin})")
        if not self.self_test["passed"]:
            print("[self-test] FAILED: cause-machinery can't validate this concept (not a linear direction, "
                  "or weak readout/examples) — labels will be REFUSED.")
        return self

    # ---------- SCREEN + REPORT ----------
    def _verdict(self, cause, ci, control_hi):
        if ci[1] - ci[0] > 1.5:
            return Verdict.INDETERMINATE
        if cause <= control_hi and cause < self.threshold:
            return Verdict.RULED_OUT
        if cause > self.threshold and cause > control_hi:
            return Verdict.NOT_RULED_OUT
        return Verdict.INDETERMINATE

    def screen(self, top_k=10):
        assert self.read is not None and self.threshold is not None, "call .fit() then .calibrate() first"
        top = self._alive[np.argsort(-np.abs(self.read[self._alive]))[:top_k]]
        rows = []
        for f in top:
            c = self._cause_feature(int(f))
            if c is None:
                continue
            rows.append(FeatureResult(int(f), float(self.read[f]), c[0], c[1], c[2], self._verdict(*c)))
        self.results = sorted(rows, key=lambda r: -r.cause)
        return self.results

    def ruled_out_thermometers(self):
        assert self.results is not None
        return [r.feature for r in self.results if r.verdict is Verdict.RULED_OUT]

    def report(self):
        if not (self.self_test and self.self_test["passed"]):
            raise RuntimeError("self-test did not pass — refusing to report untrusted labels (run .calibrate())")
        assert self.results is not None, "call .screen() first"
        print(f"\nconcept: {self.concept.name}")
        print(f"{'feature':>8}{'read':>8}{'cause':>8}{'  95% CI':>16}{'  ctrl':>8}   verdict")
        for r in self.results:
            print(f"{r.feature:>8}{r.read:>8.2f}{r.cause:>8.2f}  [{r.cause_ci[0]:.2f},{r.cause_ci[1]:.2f}]"
                  f"{r.control_hi:>8.2f}   {r.verdict.value}")
        print(f"\nRULED OUT (do NOT use as rewards): {self.ruled_out_thermometers()}")
        print("Honest scope: ONE-SIDED NEGATIVE screen. 'not_ruled_out' is NOT a safety certificate — "
              "driver-ness is necessary, not sufficient. Cause is a relative ranking, magnitude-controlled, "
              "with bootstrap CIs; self-test uses synthetic anchors for this concept.")
