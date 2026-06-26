"""FeatureScope core — driver/thermometer screening for SAE features.

A DRIVER is a feature the model *computes with*: steering it changes behaviour (read = write).
A THERMOMETER is decodable/correlated but causally inert: steering does nothing (read != write).
Only DRIVERS are safe to use as an RL reward / monitor; THERMOMETERS get gamed (Goodhart).

The honest, defensible use is a ONE-SIDED NEGATIVE SCREEN: cheaply *rule out* thermometers.
FeatureScope never certifies a feature "safe to optimize" — driver-ness is NECESSARY, not sufficient
(a verified driver can still decouple under hard optimization). See README "Scope & limits".

Design choices that make it correct (not just runnable):
  * cause is measured by steering each feature in its OWN activation units (a_hi * W_dec, not a shared
    norm) -> controls for the magnitude confound ('driver' must mean causal, not high-activation).
  * each feature's effect is compared to a RANDOM-direction control band at matched magnitude -> the
    effect must be SPECIFIC to the feature's direction, not generic perturbation.
  * bootstrap CIs over evaluation prompts -> uncertainty is reported, not hidden.
  * a SELF-TEST gates reporting: unless a known driver out-causes a known thermometer, labels are
    refused. The tool validates itself against ground truth before it will speak.
"""
from __future__ import annotations
import enum, numpy as np, torch
from dataclasses import dataclass

# verified ground truth (sentiment, Gemma-2-2b, Gemma Scope layer 12), via steering + attribution
GROUND_TRUTH = {"driver": 8000, "thermometer": 10511}
# neutral probes: steering needs headroom to move sentiment either way
PROBES = [
    "The movie was decent.", "An average film overall.", "It was okay, I guess.",
    "A film I watched last night.", "The product arrived on time.", "Here is my honest review.",
    "The book was a normal length.", "I went to see it yesterday.", "The service was as expected.",
    "This is a review of the item.", "The show ran for two hours.", "I finished it on the weekend.",
]


class Verdict(enum.Enum):
    RULED_OUT = "ruled_out"          # confidently a thermometer -> do NOT use as a reward
    NOT_RULED_OUT = "not_ruled_out"  # behaves like a driver -> cannot rule out (NOT a safety certificate)
    INDETERMINATE = "indeterminate"  # untestable / too noisy to decide


@dataclass(frozen=True)
class FeatureResult:
    feature: int
    read: float
    cause: float            # signed readout shift in the feature's own units (relative ranking quantity)
    cause_ci: tuple         # bootstrap 95% CI over probes
    control_hi: float       # 95th pct of |random-direction| effect at matched magnitude (the null band)
    verdict: Verdict
    # there is deliberately NO `safe_to_optimize` field: the tool issues no safety certificate.


class FeatureScope:
    def __init__(self, layer: int = 12, device: str | None = None):
        from transformer_lens import HookedTransformer
        from sae_lens import SAE
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.layer = layer
        self.sae = SAE.from_pretrained("gemma-scope-2b-pt-res", f"layer_{layer}/width_16k/average_l0_82", device=self.device)
        self.hook = self.sae.cfg.metadata["hook_name"]
        self.model = HookedTransformer.from_pretrained("gemma-2-2b", device=self.device, dtype=torch.bfloat16)
        self.Wdec = self.sae.W_dec.detach().float()
        self.read = None; self._X = None; self._alive = None
        self.threshold = None; self.self_test = None; self.results = None
        self._rng = np.random.RandomState(0)
        self._few = ("Review: A wonderful, heartwarming film.\nSentiment: positive\n"
                     "Review: A boring, pointless waste of time.\nSentiment: negative\n")
        self._pos = self.model.to_tokens(" positive", prepend_bos=False)[0, 0]
        self._neg = self.model.to_tokens(" negative", prepend_bos=False)[0, 0]

    # ---------- READ ----------
    def _acts(self, text, pool="max"):
        with torch.no_grad():
            _, c = self.model.run_with_cache(self.model.to_tokens(text[:300]), names_filter=[self.hook])
            f = self.sae.encode(c[self.hook][0]).float()
        return (f.max(0).values if pool == "max" else f.mean(0)).cpu().numpy()

    def fit(self, pos_texts, neg_texts, pool="max"):
        if not pos_texts or not neg_texts:
            raise ValueError("need both positive and negative examples")
        self._X = np.stack([self._acts(t, pool) for t in pos_texts + neg_texts]).astype(np.float64)
        y = np.array([1] * len(pos_texts) + [0] * len(neg_texts), dtype=np.float64)
        yc = y - y.mean(); zc = self._X - self._X.mean(0)
        self.read = np.nan_to_num((zc * yc[:, None]).mean(0) / (zc.std(0) * yc.std() + 1e-9))
        self._alive = np.where((self._X > 0).mean(0) > 0.02)[0]
        return self

    # ---------- CAUSE (own-units steering + random-direction control + bootstrap CI) ----------
    def _readout(self, text, steer=None):
        hooks = [(self.hook, lambda r, hook: r + steer)] if steer is not None else []
        with torch.no_grad():
            lg = self.model.run_with_hooks(self.model.to_tokens(self._few + f"Review: {text}\nSentiment:"), fwd_hooks=hooks)
        return (lg[0, -1, self._pos] - lg[0, -1, self._neg]).item()

    def _own_unit_vec(self, feature):
        """Steer the feature at a strong-but-natural firing level (95th pct of its active activations)."""
        col = self._X[:, feature]; active = col[col > 0]
        if active.size < 5:                                 # too sparse on this corpus to set a scale
            return None
        a_hi = float(np.percentile(active, 95))
        return a_hi * self.Wdec[feature]                    # own units; W_dec NOT renormalized

    def _shifts(self, vec, mult=2.0):
        """Per-probe signed readout shift when steering +mult*vec."""
        s = (mult * vec).to(self.model.cfg.dtype)
        base = np.array([self._readout(p) for p in PROBES])
        steer = np.array([self._readout(p, steer=s) for p in PROBES])
        return steer - base

    def _cause(self, feature, mult=2.0, n_controls=3, n_boot=500, fallback_norm=None):
        vec = self._own_unit_vec(feature)
        if vec is None:
            if fallback_norm is None:
                return None                                  # genuinely untestable on this corpus
            col = self.Wdec[feature]; vec = col / col.norm() * fallback_norm  # robust scale for anchors
        sign = float(np.sign(self.read[feature]) or 1.0)
        per_probe = self._shifts(vec, mult) * sign
        cause = float(np.median(per_probe))
        boot = [np.median(per_probe[self._rng.randint(0, len(per_probe), len(per_probe))]) for _ in range(n_boot)]
        ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        # control band: random directions at the SAME norm as vec (magnitude-matched specificity test)
        norm = float(vec.norm()); ctrl = []
        for _ in range(n_controls):
            r = torch.randn_like(vec); r = r / r.norm() * norm
            ctrl.append(abs(float(np.median(self._shifts(r, mult)))))
        control_hi = float(np.percentile(ctrl, 95)) if ctrl else 0.0
        return cause, ci, control_hi

    # ---------- CALIBRATE + SELF-TEST ----------
    def calibrate(self, mult=2.0, margin=0.2, fallback_norm=12.0):
        # anchors get a robust fallback scale so calibration never depends on the user's corpus size
        cd = self._cause(GROUND_TRUTH["driver"], mult, fallback_norm=fallback_norm)
        ct = self._cause(GROUND_TRUTH["thermometer"], mult, fallback_norm=fallback_norm)
        if cd is None or ct is None:                         # fail gracefully, no traceback
            self.self_test = {"passed": False}; self.threshold = 0.0
            print("[self-test] FAILED: anchor features not testable on this corpus — provide more/varied "
                  "examples. Labels will be REFUSED.")
            return self
        d, t = cd[0], ct[0]
        self.threshold = t + 0.5 * (d - t)
        self.self_test = {"driver_cause": d, "thermo_cause": t, "margin": d - t,
                          "passed": (d > t + margin), "mult": mult}
        print(f"[calibrate] driver(8000)={d:.2f}  thermometer(10511)={t:.2f}  threshold={self.threshold:.2f}")
        print(f"[self-test] {'PASSED' if self.self_test['passed'] else 'FAILED'} (need driver > thermometer + {margin})")
        if not self.self_test["passed"]:
            print("[self-test] WARNING: cannot separate known driver from thermometer — labels will be REFUSED.")
        return self

    # ---------- SCREEN + REPORT ----------
    def _verdict(self, cause, ci, control_hi):
        if ci[1] - ci[0] > 1.5:                       # too noisy to decide
            return Verdict.INDETERMINATE
        if cause <= control_hi and cause < self.threshold:   # indistinguishable from random, below bar -> thermometer
            return Verdict.RULED_OUT
        if cause > self.threshold and cause > control_hi:    # specific causal effect above bar
            return Verdict.NOT_RULED_OUT
        return Verdict.INDETERMINATE

    def screen(self, top_k=10, mult=2.0):
        assert self.read is not None and self.threshold is not None, "call .fit() then .calibrate() first"
        top = self._alive[np.argsort(-np.abs(self.read[self._alive]))[:top_k]]
        rows = []
        for f in top:
            c = self._cause(int(f), mult)
            if c is None:
                continue
            cause, ci, control_hi = c
            rows.append(FeatureResult(int(f), float(self.read[f]), cause, ci, control_hi,
                                      self._verdict(cause, ci, control_hi)))
        self.results = sorted(rows, key=lambda r: -r.cause)
        return self.results

    def ruled_out_thermometers(self):
        """The honest one-sided output: features confidently safe to EXCLUDE as rewards."""
        assert self.results is not None
        return [r.feature for r in self.results if r.verdict is Verdict.RULED_OUT]

    def report(self):
        if not (self.self_test and self.self_test["passed"]):
            raise RuntimeError("self-test did not pass — refusing to report untrusted labels (run .calibrate())")
        assert self.results is not None, "call .screen() first"
        print(f"\n{'feature':>8}{'read':>8}{'cause':>8}{'  95% CI':>16}{'  ctrl':>8}   verdict")
        for r in self.results:
            print(f"{r.feature:>8}{r.read:>8.2f}{r.cause:>8.2f}  [{r.cause_ci[0]:.2f},{r.cause_ci[1]:.2f}]"
                  f"{r.control_hi:>8.2f}   {r.verdict.value}")
        ro = self.ruled_out_thermometers()
        print(f"\nRULED OUT (do NOT use as rewards): {ro}")
        print("Honest scope: this is a ONE-SIDED NEGATIVE screen. 'not_ruled_out' is NOT a safety "
              "certificate — driver-ness is necessary, not sufficient (drivers can decouple under hard "
              "optimization). Cause is a relative ranking, magnitude-controlled, with bootstrap CIs.")
