"""Fast tests (no model load). Verdict logic, honesty invariants, and the self-test gate."""
import pytest
from featurescope import Verdict, FeatureResult
from featurescope.core import FeatureScope


def test_verdict_members():
    assert {v.value for v in Verdict} == {"ruled_out", "not_ruled_out", "indeterminate"}


def test_no_safety_certificate():
    # the tool must never expose a "safe_to_optimize" field — it issues no safety certificate
    r = FeatureResult(feature=1, read=0.4, cause=0.5, cause_ci=(0.3, 0.7), control_hi=0.1, verdict=Verdict.NOT_RULED_OUT)
    assert not hasattr(r, "safe_to_optimize")
    with pytest.raises(AttributeError):
        r.safe_to_optimize  # noqa


def _stub(threshold=0.3):
    fs = FeatureScope.__new__(FeatureScope)   # skip heavy __init__
    fs.threshold = threshold
    return fs


def test_verdict_rules_out_thermometer():
    fs = _stub()
    # low cause, within control band, below threshold -> ruled out (thermometer)
    assert fs._verdict(cause=0.05, ci=(0.0, 0.1), control_hi=0.12) is Verdict.RULED_OUT


def test_verdict_keeps_driver():
    fs = _stub()
    # cause clearly above threshold and above control band -> not ruled out (driver-like)
    assert fs._verdict(cause=0.5, ci=(0.4, 0.6), control_hi=0.1) is Verdict.NOT_RULED_OUT


def test_verdict_indeterminate_when_noisy():
    fs = _stub()
    assert fs._verdict(cause=0.5, ci=(-0.5, 1.5), control_hi=0.1) is Verdict.INDETERMINATE


def test_report_refuses_without_passing_self_test():
    fs = _stub()
    fs.self_test = {"passed": False}
    fs.results = []
    with pytest.raises(RuntimeError):
        fs.report()
