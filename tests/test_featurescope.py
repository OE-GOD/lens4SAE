"""Fast tests (no model load). Verdict logic, honesty invariants, and the self-test gate."""
import pytest
import numpy as np
from featurescope import Verdict, FeatureResult
from featurescope.core import FeatureScope, _robust_z


def test_verdict_members():
    assert {v.value for v in Verdict} == {"ruled_out", "not_ruled_out", "indeterminate"}


def test_no_safety_certificate():
    # the tool must never expose a "safe_to_optimize" field — it issues no safety certificate
    r = FeatureResult(feature=1, read=0.4, cause=0.5, cause_ci=(0.3, 0.7), z=5.0, verdict=Verdict.NOT_RULED_OUT)
    assert not hasattr(r, "safe_to_optimize")
    with pytest.raises(AttributeError):
        r.safe_to_optimize  # noqa


def test_robust_z_is_unit_free():
    # multiplying the whole scale by a constant must leave z unchanged (cross-domain invariance)
    null = np.array([-0.1, 0.0, 0.1, -0.05, 0.05])
    assert abs(_robust_z(0.5, null) - _robust_z(5.0, null * 10)) < 1e-6


def test_robust_z_degenerate_null_does_not_explode():
    # MAD==0 (zero-spread null) must NOT produce a 6e8 z; capped, sign-correct, and 0 for no effect
    null = np.zeros(16)
    assert 0 < _robust_z(0.62, null) <= 50.0
    assert _robust_z(0.0, null) == 0.0


def _stub():
    fs = FeatureScope.__new__(FeatureScope)   # skip heavy __init__
    fs.z_drv, fs.z_out = 3.0, 1.0
    return fs


def test_verdict_rules_out_thermometer():
    fs = _stub()
    # z at/below the rule-out gate -> thermometer
    assert fs._verdict(cause=0.05, ci=(-0.02, 0.1), z=0.5) is Verdict.RULED_OUT


def test_verdict_keeps_driver():
    fs = _stub()
    # z clears the driver gate and CI lower bound is positive -> driver-like
    assert fs._verdict(cause=0.5, ci=(0.3, 0.7), z=5.0) is Verdict.NOT_RULED_OUT


def test_verdict_indeterminate_between_gates():
    fs = _stub()
    assert fs._verdict(cause=0.3, ci=(0.1, 0.5), z=2.0) is Verdict.INDETERMINATE


def test_report_refuses_without_passing_self_test():
    fs = _stub()
    fs.self_test = {"passed": False}
    fs.results = []
    with pytest.raises(RuntimeError):
        fs.report()
