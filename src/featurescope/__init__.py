"""FeatureScope — screen SAE features as DRIVERS vs THERMOMETERS for a concept.

Public API:
    from featurescope import FeatureScope, Verdict
"""
from .core import FeatureScope, Verdict, FeatureResult

__all__ = ["FeatureScope", "Verdict", "FeatureResult"]
__version__ = "0.2.0"
