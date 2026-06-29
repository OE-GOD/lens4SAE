"""FeatureScope — screen SAE features as DRIVERS vs THERMOMETERS for a concept.

    from featurescope import FeatureScope, Verdict
    from featurescope.concepts import SENTIMENT, FORMALITY, ReadoutSpec
"""
from .core import FeatureScope, Verdict, FeatureResult
from .concepts import ReadoutSpec, SENTIMENT, FORMALITY, REGISTRY

__all__ = ["FeatureScope", "Verdict", "FeatureResult", "ReadoutSpec", "SENTIMENT", "FORMALITY", "REGISTRY"]
__version__ = "0.11.0"
