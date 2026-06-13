"""Model subpackage: interchangeable EEG backbones + classifier/projection heads.

Importing this module registers all backbones in ``MODELS``.
"""

from ssleeg.models.base import EEGBackbone, EEGClassifier, ProjectionHead, build_model
from ssleeg.models import eegnet, convnets, transformer  # noqa: F401 (registers backbones)

__all__ = ["EEGBackbone", "EEGClassifier", "ProjectionHead", "build_model"]
