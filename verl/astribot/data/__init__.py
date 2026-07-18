from .embodiment import *

__all__ = ["compose_t_layout_rgb", "preprocess_t_layout", "FeatureNormalizer", "RunningStats"]


def __getattr__(name):
    if name in {"compose_t_layout_rgb", "preprocess_t_layout"}:
        from . import image_composition
        return getattr(image_composition, name)
    if name in {"FeatureNormalizer", "RunningStats"}:
        from . import normalization
        return getattr(normalization, name)
    raise AttributeError(name)
