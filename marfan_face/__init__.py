"""Marfan facial screening — shareable encoder + linear screener.

The decoder (latent -> face) is intentionally NOT part of this package.
"""
from .encoder import SpiralEncoder
from .inference import MarfanScreener

__all__ = ["SpiralEncoder", "MarfanScreener"]
