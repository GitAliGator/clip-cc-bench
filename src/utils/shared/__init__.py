"""
Shared utilities for the isolated encoder system.
"""

from .base_types import (
    EmbeddingResult,
    EvaluationResult,
    SimilarityScore,
    EncoderEvaluationResult,
    ModelConfig,
    EncoderPaths
)

from .result_manager import SharedResultManager
from .config_loader import IsolatedEncoderConfigLoader

__all__ = [
    'EmbeddingResult',
    'EvaluationResult',
    'SimilarityScore',
    'EncoderEvaluationResult',
    'ModelConfig',
    'EncoderPaths',
    'SharedResultManager',
    'IsolatedEncoderConfigLoader'
]