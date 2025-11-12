"""
G-VEval LLaMA Implementation

Standalone implementation of G-VEval using LLaMA inference.
Replicates the methodology from "G-VEval: A Versatile Metric for Evaluating 
Image and Video Captions Using GPT-4o" but using LLaMA models.
"""

from .gveval_llama_scorer import GVEvalLLaMAScorer, GVEvalResult
from .gveval_llama_core import GVEvalLLaMAEvaluator
from .gveval_config_loader import GVEvalConfigLoader

__version__ = "1.0.0"
__all__ = [
    "GVEvalLLaMAScorer",
    "GVEvalResult", 
    "GVEvalLLaMAEvaluator",
    "GVEvalConfigLoader"
]