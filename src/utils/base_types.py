"""
Shared Base Types for Isolated Decoder System

Common data structures and types used across all decoder modules.
"""

import torch
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class SimilarityScore:
    """Container for similarity computation results."""
    # Coarse-grained metrics
    cosine_similarity: float
    normalized_cosine: float

    # Fine-grained metrics
    fine_grained_precision: Optional[float] = None
    fine_grained_recall: Optional[float] = None
    fine_grained_f1: Optional[float] = None

    # Hybrid metric: harmonic mean of coarse and fine F1
    hm_cf: Optional[float] = None

    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingResult:
    """Result container for embedding operations."""
    embeddings: torch.Tensor
    input_texts: List[str]
    model_name: str
    device: str
    success: bool = True
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationResult:
    """Result container for single evaluation operations."""
    video_id: str
    ground_truth_text: str
    prediction_text: str
    similarity_score: SimilarityScore
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class DecoderEvaluationResult:
    """Result container for decoder evaluation of a single video."""
    video_id: str
    model_name: str
    ground_truth_text: str
    prediction_text: str
    decoder_similarities: Dict[str, SimilarityScore]
    timestamp: str
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ModelConfig:
    """Configuration for an individual decoder model."""
    name: str
    path: str
    type: str
    batch_size: int
    max_length: int
    device_map: str
    trust_remote_code: bool
    additional_params: Optional[Dict[str, Any]] = None


class DecoderPaths:
    """Standardized path management for decoder modules."""

    def __init__(self, base_dir: Path, decoder_name: str):
        self.base_dir = Path(base_dir)
        self.decoder_name = decoder_name

        # Module-specific directories (flattened structure)
        self.configs_dir = self.base_dir / "configs"
        self.scripts_dir = self.base_dir / "scripts"
        self.utils_dir = self.base_dir / "utils"

        self.results_base_dir = self.base_dir / "results"

        # Result directories (shared across all decoders)
        self.individual_csv_dir = self.results_base_dir / "decoders" / "individual_results" / "csv"
        self.individual_json_dir = self.results_base_dir / "decoders" / "individual_results" / "json"
        self.aggregated_results_dir = self.results_base_dir / "decoders" / "aggregated_results"
        self.logs_dir = self.results_base_dir / "decoders" / "logs"
        self.cache_dir = self.results_base_dir / "decoders" / "cache"

        # Data directories
        self.data_dir = self.base_dir / "data"
        self.ground_truth_file = self.data_dir / "ground_truth" / "clip_cc_dataset.json"
        self.predictions_dir = self.data_dir / "models"

    def ensure_directories(self):
        """Create all necessary directories."""
        for dir_path in [
            self.configs_dir, self.scripts_dir, self.utils_dir,
            self.individual_csv_dir, self.individual_json_dir,
            self.aggregated_results_dir, self.logs_dir, self.cache_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_config_file(self) -> Path:
        """Get the decoder-specific config file path."""
        return self.configs_dir / f"{self.decoder_name}.yaml"

    def get_requirements_file(self) -> Path:
        """Get the decoder-specific requirements file path."""
        return self.configs_dir / "requirements" / f"{self.decoder_name}.txt"

    def get_run_script(self) -> Path:
        """Get the decoder-specific run script path."""
        return self.scripts_dir / f"run_{self.decoder_name}_evaluation.py"
