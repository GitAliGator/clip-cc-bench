"""
Shared Base Types for Isolated Encoder System

Common data structures and types used across all encoder modules.
"""

import torch
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class SimilarityScore:
    """Container for similarity computation results."""
    cosine_similarity: float
    normalized_cosine: float
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
class EncoderEvaluationResult:
    """Result container for encoder evaluation of a single video."""
    video_id: str
    model_name: str
    ground_truth_text: str
    prediction_text: str
    encoder_similarities: Dict[str, SimilarityScore]
    timestamp: str
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ModelConfig:
    """Configuration for an individual encoder model."""
    name: str
    path: str
    type: str
    batch_size: int
    max_length: int
    device_map: str
    trust_remote_code: bool
    additional_params: Optional[Dict[str, Any]] = None


class EncoderPaths:
    """Standardized path management for encoder modules."""

    def __init__(self, base_dir: Path, encoder_name: str):
        self.base_dir = Path(base_dir)
        self.encoder_name = encoder_name

        # Module-specific directories
        self.config_dir = self.base_dir / "config" / encoder_name
        self.scripts_dir = self.base_dir / "scripts" / encoder_name
        self.utils_dir = self.base_dir / "utils" / encoder_name
        self.venv_dir = self.scripts_dir / "venv"

        # Shared directories
        self.shared_utils_dir = self.base_dir / "utils" / "shared"
        self.results_base_dir = self.base_dir / "results"

        # Result directories (shared across all encoders)
        self.individual_csv_dir = self.results_base_dir / "encoders" / "individual_results" / "csv"
        self.individual_json_dir = self.results_base_dir / "encoders" / "individual_results" / "json"
        self.aggregated_results_dir = self.results_base_dir / "encoders" / "aggregated_results"
        self.logs_dir = self.results_base_dir / "encoders" / "logs"
        self.cache_dir = self.results_base_dir / "encoders" / "cache"

        # Data directories
        self.data_dir = self.base_dir / "data"
        self.ground_truth_file = self.data_dir / "ground_truth" / "clip_cc_dataset.json"
        self.predictions_dir = self.data_dir / "models"

    def ensure_directories(self):
        """Create all necessary directories."""
        for dir_path in [
            self.config_dir, self.scripts_dir, self.utils_dir,
            self.individual_csv_dir, self.individual_json_dir,
            self.aggregated_results_dir, self.logs_dir, self.cache_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_config_file(self) -> Path:
        """Get the encoder-specific config file path."""
        return self.config_dir / "encoders_config.yaml"

    def get_requirements_file(self) -> Path:
        """Get the encoder-specific requirements file path."""
        return self.config_dir / "requirements.txt"

    def get_run_script(self) -> Path:
        """Get the encoder-specific run script path."""
        return self.scripts_dir / f"run_{self.encoder_name}_evaluation.py"