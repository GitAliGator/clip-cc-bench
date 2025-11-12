"""
Nomic-Embed-text-v1.5 Specific Embedding Model Implementation

Isolated implementation for Nomic-Embed-text-v1.5 embeddings with symmetric task configuration.
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import gc
import time

# Import shared types
import sys
sys.path.append(str(Path(__file__).parent.parent / "shared"))
from base_types import EmbeddingResult, SimilarityScore, EvaluationResult

# Import path management
from paths import get_project_paths


class NomicEmbedModel:
    """Nomic-Embed-text-v1.5 model implementation with 768D embeddings."""

    def __init__(self, model_path: str, device: str = "cuda:0", **kwargs):
        self.model_path = model_path
        self.device = device
        self.batch_size = kwargs.get('batch_size', 32)
        self.max_length = kwargs.get('max_length', 8192)
        self.embedding_dimension = kwargs.get('embedding_dimension', 768)
        self.task_type = kwargs.get('task_type', 'classification')
        self.torch_dtype = kwargs.get('torch_dtype', 'bfloat16')
        self.normalize_embeddings = kwargs.get('normalize_embeddings', True)
        self.pooling_strategy = kwargs.get('pooling_strategy', 'mean')

        self.logger = logging.getLogger('nomic_embed_model')
        self.model = None

        # Performance tracking
        self.embed_times = []
        self.memory_usage = []

    def _get_model_path(self):
        """Determine the best path for loading the model (local vs remote)."""
        try:
            # Get project paths
            project_paths = get_project_paths()

            # Check if we have a local model path
            local_model_path = project_paths.get_encoder_models_dir() / "nomic-embed-text-v1.5"

            # Check if local model exists and has model files
            if local_model_path.exists():
                # Check for key model files
                config_file = local_model_path / "config.json"
                modules_file = local_model_path / "modules.json"

                # Check for various model file patterns Nomic-Embed might use
                safetensors_files = list(local_model_path.glob("*.safetensors"))
                model_files = list(local_model_path.glob("*.bin"))

                if config_file.exists() and modules_file.exists() and (safetensors_files or model_files):
                    self.logger.info(f"Found local Nomic-Embed model at: {local_model_path}")
                    self.logger.info(f"  config.json: ✓")
                    self.logger.info(f"  modules.json: ✓")
                    self.logger.info(f"  Model files: {len(safetensors_files + model_files)} files")
                    return str(local_model_path)
                else:
                    self.logger.info("Local model directory exists but incomplete, falling back to remote")
                    self.logger.info(f"  config.json: {'✓' if config_file.exists() else '✗'}")
                    self.logger.info(f"  modules.json: {'✓' if modules_file.exists() else '✗'}")
                    self.logger.info(f"  Model files: {len(safetensors_files + model_files)} files")
            else:
                self.logger.info("No local model found, will use remote HuggingFace model")

            # Fall back to the original HuggingFace path
            return self.model_path

        except Exception as e:
            self.logger.warning(f"Error checking for local model: {e}, using remote path")
            return self.model_path

    def load_model(self):
        """Load Nomic-Embed-text-v1.5 model using SentenceTransformer with local fallback."""
        try:
            # Try to determine the best model path
            model_path_to_use = self._get_model_path()
            self.logger.info(f"Loading Nomic-Embed-text-v1.5 model from {model_path_to_use}")

            # Initialize model with trust_remote_code for nomic models
            self.model = SentenceTransformer(
                model_path_to_use,
                device=self.device,
                trust_remote_code=True
            )

            # Set model to evaluation mode
            self.model.eval()

            # Configure model settings
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = self.max_length

            # Log model info
            model_info = self._get_model_info()
            self.logger.info(f"Model loaded successfully:")
            self.logger.info(f"  Device: {self.device}")
            self.logger.info(f"  Max sequence length: {self.max_length}")
            self.logger.info(f"  Embedding dimension: {self.embedding_dimension}")
            self.logger.info(f"  Task type: {self.task_type}")
            self.logger.info(f"  Model info: {model_info}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load Nomic-Embed model: {e}")
            return False

    def _get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {}

        try:
            info = {
                "model_name": getattr(self.model, '_model_name', 'nomic-embed-text-v1.5'),
                "max_seq_length": getattr(self.model, 'max_seq_length', self.max_length),
                "device": str(self.model.device) if hasattr(self.model, 'device') else self.device
            }

            # Try to get embedding dimension from the model
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                info["embedding_dimension"] = self.model.get_sentence_embedding_dimension()
            else:
                info["embedding_dimension"] = self.embedding_dimension

            return info
        except Exception as e:
            self.logger.warning(f"Could not get model info: {e}")
            return {"error": str(e)}

    def encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings using Nomic-Embed-text-v1.5.

        Args:
            texts: List of texts to encode
            show_progress: Whether to show progress bar

        Returns:
            numpy array of embeddings
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not texts:
            return np.array([])

        start_time = time.time()

        try:
            # Clear cache before encoding
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # For Nomic-Embed, we use symmetric task configuration
            # This means we don't use instruction prefixes that might cause asymmetry

            # Encode with batch processing
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings
            )

            # Ensure embeddings are properly shaped
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            # Verify embedding dimension
            if embeddings.shape[1] != self.embedding_dimension:
                self.logger.warning(
                    f"Expected {self.embedding_dimension}D embeddings, "
                    f"got {embeddings.shape[1]}D. Using actual dimension."
                )
                self.embedding_dimension = embeddings.shape[1]

            encoding_time = time.time() - start_time
            self.embed_times.append(encoding_time)

            # Track memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
                self.memory_usage.append(memory_used)
                torch.cuda.reset_peak_memory_stats()

            self.logger.debug(
                f"Encoded {len(texts)} texts in {encoding_time:.2f}s "
                f"({len(texts)/encoding_time:.1f} texts/sec)"
            )

            return embeddings

        except Exception as e:
            self.logger.error(f"Error encoding texts: {e}")
            raise

    def compute_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two sets of embeddings.

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings

        Returns:
            Similarity scores matrix
        """
        try:
            # Ensure inputs are numpy arrays
            if torch.is_tensor(embeddings1):
                embeddings1 = embeddings1.cpu().numpy()
            if torch.is_tensor(embeddings2):
                embeddings2 = embeddings2.cpu().numpy()

            # Normalize embeddings if not already normalized
            if not self.normalize_embeddings:
                embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
                embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

            # Compute cosine similarity
            similarity_matrix = np.dot(embeddings1, embeddings2.T)

            return similarity_matrix

        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            raise

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.embed_times:
            return {}

        stats = {
            "total_embeddings": len(self.embed_times),
            "total_time": sum(self.embed_times),
            "avg_time_per_batch": np.mean(self.embed_times),
            "std_time_per_batch": np.std(self.embed_times),
            "min_time": min(self.embed_times),
            "max_time": max(self.embed_times)
        }

        if self.memory_usage:
            stats.update({
                "avg_memory_gb": np.mean(self.memory_usage),
                "max_memory_gb": max(self.memory_usage),
                "min_memory_gb": min(self.memory_usage)
            })

        return stats

    def cleanup(self):
        """Clean up model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        self.logger.info("Model cleanup completed")


class NomicEmbedEvaluator:
    """Evaluator for Nomic-Embed-text-v1.5 encoder."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.encoder_config = config['encoder']
        self.processing_config = config.get('processing', {})

        self.logger = logging.getLogger('nomic_embed_evaluator')
        self.model = None

    def initialize(self) -> bool:
        """Initialize the Nomic-Embed evaluator."""
        try:
            self.model = NomicEmbedModel(
                model_path=self.encoder_config['path'],
                device=self.processing_config.get('device', 'cuda:0'),
                **self.encoder_config.get('additional_params', {})
            )

            # Load the model
            if not self.model.load_model():
                self.logger.error("Failed to load Nomic-Embed model")
                return False

            self.logger.info("Nomic-Embed evaluator initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Nomic-Embed evaluator: {e}")
            return False

    def evaluate_single(self, ground_truth_text: str, prediction_text: str,
                       video_id: str) -> EvaluationResult:
        """Evaluate a single ground truth vs prediction pair."""
        try:
            similarity_score = self.compute_similarity(ground_truth_text, prediction_text)

            return EvaluationResult(
                video_id=video_id,
                ground_truth_text=ground_truth_text,
                prediction_text=prediction_text,
                similarity_score=similarity_score,
                success=True,
                error_message=None
            )

        except Exception as e:
            self.logger.error(f"Evaluation failed for {video_id}: {e}")
            return EvaluationResult(
                video_id=video_id,
                ground_truth_text=ground_truth_text,
                prediction_text=prediction_text,
                similarity_score=SimilarityScore(0.0, 0.5, {'error': str(e), 'fallback_score': True}),
                success=False,
                error_message=str(e)
            )

    def compute_similarity(self, ground_truth_text: str, prediction_text: str) -> SimilarityScore:
        """
        Compute similarity between ground truth and predicted text.

        Args:
            ground_truth_text: Ground truth text
            prediction_text: Predicted text

        Returns:
            Similarity score
        """
        if not ground_truth_text or not prediction_text:
            return SimilarityScore(
                cosine_similarity=0.0,
                normalized_cosine=0.5,
                metadata={'error': 'Empty text input', 'fallback_score': True}
            )

        try:
            # Encode both texts
            gt_embedding = self.model.encode_texts([ground_truth_text], show_progress=False)
            pred_embedding = self.model.encode_texts([prediction_text], show_progress=False)

            # Compute cosine similarity
            similarity_score = float(self.model.compute_similarity(gt_embedding, pred_embedding)[0, 0])

            # Calculate normalized cosine (shift from [-1, 1] to [0, 1])
            normalized_cosine = (similarity_score + 1.0) / 2.0

            return SimilarityScore(
                cosine_similarity=similarity_score,
                normalized_cosine=normalized_cosine,
                metadata={
                    'encoder_name': self.encoder_config['name'],
                    'embedding_dimension': self.model.embedding_dimension,
                    'task_type': self.model.task_type
                }
            )

        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            return SimilarityScore(
                cosine_similarity=0.0,
                normalized_cosine=0.5,
                metadata={'error': str(e), 'fallback_score': True}
            )

    def cleanup(self):
        """Clean up resources."""
        if self.model:
            self.model.cleanup()
        self.logger.info("Evaluator cleanup completed")