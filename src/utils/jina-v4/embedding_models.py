"""
Jina-v4 Specific Embedding Model Implementation

Isolated implementation for Jina-v4 embeddings with optimized 2048D settings.
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
from base_types import EmbeddingResult, EvaluationResult, SimilarityScore

# Import path management
from paths import get_project_paths


class JinaV4Model:
    """Jina-v4 model implementation with optimized 2048D embeddings."""

    def __init__(self, model_path: str, device: str = "cuda:0", **kwargs):
        self.model_path = model_path
        self.device = device
        self.batch_size = kwargs.get('batch_size', 8)
        self.max_length = kwargs.get('max_length', 32768)
        self.embedding_dimension = kwargs.get('embedding_dimension', 2048)
        self.task_type = kwargs.get('task_type', 'text-matching')
        self.torch_dtype = kwargs.get('torch_dtype', 'bfloat16')
        self.use_matryoshka = kwargs.get('use_matryoshka', True)
        self.instruction_for_retrieval = kwargs.get('instruction_for_retrieval',
            "Given a ground truth video summary, assess the quality and alignment of predicted summaries with fine-grained discrimination.")

        self.logger = logging.getLogger('jina_v4_model')
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
            local_model_path = project_paths.get_encoder_models_dir() / "jina-embeddings-v4"

            # Check if local model exists and has model files
            if local_model_path.exists():
                # Check for key model files
                config_file = local_model_path / "config.json"
                custom_st_file = local_model_path / "custom_st.py"

                # Check for various model file patterns Jina-v4 might use
                safetensors_files = list(local_model_path.glob("*.safetensors"))
                model_files = list(local_model_path.glob("*.bin"))

                if config_file.exists() and custom_st_file.exists() and (safetensors_files or model_files):
                    self.logger.info(f"Found local Jina-v4 model at: {local_model_path}")
                    self.logger.info(f"  config.json: ✓")
                    self.logger.info(f"  custom_st.py: ✓")
                    self.logger.info(f"  Model files: {len(safetensors_files + model_files)} files")
                    return str(local_model_path)
                else:
                    self.logger.info("Local model directory exists but incomplete, falling back to remote")
                    self.logger.info(f"  config.json: {'✓' if config_file.exists() else '✗'}")
                    self.logger.info(f"  custom_st.py: {'✓' if custom_st_file.exists() else '✗'}")
                    self.logger.info(f"  Model files: {len(safetensors_files + model_files)} files")
            else:
                self.logger.info("No local model found, will use remote HuggingFace model")

            # Fall back to the original HuggingFace path
            return self.model_path

        except Exception as e:
            self.logger.warning(f"Error checking for local model: {e}, using remote path")
            return self.model_path

    def load_model(self):
        """Load Jina-v4 model using SentenceTransformer with local fallback."""
        try:
            # Try to determine the best model path
            model_path_to_use = self._get_model_path()
            self.logger.info(f"Loading Jina-v4 model from {model_path_to_use}")

            # If using local model, add its directory to Python path for custom_st import
            if model_path_to_use != self.model_path:  # Local path being used
                import sys
                from pathlib import Path
                local_model_dir = str(Path(model_path_to_use))
                if local_model_dir not in sys.path:
                    sys.path.insert(0, local_model_dir)
                    self.logger.info(f"Added {local_model_dir} to Python path for custom modules")

            # Configure model kwargs for Jina-v4
            model_kwargs = {
                'trust_remote_code': True,
                'torch_dtype': getattr(torch, self.torch_dtype, torch.bfloat16)
            }

            # Load the model using SentenceTransformer
            # This automatically handles Jina-v4's architecture and task adapters
            self.model = SentenceTransformer(
                model_path_to_use,
                device=self.device,
                model_kwargs=model_kwargs,
                trust_remote_code=True  # Required for Jina-v4's custom code
            )

            # Configure for text-matching task (symmetric embeddings)
            # This ensures we get consistent embeddings for both ground truth and predictions
            if hasattr(self.model, 'prompts'):
                # Use text-matching prompt for symmetric embeddings
                self.model.prompts = {"text-matching": ""}

            self.logger.info("✅ Jina-v4 model loaded successfully")
            self.logger.info(f"📊 Model max sequence length: {self.model.max_seq_length}")
            self.logger.info(f"🔢 Embedding dimension: {self.embedding_dimension}")
            self.logger.info(f"🎯 Task type: {self.task_type}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load Jina-v4 model: {e}")
            return False

    def encode_texts(self, texts: List[str], add_instruction: bool = False) -> np.ndarray:
        """Encode texts to 2048D embeddings using Jina-v4."""
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        try:
            # Configure encode parameters for maximum capability
            encode_kwargs = {
                'batch_size': self.batch_size,
                'normalize_embeddings': True,  # Always normalize for cosine similarity
                'convert_to_numpy': True,
                'show_progress_bar': False
            }

            # For Jina-v4, ensure task is specified as required by the model
            # Based on the error message, we need to pass 'task' parameter
            encode_kwargs['task'] = 'text-matching'

            # Add instruction prefix if specified and model supports it
            processed_texts = texts
            if add_instruction and self.instruction_for_retrieval:
                # Jina-v4 can handle instructions directly through task_type
                # For text-matching, we typically don't need instruction prefixes
                pass

            # Generate embeddings with full 2048D capability
            embeddings = self.model.encode(
                processed_texts,
                **encode_kwargs
            )

            # Ensure we get the full 2048D embeddings (no truncation)
            if embeddings.shape[1] != self.embedding_dimension:
                self.logger.warning(f"Expected {self.embedding_dimension}D embeddings, got {embeddings.shape[1]}D")

            # Ensure embeddings are normalized
            if not encode_kwargs.get('normalize_embeddings', False):
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            # Track performance
            embed_time = time.time() - start_time
            self.embed_times.append(embed_time)

            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
                self.memory_usage.append(memory_used)

            self.logger.debug(f"Encoded {len(texts)} texts in {embed_time:.2f}s to {embeddings.shape[1]}D")
            return embeddings

        except Exception as e:
            self.logger.error(f"Error encoding texts: {e}")
            raise

    def compute_similarity(self, ground_truth_text: str, prediction_text: str) -> SimilarityScore:
        """Compute similarity between ground truth and prediction texts."""
        try:
            # Encode both texts using text-matching task (symmetric embeddings)
            gt_embedding = self.encode_texts([ground_truth_text], add_instruction=False)
            pred_embedding = self.encode_texts([prediction_text], add_instruction=False)

            # Compute cosine similarity
            cosine_sim = float(np.dot(gt_embedding[0], pred_embedding[0]))

            # Normalize to [0, 1] range: (cosine_sim + 1) / 2
            # This is the standard normalization used across all encoders
            normalized_cosine = (cosine_sim + 1) / 2

            # Create metadata
            metadata = {
                'encoder_name': 'jina-v4',
                'ground_truth_length': len(ground_truth_text),
                'prediction_length': len(prediction_text),
                'embedding_dim': gt_embedding.shape[1],
                'task_type': self.task_type,
                'max_seq_length': self.max_length,
                'computation_time': self.embed_times[-1] if self.embed_times else 0.0
            }

            return SimilarityScore(
                cosine_similarity=cosine_sim,
                normalized_cosine=normalized_cosine,
                metadata=metadata
            )

        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            # Return fallback similarity score on error
            return SimilarityScore(
                cosine_similarity=0.0,
                normalized_cosine=0.5,
                metadata={'error': str(e), 'fallback_score': True}
            )

    def clear_cache(self):
        """Clear GPU cache and perform garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.logger.debug("Cache cleared")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.embed_times:
            return {}

        stats = {
            'total_embeddings': len(self.embed_times),
            'avg_embed_time': np.mean(self.embed_times),
            'total_embed_time': np.sum(self.embed_times),
            'min_embed_time': np.min(self.embed_times),
            'max_embed_time': np.max(self.embed_times),
            'embedding_dimension': self.embedding_dimension,
            'max_sequence_length': self.max_length
        }

        if self.memory_usage:
            stats.update({
                'avg_memory_usage_gb': np.mean(self.memory_usage),
                'max_memory_usage_gb': np.max(self.memory_usage),
                'min_memory_usage_gb': np.min(self.memory_usage)
            })

        return stats

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'model') and self.model:
            del self.model
        self.clear_cache()


class JinaV4Evaluator:
    """Isolated evaluator for Jina-v4 model."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('jina_v4_evaluator')
        self.model = None

        # Initialize model
        encoder_config = config['encoder']
        additional_params = encoder_config.get('additional_params', {})

        self.model = JinaV4Model(
            model_path=encoder_config['path'],
            device=config['processing']['device'],
            batch_size=encoder_config['batch_size'],
            max_length=encoder_config['max_length'],
            embedding_dimension=additional_params.get('embedding_dimension', 2048),
            task_type=additional_params.get('task_type', 'text-matching'),
            torch_dtype=additional_params.get('torch_dtype', 'bfloat16'),
            use_matryoshka=additional_params.get('use_matryoshka', True),
            instruction_for_retrieval=additional_params.get('instruction_for_retrieval', "")
        )

    def initialize(self) -> bool:
        """Initialize the evaluator."""
        return self.model.load_model()

    def evaluate_single(self, ground_truth_text: str, prediction_text: str,
                       video_id: str) -> EvaluationResult:
        """Evaluate a single ground truth vs prediction pair."""
        try:
            similarity_score = self.model.compute_similarity(ground_truth_text, prediction_text)

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

    def cleanup(self):
        """Cleanup resources."""
        if self.model:
            self.model.clear_cache()

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if self.model:
            return self.model.get_performance_stats()
        return {}