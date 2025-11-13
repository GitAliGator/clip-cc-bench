"""
NV-Embed Specific Embedding Model Implementation

Isolated implementation for NV-Embed encoder with optimized settings.
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoModel, AutoTokenizer
import gc
import time

# Import shared types
import sys
sys.path.append(str(Path(__file__).parent.parent / "shared"))
from base_types import EmbeddingResult, EvaluationResult, SimilarityScore

# Add local NV-Embed implementation path dynamically
from paths import get_project_paths
_project_paths = get_project_paths()
_nv_embed_path = str(_project_paths.get_encoder_models_dir() / "nv-embed")
sys.path.append(_nv_embed_path)
from modeling_nvembed import NVEmbedModel as LocalNVEmbedModel


class NVEmbedModel:
    """NV-Embed model implementation with optimized settings."""

    def __init__(self, model_path: str, device: str = "cuda:0", **kwargs):
        self.model_path = model_path
        self.device = device
        self.batch_size = kwargs.get('batch_size', 8)
        self.max_length = kwargs.get('max_length', 32768)
        self.instruction_for_retrieval = kwargs.get('instruction_for_retrieval',
            "Given a web search query, retrieve relevant passages that answer the query.")

        self.logger = logging.getLogger('nv_embed_model')
        self.model = None
        self.tokenizer = None

        # Performance tracking
        self.embed_times = []
        self.memory_usage = []

    def load_model(self):
        """Load NV-Embed model with local implementation."""
        try:
            self.logger.info(f"Loading NV-Embed model from {self.model_path}")

            # Load the local NV-Embed v2 model
            self.model = LocalNVEmbedModel.from_pretrained(
                self.model_path,
                device_map={"" : self.device},
                trust_remote_code=True,
                torch_dtype=torch.float16
            )

            # Set model to eval mode
            self.model.eval()
            self.model = self.model.to(self.device)

            self.logger.info("✅ NV-Embed model loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load NV-Embed model: {e}")
            return False

    def prepare_texts(self, texts: List[str], add_instruction: bool = True) -> List[str]:
        """Prepare texts with instruction prefix if needed."""
        # The local NV-Embed implementation handles instructions internally
        # so we just return the texts as-is
        return texts

    def encode_texts(self, texts: List[str], add_instruction: bool = True) -> np.ndarray:
        """Encode texts to embeddings."""
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        try:
            # Prepare texts with instruction if needed
            prepared_texts = self.prepare_texts(texts, add_instruction)

            # Generate embeddings using local NV-Embed implementation
            with torch.no_grad():
                embeddings = self.model.encode(
                    prepared_texts,
                    instruction=self.instruction_for_retrieval if add_instruction else "",
                    max_length=self.max_length
                )

                # Convert to numpy array
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.cpu().numpy()

                # Normalize embeddings
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            # Track performance
            embed_time = time.time() - start_time
            self.embed_times.append(embed_time)

            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
                self.memory_usage.append(memory_used)

            self.logger.debug(f"Encoded {len(texts)} texts in {embed_time:.2f}s")
            return embeddings

        except Exception as e:
            self.logger.error(f"Error encoding texts: {e}")
            raise

    def compute_similarity(self, ground_truth_text: str, prediction_text: str) -> SimilarityScore:
        """Compute similarity between ground truth and prediction texts."""
        try:
            # Encode both texts
            gt_embedding = self.encode_texts([ground_truth_text], add_instruction=False)
            pred_embedding = self.encode_texts([prediction_text], add_instruction=False)

            # Compute cosine similarity
            cosine_sim = float(np.dot(gt_embedding[0], pred_embedding[0]))

            # Normalize to [0, 1] range
            normalized_cosine = (cosine_sim + 1) / 2

            # Create metadata
            metadata = {
                'encoder_name': 'nv-embed',
                'ground_truth_length': len(ground_truth_text),
                'prediction_length': len(prediction_text),
                'embedding_dim': gt_embedding.shape[1] if len(gt_embedding.shape) > 1 else len(gt_embedding),
                'computation_time': self.embed_times[-1] if self.embed_times else 0.0
            }

            return SimilarityScore(
                cosine_similarity=cosine_sim,
                normalized_cosine=normalized_cosine,
                metadata=metadata
            )

        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            # Return fallback similarity score on error (should be marked as failure)
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
            'max_embed_time': np.max(self.embed_times)
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


class NVEmbedEvaluator:
    """Isolated evaluator for NV-Embed model."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('nv_embed_evaluator')
        self.model = None

        # Initialize model
        encoder_config = config['encoder']
        self.model = NVEmbedModel(
            model_path=encoder_config['path'],
            device=config['processing']['device'],
            batch_size=encoder_config['batch_size'],
            max_length=encoder_config['max_length'],
            instruction_for_retrieval=encoder_config.get('additional_params', {}).get('instruction_for_retrieval', "")
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