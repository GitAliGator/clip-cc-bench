"""
En-Vision Specific Embedding Model Implementation

Direct model loading implementation for En-Vision encoder with optimized settings.
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

# Add local En-Vision implementation path dynamically
from paths import get_project_paths
_project_paths = get_project_paths()
_envision_path = str(_project_paths.get_encoder_models_dir() / "en-vision")
sys.path.append(_envision_path)


class EnVisionModel:
    """En-Vision model implementation with direct model loading."""

    def __init__(self, model_path: str, device: str = "cuda:0", **kwargs):
        self.model_path = model_path
        self.device = device
        self.batch_size = kwargs.get('batch_size', 16)
        self.max_length = kwargs.get('max_length', 4096)
        self.normalize_embeddings = kwargs.get('normalize_embeddings', True)

        self.logger = logging.getLogger('envision_model')
        self.model = None
        self.tokenizer = None

        # Performance tracking
        self.embed_times = []
        self.memory_usage = []

    def load_model(self):
        """Load En-Vision model with direct model loading."""
        try:
            self.logger.info(f"Loading En-Vision model from {self.model_path}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False  # Use slow tokenizer to avoid issues
            )

            # Load the model
            self.model = AutoModel.from_pretrained(
                self.model_path,
                device_map={"": self.device},
                trust_remote_code=True,
                torch_dtype=torch.float16
            )

            # Set model to eval mode
            self.model.eval()
            self.model = self.model.to(self.device)

            self.logger.info("✅ En-Vision model loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load En-Vision model: {e}")
            return False

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings using direct model inference."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        try:
            # Tokenize texts
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)

                # Get embeddings from sentence_embedding output
                # En-Vision model returns a dict with 'sentence_embedding' key
                if isinstance(outputs, dict) and 'sentence_embedding' in outputs:
                    embeddings = outputs['sentence_embedding']
                else:
                    # Fallback to mean pooling if sentence_embedding not available
                    embeddings = outputs.last_hidden_state.mean(dim=1)

                # Normalize embeddings if required
                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # Convert to numpy
                embeddings = embeddings.cpu().numpy()

            # Track performance
            embed_time = time.time() - start_time
            self.embed_times.append(embed_time)

            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
                self.memory_usage.append(memory_used)

            self.logger.debug(f"Encoded {len(texts)} texts in {embed_time:.2f}s, shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            self.logger.error(f"Error encoding texts: {e}")
            raise

    def compute_similarity(self, ground_truth_text: str, prediction_text: str) -> SimilarityScore:
        """Compute similarity between ground truth and prediction texts."""
        try:
            # Encode both texts
            gt_embedding = self.encode_texts([ground_truth_text])
            pred_embedding = self.encode_texts([prediction_text])

            # Compute cosine similarity
            cosine_sim = float(np.dot(gt_embedding[0], pred_embedding[0]))

            # Normalize to [0, 1] range
            normalized_cosine = (cosine_sim + 1) / 2

            # Create metadata
            metadata = {
                'encoder_name': 'envision',
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
        if hasattr(self, 'tokenizer') and self.tokenizer:
            del self.tokenizer
        self.clear_cache()


class EnVisionEvaluator:
    """Isolated evaluator for En-Vision model."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('envision_evaluator')
        self.model = None

        # Initialize model
        encoder_config = config['encoder']
        self.model = EnVisionModel(
            model_path=encoder_config['path'],
            device=config['processing']['device'],
            batch_size=encoder_config['batch_size'],
            max_length=encoder_config['max_length'],
            normalize_embeddings=encoder_config.get('additional_params', {}).get('normalize_embeddings', True)
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