"""
Stella-en-1.5b-v5 Specific Embedding Model Implementation

Isolated implementation for Stella-en-1.5b-v5 encoder using sentence-transformers.
Lightweight alternative to BGE-ICL with excellent performance and lower memory requirements.
Supports MRL (Multiple Representation Learning) with configurable dimensions.
"""

import torch
import numpy as np
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import gc
import time

# Import shared types
import sys
sys.path.append(str(Path(__file__).parent.parent / "shared"))
from base_types import EmbeddingResult, EvaluationResult, SimilarityScore


class StellaModel:
    """Stella-en-1.5b-v5 model implementation using sentence-transformers."""

    def __init__(self, model_path: str, device: str = "cuda:0", **kwargs):
        self.model_path = model_path
        self.device = device
        self.batch_size = kwargs.get('batch_size', 16)  # Recommended batch size for Stella
        self.max_length = kwargs.get('max_length', 512)  # Recommended text length for Stella
        self.normalize_embeddings = kwargs.get('normalize_embeddings', True)
        self.embedding_dimension = kwargs.get('embedding_dimension', 1024)  # Default 1024d
        self.prompt_type = kwargs.get('prompt_type', 's2s')  # s2s or s2p

        self.logger = logging.getLogger('stella_model')
        self.model = None

        # Performance tracking
        self.embed_times = []
        self.memory_usage = []

        # Official Stella prompts from HuggingFace
        self.prompts = {
            's2s': "Instruct: Retrieve semantically similar text.\nQuery: ",  # Official s2s prompt
            's2p': "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: ",  # Official s2p prompt
            'discriminative': "Instruct: Given a ground truth video summary, retrieve semantically similar predicted summaries.\nQuery: "  # Similarity prompt for video evaluation
        }

    def _get_model_identifier(self) -> str:
        """Get the model identifier for loading."""
        # For Stella, prefer HuggingFace due to local format issues
        # The local model has incompatible format for sentence-transformers
        self.logger.info("Using HuggingFace model hub for Stella (better compatibility)")
        return "dunzhang/stella_en_1.5B_v5"

    def _setup_dimension_config(self, model_identifier: str):
        """Setup model for specific embedding dimension if not 1024."""
        if self.embedding_dimension != 1024:
            self.logger.info(f"Configuring model for {self.embedding_dimension}d embeddings")
            # Note: For non-1024 dimensions, users need to modify modules.json manually
            # This is documented in the research findings
            self.logger.warning(f"Non-1024d embeddings require manual modules.json modification")
            self.logger.warning(f"Replace '2_Dense_1024' with '2_Dense_{self.embedding_dimension}' in modules.json")

    def load_model(self):
        """Load Stella model using sentence-transformers."""
        try:
            model_identifier = self._get_model_identifier()
            self.logger.info(f"Loading Stella model from {model_identifier}")

            # Setup dimension configuration
            self._setup_dimension_config(model_identifier)

            # Try to load model with trust_remote_code first
            try:
                # Load model using SentenceTransformer with trust_remote_code
                self.model = SentenceTransformer(
                    model_identifier,
                    device=self.device,
                    trust_remote_code=True  # Required for Stella models
                )
            except Exception as trust_error:
                self.logger.warning(f"Failed to load with trust_remote_code: {trust_error}")
                self.logger.info("Attempting to load without trust_remote_code...")
                # Fallback to loading without trust_remote_code
                self.model = SentenceTransformer(
                    model_identifier,
                    device=self.device
                )

            # Set max sequence length
            self.model.max_seq_length = self.max_length

            self.logger.info(f"Device: {self.device}")
            self.logger.info(f"Batch size: {self.batch_size}")
            self.logger.info(f"Max length: {self.max_length}")
            self.logger.info(f"Embedding dimension: {self.embedding_dimension}")
            self.logger.info(f"Prompt type: {self.prompt_type}")
            self.logger.info(f"Normalize embeddings: {self.normalize_embeddings}")
            self.logger.info("✅ Stella model loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load Stella model: {e}")
            # Try fallback to HuggingFace if local fails
            if "stella_en_1.5B_v5" not in str(model_identifier):
                try:
                    self.logger.info("Attempting to load from HuggingFace hub...")
                    self.model = SentenceTransformer(
                        "dunzhang/stella_en_1.5B_v5",
                        device=self.device
                    )
                    self.model.max_seq_length = self.max_length
                    self.logger.info("✅ Stella model loaded from HuggingFace hub")
                    return True
                except Exception as hub_error:
                    self.logger.error(f"Failed to load from HuggingFace hub: {hub_error}")
            return False

    def _apply_prompt(self, texts: List[str]) -> List[str]:
        """Apply Stella-specific prompts to texts."""
        if self.prompt_type not in self.prompts:
            self.logger.warning(f"Unknown prompt type: {self.prompt_type}, using s2s")
            prompt = self.prompts['s2s']
        else:
            prompt = self.prompts[self.prompt_type]

        return [prompt + text for text in texts]

    def encode_texts(self, texts: List[str], apply_prompt: bool = True) -> np.ndarray:
        """Encode texts to embeddings using Stella model."""
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        try:
            # Apply prompts if requested (recommended for Stella)
            if apply_prompt:
                texts = self._apply_prompt(texts)

            # Encode texts with Stella
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            # Track performance
            embed_time = time.time() - start_time
            self.embed_times.append(embed_time)

            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                self.memory_usage.append(memory_used)

            self.logger.debug(f"Encoded {len(texts)} texts in {embed_time:.2f}s")
            return embeddings

        except Exception as e:
            self.logger.error(f"Error encoding texts: {e}")
            raise

    def compute_similarity(self, ground_truth_text: str, prediction_text: str) -> SimilarityScore:
        """Compute similarity between ground truth and prediction texts."""
        try:
            # Use discriminative prompt for similarity computation (more discriminative than s2s)
            original_prompt_type = self.prompt_type
            similarity_prompt_type = 'discriminative' if 'discriminative' in self.prompts else 's2s'
            self.prompt_type = similarity_prompt_type

            # Encode both texts with discriminative prompting
            gt_embedding = self.encode_texts([ground_truth_text], apply_prompt=True)
            pred_embedding = self.encode_texts([prediction_text], apply_prompt=True)

            # Restore original prompt type
            self.prompt_type = original_prompt_type

            # Compute cosine similarity
            cosine_sim = float(np.dot(gt_embedding[0], pred_embedding[0]))

            # Standard normalization to [0, 1] range (same as all other evaluations)
            normalized_cosine = (cosine_sim + 1) / 2

            # Create metadata
            metadata = {
                'encoder_name': 'stella-en-1.5b-v5',
                'ground_truth_length': len(ground_truth_text),
                'prediction_length': len(prediction_text),
                'embedding_dim': gt_embedding.shape[1] if len(gt_embedding.shape) > 1 else len(gt_embedding),
                'computation_time': self.embed_times[-1] if self.embed_times else 0.0,
                'batch_size': self.batch_size,
                'max_length': self.max_length,
                'prompt_type': similarity_prompt_type,
                'normalize_embeddings': self.normalize_embeddings,
                'sentence_transformers_version': True
            }

            return SimilarityScore(
                cosine_similarity=cosine_sim,
                normalized_cosine=normalized_cosine,  # Standard normalized cosine similarity
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
            'batch_size': self.batch_size,
            'embedding_dimension': self.embedding_dimension,
            'prompt_type': self.prompt_type
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


class StellaEvaluator:
    """Isolated evaluator for Stella-en-1.5b-v5 model."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('stella_evaluator')
        self.model = None

        # Initialize model
        encoder_config = config['encoder']
        processing_config = config.get('processing', {})

        self.model = StellaModel(
            model_path=encoder_config['path'],
            device=processing_config.get('device', 'cuda:0'),
            batch_size=encoder_config['batch_size'],
            max_length=encoder_config['max_length'],
            normalize_embeddings=encoder_config.get('additional_params', {}).get('normalize_embeddings', True),
            embedding_dimension=encoder_config.get('additional_params', {}).get('embedding_dimension', 1024),
            prompt_type=encoder_config.get('additional_params', {}).get('prompt_type', 's2s')
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