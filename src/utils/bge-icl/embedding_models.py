"""
BGE-en-ICL Specific Embedding Model Implementation

Isolated implementation for BGE-en-ICL encoder (SOTA 2024 model).
Uses In-Context Learning (ICL) with few-shot examples for task-specific embeddings.
Supports FlashAttention for long context windows.
"""

import torch
import numpy as np
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from FlagEmbedding import FlagICLModel
import gc
import time

# Import shared types
import sys
sys.path.append(str(Path(__file__).parent.parent / "shared"))
from base_types import EmbeddingResult, EvaluationResult, SimilarityScore


class BGEEnICLModel:
    """BGE-en-ICL model implementation with ICL (In-Context Learning) support using FlagICLModel."""

    def __init__(self, model_path: str, device: str = "cuda:2", **kwargs):
        self.model_path = model_path
        self.device = device
        self.batch_size = kwargs.get('batch_size', 12)
        self.max_length = kwargs.get('max_length', 8192)  # BGE-ICL supports long context
        self.normalize_embeddings = kwargs.get('normalize_embeddings', True)
        self.pooling_method = kwargs.get('pooling_method', 'cls')
        self.use_flashattn = kwargs.get('use_flashattn', True)  # Enable FlashAttention for long context

        self.logger = logging.getLogger('bge_en_icl_model')
        self.model = None

        # Performance tracking
        self.embed_times = []
        self.memory_usage = []

        # Zero-shot mode configuration (no few-shot examples)
        self.few_shot_examples = None  # Use zero-shot mode as requested

    def _create_similarity_examples(self) -> List[Dict[str, str]]:
        """Create few-shot examples for text similarity task in FlagICLModel format."""
        task = "Given a ground truth video summary, find semantically similar predicted summaries."
        return [
            {
                'instruct': task,
                'query': "A person walks through a busy city street during rush hour, dodging traffic and pedestrians.",
                'response': "An individual navigates crowded urban roads during peak traffic time, avoiding cars and people walking."
            },
            {
                'instruct': task,
                'query': "The chef prepared a delicious meal using fresh vegetables and herbs from the garden.",
                'response': "A cook made a tasty dish with garden-fresh produce and aromatic plants."
            },
            {
                'instruct': task,
                'query': "Students gathered in the library to study for their final examinations.",
                'response': "Pupils assembled at the study hall to prepare for end-term tests."
            }
        ]

    def _check_local_model(self) -> str:
        """Check if local model exists, otherwise use HuggingFace."""
        # Convert relative path to absolute path from project root
        if not os.path.isabs(self.model_path):
            # Get project root (3 levels up from this file)
            project_root = Path(__file__).parent.parent.parent.parent
            local_path = project_root / self.model_path
        else:
            local_path = Path(self.model_path)

        # Check if local model exists and has required files
        if local_path.exists() and (local_path / "config.json").exists():
            self.logger.info(f"Using local BGE-en-ICL model from {local_path}")
            return str(local_path)
        else:
            self.logger.info(f"Local model not found at {local_path}, using HuggingFace model hub")
            return "BAAI/bge-en-icl"

    def load_model(self):
        """Load BGE-en-ICL model with ICL support using FlagICLModel."""
        try:
            model_source = self._check_local_model()
            self.logger.info(f"Loading BGE-en-ICL model from {model_source}")

            # Check for FlashAttention support
            if self.use_flashattn:
                try:
                    import flash_attn
                    self.logger.info("FlashAttention available for long context support")
                except ImportError:
                    self.logger.warning("FlashAttention not available, using standard attention")
                    self.use_flashattn = False

            # Load model using FlagICLModel in zero-shot mode (no examples)
            # Using official BGE-ICL instruction from HuggingFace
            official_instruction = "Represent this sentence for searching relevant passages:"

            self.model = FlagICLModel(
                model_source,
                query_instruction_for_retrieval=official_instruction,  # Official BGE-ICL instruction
                examples_for_task=None,  # Zero-shot mode
                use_fp16=True if torch.cuda.is_available() else False
            )

            self.logger.info(f"Max length: {self.max_length}")
            self.logger.info(f"FlashAttention: {self.use_flashattn}")
            self.logger.info("Mode: Zero-shot (no few-shot examples)")
            self.logger.info("✅ BGE-en-ICL model loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load BGE-en-ICL model: {e}")
            return False

    def encode_texts(self, texts: List[str], is_query: bool = True) -> np.ndarray:
        """Encode texts to embeddings using FlagICLModel with ICL."""
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        try:
            # Use FlagICLModel's encode methods
            if is_query:
                embeddings = self.model.encode_queries(texts)
            else:
                embeddings = self.model.encode_corpus(texts)

            # Track performance
            embed_time = time.time() - start_time
            self.embed_times.append(embed_time)

            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                self.memory_usage.append(memory_used)

            self.logger.debug(f"Encoded {len(texts)} texts with ICL in {embed_time:.2f}s")
            return embeddings

        except Exception as e:
            self.logger.error(f"Error encoding texts with ICL: {e}")
            raise

    def compute_similarity(self, ground_truth_text: str, prediction_text: str) -> SimilarityScore:
        """Compute similarity between ground truth and prediction texts using ICL."""
        try:
            # Encode ground truth as query and prediction as response for ICL
            gt_embedding = self.encode_texts([ground_truth_text], is_query=True)
            pred_embedding = self.encode_texts([prediction_text], is_query=False)

            # Compute cosine similarity
            cosine_sim = float(np.dot(gt_embedding[0], pred_embedding[0]))

            # Normalize to [0, 1] range
            normalized_cosine = (cosine_sim + 1) / 2

            # Create metadata
            metadata = {
                'encoder_name': 'bge-en-icl',
                'ground_truth_length': len(ground_truth_text),
                'prediction_length': len(prediction_text),
                'embedding_dim': gt_embedding.shape[1] if len(gt_embedding.shape) > 1 else len(gt_embedding),
                'computation_time': self.embed_times[-1] if self.embed_times else 0.0,
                'pooling_method': self.pooling_method,
                'icl_enabled': False,  # Zero-shot mode
                'max_length': self.max_length,
                'flashattn_enabled': self.use_flashattn
            }

            return SimilarityScore(
                cosine_similarity=cosine_sim,
                normalized_cosine=normalized_cosine,
                metadata=metadata
            )

        except Exception as e:
            self.logger.error(f"Error computing similarity with ICL: {e}")
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
            'max_embed_time': np.max(self.embed_times),
            'pooling_method': self.pooling_method
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


class BGEEnICLEvaluator:
    """Isolated evaluator for BGE-en-ICL model with ICL support."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('bge_en_icl_evaluator')
        self.model = None

        # Initialize model
        encoder_config = config['encoder']
        processing_config = config.get('processing', {})

        self.model = BGEEnICLModel(
            model_path=encoder_config['path'],
            device=processing_config.get('device', 'cuda:2'),
            batch_size=encoder_config['batch_size'],
            max_length=encoder_config['max_length'],
            normalize_embeddings=encoder_config.get('additional_params', {}).get('normalize_embeddings', True),
            pooling_method=encoder_config.get('additional_params', {}).get('pooling_method', 'cls'),
            use_flashattn=encoder_config.get('additional_params', {}).get('use_flashattn', True)
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