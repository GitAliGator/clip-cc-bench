"""
E5-Mistral-7B-Instruct Specific Embedding Model Implementation

Isolated implementation for E5-Mistral-7B-Instruct encoder using sentence-transformers.
Memory-optimized for 7B parameter model with instruction-based embeddings.
Supports long context (4096 tokens) with conservative memory management.
"""

import torch
import numpy as np
import logging
import os
import gc
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer

# Import shared types
import sys
sys.path.append(str(Path(__file__).parent.parent / "shared"))
from base_types import EmbeddingResult, EvaluationResult, SimilarityScore


class E5MistralModel:
    """E5-Mistral-7B-Instruct model implementation using sentence-transformers with memory optimization."""

    def __init__(self, model_path: str, device: str = "cuda:0", **kwargs):
        self.model_path = model_path
        self.device = device

        # Conservative settings for 7B model
        self.batch_size = kwargs.get('batch_size', 2)  # Very conservative for 7B
        self.max_length = kwargs.get('max_length', 4096)  # E5-Mistral supports long context
        self.normalize_embeddings = kwargs.get('normalize_embeddings', True)
        self.use_fp16 = kwargs.get('use_fp16', True)  # Essential for 7B model
        self.embedding_dimension = 4096  # Fixed for E5-Mistral-7B

        self.logger = logging.getLogger('e5_mistral_model')
        self.model = None

        # Performance tracking
        self.embed_times = []
        self.memory_usage = []

        # E5-Mistral specific instruction prompts
        self.instruction_prompts = {
            'query': "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: ",
            'retrieval': "Instruct: Retrieve semantically similar text.\nQuery: ",
            'similarity': "Instruct: Given a ground truth text, find semantically similar predicted text.\nQuery: ",
            'classification': "Instruct: Classify the following text.\nQuery: ",
            'clustering': "Instruct: Identify the main topic of the following text.\nQuery: "
        }

        # Default instruction type for this use case
        self.default_instruction = kwargs.get('instruction_type', 'similarity')

    def _get_model_identifier(self) -> str:
        """Get the model identifier for loading, with local model detection."""
        # Check if local model exists first
        if not os.path.isabs(self.model_path):
            # Convert relative path to absolute path from project root
            project_root = Path(__file__).parent.parent.parent.parent
            local_path = project_root / self.model_path
        else:
            local_path = Path(self.model_path)

        # Check if local model exists and has required files
        if local_path.exists() and (local_path / "config.json").exists():
            self.logger.info(f"Using local E5-Mistral model from {local_path}")
            return str(local_path)
        else:
            self.logger.info(f"Local model not found at {local_path}, using HuggingFace model hub")
            return "intfloat/e5-mistral-7b-instruct"

    def _check_gpu_memory(self) -> Dict[str, float]:
        """Check available GPU memory before loading model."""
        if not torch.cuda.is_available():
            return {"available": 0, "total": 0}

        try:
            available = torch.cuda.memory_reserved() / 1024**3  # GB
            total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3  # GB
            free = total - available

            self.logger.info(f"GPU Memory - Total: {total:.1f}GB, Available: {free:.1f}GB")

            if free < 12:  # Need ~14GB for float16, warn if less than 12GB available
                self.logger.warning(f"Low GPU memory: {free:.1f}GB available. E5-Mistral-7B needs ~14GB.")
                self.logger.warning("Consider using smaller batch sizes or quantization.")

            return {"available": free, "total": total, "used": available}

        except Exception as e:
            self.logger.warning(f"Could not check GPU memory: {e}")
            return {"available": 0, "total": 0}

    def load_model(self):
        """Load E5-Mistral-7B-Instruct model with compatibility handling."""
        try:
            # Check GPU memory before loading
            memory_info = self._check_gpu_memory()

            model_identifier = self._get_model_identifier()
            self.logger.info(f"Loading E5-Mistral-7B-Instruct model from {model_identifier}")

            # Set memory optimization environment variables
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

            # Clear any existing cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Try multiple loading strategies for compatibility
            self.model = self._load_model_with_fallback(model_identifier)

            if self.model is None:
                return False

            # Set model to evaluation mode and enable fp16 if requested
            self.model.eval()
            if self.use_fp16 and torch.cuda.is_available():
                self.model.half()

            self.logger.info(f"Device: {self.device}")
            self.logger.info(f"Batch size: {self.batch_size}")
            self.logger.info(f"Max length: {self.max_length}")
            self.logger.info(f"Embedding dimension: {self.embedding_dimension}")
            self.logger.info(f"FP16 enabled: {self.use_fp16}")
            self.logger.info(f"Default instruction: {self.default_instruction}")
            self.logger.info("✅ E5-Mistral-7B-Instruct model loaded successfully")

            # Check memory usage after loading
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                self.logger.info(f"Model loaded - GPU memory used: {memory_used:.1f}GB")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load E5-Mistral-7B-Instruct model: {e}")
            return False

    def _load_model_with_fallback(self, model_identifier: str):
        """Load model with multiple fallback strategies for compatibility."""
        # Strategy 1: Try loading local model with relaxed configuration
        if str(model_identifier).startswith('/'):
            self.logger.info("Strategy 1: Loading local model with compatibility mode")
            try:
                # Use basic configuration for local model
                model = SentenceTransformer(
                    model_identifier,
                    device=self.device,
                    trust_remote_code=True,
                    # Remove advanced kwargs that might cause compatibility issues
                )
                self.logger.info("✅ Local model loaded successfully with compatibility mode")
                return model
            except Exception as e:
                self.logger.warning(f"Strategy 1 failed: {e}")
                self.logger.info("Trying Strategy 2...")

        # Strategy 2: Force HuggingFace download (latest compatible version)
        self.logger.info("Strategy 2: Loading from HuggingFace with latest compatible version")
        try:
            model = SentenceTransformer(
                "intfloat/e5-mistral-7b-instruct",
                device=self.device,
                trust_remote_code=True
            )
            self.logger.info("✅ HuggingFace model loaded successfully")
            return model
        except Exception as e:
            self.logger.warning(f"Strategy 2 failed: {e}")
            self.logger.info("Trying Strategy 3...")

        # Strategy 3: Load with minimal configuration
        self.logger.info("Strategy 3: Loading with minimal configuration")
        try:
            model = SentenceTransformer(
                "intfloat/e5-mistral-7b-instruct",
                device='cpu',  # Start on CPU then move to GPU
                trust_remote_code=True
            )
            # Move to GPU after loading
            if torch.cuda.is_available() and self.device != 'cpu':
                model = model.to(self.device)
            self.logger.info("✅ Model loaded with minimal configuration")
            return model
        except Exception as e:
            self.logger.error(f"Strategy 3 failed: {e}")
            self.logger.error("All loading strategies failed")
            return None

    def _apply_instruction(self, texts: List[str], instruction_type: str = None) -> List[str]:
        """Apply E5-Mistral specific instructions to texts."""
        if instruction_type is None:
            instruction_type = self.default_instruction

        if instruction_type not in self.instruction_prompts:
            self.logger.warning(f"Unknown instruction type: {instruction_type}, using 'similarity'")
            instruction_type = 'similarity'

        instruction = self.instruction_prompts[instruction_type]
        return [instruction + text for text in texts]

    def encode_texts(self, texts: List[str], apply_instruction: bool = True,
                    instruction_type: str = None, batch_size: int = None) -> np.ndarray:
        """Encode texts to embeddings using E5-Mistral model with memory management."""
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        # Use provided batch size or default
        effective_batch_size = batch_size if batch_size is not None else self.batch_size

        try:
            # Apply instructions if requested (recommended for E5-Mistral)
            if apply_instruction:
                texts = self._apply_instruction(texts, instruction_type)

            # Clear cache before encoding for memory optimization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Encode texts with E5-Mistral
            embeddings = self.model.encode(
                texts,
                batch_size=effective_batch_size,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
                device=self.device
            )

            # Track performance
            embed_time = time.time() - start_time
            self.embed_times.append(embed_time)

            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                self.memory_usage.append(memory_used)

            self.logger.debug(f"Encoded {len(texts)} texts in {embed_time:.2f}s (batch_size={effective_batch_size})")

            return embeddings

        except Exception as e:
            self.logger.error(f"Error encoding texts: {e}")
            # Clear cache on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    def compute_similarity(self, ground_truth_text: str, prediction_text: str) -> SimilarityScore:
        """Compute similarity between ground truth and prediction texts using E5-Mistral."""
        try:
            # Use similarity instruction for both texts
            gt_embedding = self.encode_texts([ground_truth_text],
                                           apply_instruction=True,
                                           instruction_type='similarity',
                                           batch_size=1)

            # For the prediction text, we can use a different instruction or none
            # In this case, let's use the same similarity instruction for consistency
            pred_embedding = self.encode_texts([prediction_text],
                                             apply_instruction=True,
                                             instruction_type='similarity',
                                             batch_size=1)

            # Compute cosine similarity
            cosine_sim = float(np.dot(gt_embedding[0], pred_embedding[0]))

            # Normalize to [0, 1] range for scoring
            normalized_cosine = (cosine_sim + 1) / 2

            # Create metadata
            metadata = {
                'encoder_name': 'e5-mistral-7b-instruct',
                'ground_truth_length': len(ground_truth_text),
                'prediction_length': len(prediction_text),
                'embedding_dim': gt_embedding.shape[1] if len(gt_embedding.shape) > 1 else len(gt_embedding),
                'computation_time': self.embed_times[-1] if self.embed_times else 0.0,
                'batch_size': self.batch_size,
                'max_length': self.max_length,
                'use_fp16': self.use_fp16,
                'instruction_type': 'similarity',
                'normalize_embeddings': self.normalize_embeddings,
                'model_parameters': '7B',
                'model_layers': 32
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
        """Clear GPU cache and perform garbage collection for 7B model."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            self.logger.debug("Cache cleared for 7B model")
        except Exception as e:
            self.logger.warning(f"Error clearing cache: {e}")

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
            'max_length': self.max_length,
            'use_fp16': self.use_fp16,
            'model_parameters': '7B'
        }

        if self.memory_usage:
            stats.update({
                'avg_memory_usage_gb': np.mean(self.memory_usage),
                'max_memory_usage_gb': np.max(self.memory_usage),
                'min_memory_usage_gb': np.min(self.memory_usage)
            })

        return stats

    def __del__(self):
        """Cleanup resources for 7B model."""
        try:
            if hasattr(self, 'model') and self.model:
                del self.model
            self.clear_cache()
        except Exception as e:
            # Ignore cleanup errors during destruction
            pass


class E5MistralEvaluator:
    """Isolated evaluator for E5-Mistral-7B-Instruct model."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('e5_mistral_evaluator')
        self.model = None

        # Initialize model
        encoder_config = config['encoder']
        processing_config = config.get('processing', {})

        self.model = E5MistralModel(
            model_path=encoder_config['path'],
            device=processing_config.get('device', 'cuda:0'),
            batch_size=encoder_config['batch_size'],
            max_length=encoder_config['max_length'],
            normalize_embeddings=encoder_config.get('additional_params', {}).get('normalize_embeddings', True),
            use_fp16=encoder_config.get('additional_params', {}).get('use_fp16', True),
            instruction_type=encoder_config.get('additional_params', {}).get('instruction_type', 'similarity')
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