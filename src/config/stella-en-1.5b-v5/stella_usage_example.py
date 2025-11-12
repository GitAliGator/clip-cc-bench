#!/usr/bin/env python3
"""
Stella-en-1.5b-v5 Usage Example and Implementation Guide

This script demonstrates how to use the Stella embedding model with sentence-transformers
for production-ready text similarity evaluation following the clip-cc-bench patterns.
"""

import os
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StellaEmbedder:
    """
    Production-ready Stella embedding class with optimized settings.

    Features:
    - Automatic device selection and memory management
    - Configurable prompts for different tasks
    - Batch processing optimization
    - Memory efficient caching
    """

    def __init__(
        self,
        model_name: str = "dunzhang/stella_en_1.5B_v5",
        device: str = None,
        batch_size: int = 16,
        max_length: int = 512,
        embedding_dimension: int = 1024,
        normalize_embeddings: bool = True
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.embedding_dimension = embedding_dimension
        self.normalize_embeddings = normalize_embeddings

        # Stella-specific prompts
        self.prompts = {
            's2s': "Instruct: Retrieve semantically similar text.\nQuery: ",
            's2p': "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "
        }

        self.model = None
        logger.info(f"Initialized Stella embedder with device: {self.device}")

    def load_model(self) -> bool:
        """Load the Stella model with optimized settings."""
        try:
            logger.info(f"Loading Stella model: {self.model_name}")

            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True  # Required for Stella models
            )

            logger.info("✅ Stella model loaded successfully")
            logger.info(f"Max sequence length: {self.model.max_seq_length}")
            logger.info(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

            return True

        except Exception as e:
            logger.error(f"Failed to load Stella model: {e}")
            return False

    def encode_with_prompts(
        self,
        texts: List[str],
        prompt_type: str = 's2s',
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts with Stella-specific prompts.

        Args:
            texts: List of texts to encode
            prompt_type: Either 's2s' (sentence-to-sentence) or 's2p' (sentence-to-passage)
            show_progress: Whether to show encoding progress

        Returns:
            Numpy array of embeddings
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Apply prompts
        if prompt_type in self.prompts:
            prompted_texts = [self.prompts[prompt_type] + text for text in texts]
        else:
            logger.warning(f"Unknown prompt type: {prompt_type}, using raw text")
            prompted_texts = texts

        # Encode with optimized settings
        embeddings = self.model.encode(
            prompted_texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )

        logger.debug(f"Encoded {len(texts)} texts to {embeddings.shape}")
        return embeddings

    def compute_similarity(
        self,
        text1: str,
        text2: str,
        prompt_type: str = 's2s'
    ) -> Dict[str, float]:
        """
        Compute similarity between two texts.

        Args:
            text1: First text (e.g., ground truth)
            text2: Second text (e.g., prediction)
            prompt_type: Prompt type to use

        Returns:
            Dictionary with similarity scores
        """
        # Encode both texts
        emb1 = self.encode_with_prompts([text1], prompt_type=prompt_type)
        emb2 = self.encode_with_prompts([text2], prompt_type=prompt_type)

        # Compute cosine similarity
        cosine_sim = float(np.dot(emb1[0], emb2[0]))

        # Normalize to [0, 1] range for evaluation
        normalized_cosine = (cosine_sim + 1) / 2

        return {
            'cosine_similarity': cosine_sim,
            'normalized_cosine': normalized_cosine
        }

    def batch_similarity(
        self,
        ground_truth_texts: List[str],
        prediction_texts: List[str],
        prompt_type: str = 's2s'
    ) -> List[Dict[str, float]]:
        """
        Compute similarities for batches of text pairs efficiently.

        Args:
            ground_truth_texts: List of ground truth texts
            prediction_texts: List of prediction texts
            prompt_type: Prompt type to use

        Returns:
            List of similarity dictionaries
        """
        # Encode all texts in batches for efficiency
        gt_embeddings = self.encode_with_prompts(ground_truth_texts, prompt_type=prompt_type)
        pred_embeddings = self.encode_with_prompts(prediction_texts, prompt_type=prompt_type)

        similarities = []
        for gt_emb, pred_emb in zip(gt_embeddings, pred_embeddings):
            cosine_sim = float(np.dot(gt_emb, pred_emb))
            normalized_cosine = (cosine_sim + 1) / 2

            similarities.append({
                'cosine_similarity': cosine_sim,
                'normalized_cosine': normalized_cosine
            })

        return similarities

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage if available."""
        if torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'reserved_gb': torch.cuda.memory_reserved() / 1024**3
            }
        return {}

    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("GPU cache cleared")


def main():
    """Demonstration of Stella model usage."""

    # Initialize embedder with optimized settings
    embedder = StellaEmbedder(
        batch_size=16,        # Recommended for most GPUs
        max_length=512,       # Optimal for Stella performance
        embedding_dimension=1024,  # Default dimension (best performance/efficiency)
        normalize_embeddings=True
    )

    # Load model
    if not embedder.load_model():
        logger.error("Failed to load model, exiting")
        return

    # Example texts for similarity computation
    ground_truth = "A person walks through a busy city street during rush hour."
    prediction = "An individual navigates crowded urban roads during peak traffic time."

    # Compute similarity for single pair
    logger.info("Computing similarity for single text pair...")
    similarity = embedder.compute_similarity(ground_truth, prediction, prompt_type='s2s')
    logger.info(f"Cosine similarity: {similarity['cosine_similarity']:.4f}")
    logger.info(f"Normalized score: {similarity['normalized_cosine']:.4f}")

    # Example batch processing
    logger.info("\nDemonstrating batch processing...")
    gt_texts = [
        "A chef prepares a delicious meal with fresh vegetables.",
        "Students study in the library for final exams.",
        "The cat sleeps peacefully on the windowsill."
    ]

    pred_texts = [
        "A cook makes a tasty dish using garden produce.",
        "Pupils prepare for end-term tests in the study hall.",
        "A feline rests quietly by the window."
    ]

    batch_similarities = embedder.batch_similarity(gt_texts, pred_texts, prompt_type='s2s')

    for i, sim in enumerate(batch_similarities):
        logger.info(f"Pair {i+1} - Cosine: {sim['cosine_similarity']:.4f}, "
                   f"Normalized: {sim['normalized_cosine']:.4f}")

    # Show memory usage
    memory_info = embedder.get_memory_usage()
    if memory_info:
        logger.info(f"\nGPU Memory - Allocated: {memory_info['allocated_gb']:.2f}GB, "
                   f"Reserved: {memory_info['reserved_gb']:.2f}GB")

    # Cleanup
    embedder.clear_cache()
    logger.info("✅ Demo completed successfully")


if __name__ == "__main__":
    main()