#!/usr/bin/env python3
"""
Nomic-Embed-text-v1.5 Isolated Evaluation Script

Standalone evaluation script for Nomic-Embed-text-v1.5 encoder with its own environment.
Implements 768D embeddings with symmetric classification task configuration.
"""

import sys
import json
import logging
import argparse
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add paths for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent.parent
sys.path.extend([
    str(src_dir / "utils" / "nomic-embed-text-v1.5"),
    str(src_dir / "utils" / "shared")
])

from embedding_models import NomicEmbedEvaluator
from config_loader import IsolatedEncoderConfigLoader
from result_manager import SharedResultManager
from base_types import EncoderEvaluationResult, SimilarityScore


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')

    # Use log_dir from config or create from results_base_dir
    if 'log_dir' in log_config:
        log_dir = Path(log_config['log_dir'])
    elif 'paths' in config and 'logs_dir' in config['paths']:
        log_dir = Path(config['paths']['logs_dir'])
    else:
        results_base_dir = Path(config['data_paths']['results_base_dir'])
        log_dir = results_base_dir / "encoders" / "logs" / "nomic-embed-text-v1.5"

    log_dir.mkdir(parents=True, exist_ok=True)

    log_prefix = log_config.get('log_prefix', 'nomic_embed_evaluation')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{log_prefix}_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger('nomic_embed_evaluation')
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")
    return logger


def load_ground_truth_data(ground_truth_file: Path) -> List[Dict[str, Any]]:
    """Load ground truth dataset."""
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_predictions(predictions_dir: Path, model_name: str) -> Dict[str, str]:
    """Load predictions for a specific model."""
    prediction_file = predictions_dir / f"{model_name}.json"

    if not prediction_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_file}")

    with open(prediction_file, 'r') as f:
        data = json.load(f)

    # Convert to flat format: {video_id: text}
    if isinstance(data, list):
        # Ground truth format: [{"id": "001", "summary": "text"}, ...]
        return {item['id']: item['summary'] for item in data}
    elif isinstance(data, dict):
        # Check if it's already in the expected format
        if data and isinstance(next(iter(data.values())), dict):
            # Format: {video_id: {"summary": text}}
            return {video_id: item['summary'] for video_id, item in data.items()}
        else:
            # Flat format: {video_id: text}
            return data
    else:
        return {}


def main():
    parser = argparse.ArgumentParser(description='Nomic-Embed-text-v1.5 Isolated Evaluation')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--models', nargs='+', help='Specific models to evaluate (overrides config)')
    parser.add_argument('--base-dir', help='Base directory path (defaults to project root)')

    args = parser.parse_args()

    # Determine base directory
    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        # Default to project root (3 levels up from script)
        base_dir = Path(__file__).parent.parent.parent

    # Load configuration
    config_loader = IsolatedEncoderConfigLoader('nomic-embed-text-v1.5', base_dir)
    config = config_loader.load_config()

    # Setup logging
    logger = setup_logging(config)
    logger.info("🚀 Starting Nomic-Embed-text-v1.5 evaluation")
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Config: {config_loader.paths.get_config_file()}")

    # Load ground truth data
    logger.info("Loading ground truth data...")
    ground_truth_file = Path(config['data_paths']['ground_truth_file'])
    ground_truth_data = load_ground_truth_data(ground_truth_file)
    logger.info(f"Loaded {len(ground_truth_data)} ground truth samples")

    # Initialize result manager
    results_base_dir = Path(config['data_paths']['results_base_dir'])
    result_manager = SharedResultManager(results_base_dir, "nomic-embed-text-v1.5")

    # Initialize evaluator
    logger.info("Initializing Nomic-Embed-text-v1.5 evaluator...")
    evaluator = NomicEmbedEvaluator(config)

    if not evaluator.initialize():
        logger.error("Failed to initialize Nomic-Embed evaluator")
        return {'success': False, 'error': 'Evaluator initialization failed'}

    logger.info("✅ Nomic-Embed evaluator initialized successfully")

    # Get models to evaluate
    models_to_evaluate = args.models or config.get('models_to_evaluate', [])
    predictions_dir = Path(config['data_paths']['predictions_dir'])

    # Track results
    all_results = []
    model_results = {}  # Group results by model
    total_evaluations = 0
    successful_evaluations = 0
    processing_config = config.get('processing', {})
    clear_cache_interval = processing_config.get('clear_cache_interval', 25)

    logger.info(f"Starting evaluation for {len(models_to_evaluate)} models")

    for model_idx, model_name in enumerate(models_to_evaluate, 1):
        logger.info(f"[{model_idx}/{len(models_to_evaluate)}] Processing model: {model_name}")

        try:
            # Load predictions for this model
            predictions = load_predictions(predictions_dir, model_name)
            logger.info(f"Loaded {len(predictions)} predictions for {model_name}")

            model_successful = 0
            model_results[model_name] = []  # Initialize results for this model

            # Process each video
            for video_idx, gt_item in enumerate(ground_truth_data, 1):
                video_id = gt_item['id']  # Ground truth uses 'id', not 'video_id'
                ground_truth_text = gt_item['summary']  # Ground truth uses 'summary', not 'text'

                # Get prediction for this video
                if video_id not in predictions:
                    logger.warning(f"No prediction found for video {video_id} in model {model_name}")
                    continue

                prediction_text = predictions[video_id]
                total_evaluations += 1

                # Run evaluation
                try:
                    embedding_result = evaluator.evaluate_single(
                        ground_truth_text, prediction_text, video_id
                    )

                    # Create encoder evaluation result
                    encoder_result = EncoderEvaluationResult(
                        video_id=video_id,
                        model_name=model_name,
                        ground_truth_text=ground_truth_text,
                        prediction_text=prediction_text,
                        encoder_similarities={'nomic-embed-text-v1.5': embedding_result.similarity_score},
                        success=embedding_result.success,
                        error_message=embedding_result.error_message,
                        timestamp=datetime.now().isoformat()
                    )

                    # Save result
                    all_results.append(encoder_result)
                    model_results[model_name].append(encoder_result)  # Add to model-specific results
                    if encoder_result.success:
                        successful_evaluations += 1
                        model_successful += 1

                    # Progress logging
                    if video_idx % processing_config.get('progress_interval', 10) == 0:
                        success_rate = (model_successful / video_idx) * 100
                        logger.info(f"  Progress: {video_idx}/{len(ground_truth_data)} "
                                   f"({success_rate:.1f}% success)")

                    # Clear cache periodically (only clear CUDA cache, don't unload model)
                    if video_idx % clear_cache_interval == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.debug(f"CUDA cache cleared after {video_idx} evaluations")

                except Exception as e:
                    logger.error(f"Failed to evaluate {video_id} for model {model_name}: {e}")
                    continue

            logger.info(f"✅ Completed {model_name}: {model_successful}/{len(ground_truth_data)} successful")

        except FileNotFoundError as e:
            logger.error(f"Skipping model {model_name}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error processing model {model_name}: {e}")
            continue

    # Save individual results
    for result in all_results:
        result_manager.save_individual_result(result)

    # Save per-model summaries
    for model_name, results in model_results.items():
        if results:
            result_manager.save_per_model_summary(results, model_name)
            logger.info(f"📊 Saved per-model summary for {model_name}")

    # Generate overall summary
    logger.info("📊 Generating evaluation summary...")
    overall_summary = result_manager.create_encoder_summary(all_results)
    logger.info(f"📈 Overall Results:")
    logger.info(f"  Total evaluations: {total_evaluations}")
    logger.info(f"  Successful evaluations: {successful_evaluations}")
    logger.info(f"  Success rate: {(successful_evaluations/total_evaluations)*100:.1f}%")
    logger.info(f"  Models evaluated: {len(models_to_evaluate)}")

    # Cleanup
    evaluator.cleanup()
    logger.info("🎉 Nomic-Embed-text-v1.5 evaluation completed successfully!")

    return {
        'success': True,
        'total_evaluations': total_evaluations,
        'successful_evaluations': successful_evaluations,
        'models_evaluated': len(models_to_evaluate),
        'results_saved': len(all_results)
    }


if __name__ == "__main__":
    result = main()
    if not result['success']:
        sys.exit(1)