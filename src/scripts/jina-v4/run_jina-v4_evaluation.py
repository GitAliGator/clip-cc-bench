#!/usr/bin/env python3
"""
Jina-v4 Isolated Evaluation Script

Standalone evaluation script for Jina-v4 encoder with its own environment.
Implements 2048D embeddings with standard normalized cosine similarity.
"""

import sys
import json
import logging
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import time

# Add paths for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent.parent
sys.path.extend([
    str(src_dir / "utils" / "jina-v4"),
    str(src_dir / "utils" / "shared")
])

# Import Jina-v4 specific modules
import importlib.util
jina_v4_path = src_dir / "utils" / "jina-v4" / "embedding_models.py"
spec = importlib.util.spec_from_file_location("jina_v4_embedding_models", jina_v4_path)
jina_v4_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jina_v4_module)
JinaV4Evaluator = jina_v4_module.JinaV4Evaluator

# Import shared types
base_types_path = src_dir / "utils" / "shared" / "base_types.py"
spec = importlib.util.spec_from_file_location("base_types", base_types_path)
base_types_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_types_module)
EncoderEvaluationResult = base_types_module.EncoderEvaluationResult
SimilarityScore = base_types_module.SimilarityScore

# Import shared result manager
result_manager_path = src_dir / "utils" / "shared" / "result_manager.py"
spec = importlib.util.spec_from_file_location("result_manager", result_manager_path)
result_manager_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(result_manager_module)
SharedResultManager = result_manager_module.SharedResultManager


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')

    # Create log directory
    results_base_dir = Path(config['data_paths']['results_base_dir'])
    log_dir = results_base_dir / "encoders" / "logs" / "jina-v4"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_prefix = log_config.get('log_prefix', 'jina_v4_evaluation')
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

    logger = logging.getLogger('jina_v4_evaluation')
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_ground_truth_data(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Load ground truth data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Convert to {video_id: {"summary": text}} format
    if isinstance(data, list):
        return {item['id']: {"summary": item['summary']} for item in data}
    else:
        return data


def load_model_predictions(predictions_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load model predictions from directory."""
    predictions_path = Path(predictions_dir)
    all_predictions = {}

    for model_file in predictions_path.glob("*.json"):
        model_name = model_file.stem
        with open(model_file, 'r') as f:
            model_data = json.load(f)

        # Convert predictions to standardized format
        # Handle different JSON formats
        if isinstance(model_data, dict):
            # Direct video_id -> prediction mapping
            for video_id, prediction in model_data.items():
                if video_id not in all_predictions:
                    all_predictions[video_id] = {'predictions': {}}
                all_predictions[video_id]['predictions'][model_name] = prediction
        elif isinstance(model_data, list):
            # List of objects with video_id and prediction keys
            for item in model_data:
                video_id = item['video_id']
                if video_id not in all_predictions:
                    all_predictions[video_id] = {'predictions': {}}
                all_predictions[video_id]['predictions'][model_name] = item['prediction']

    return all_predictions


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Jina-v4 embedding evaluation on CLIP-CC-Bench"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../../config/jina-v4/encoders_config.yaml",
        help="Path to Jina-v4 configuration file"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Specific models to evaluate (default: all models in config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for evaluation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size from config"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit number of samples for testing"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    return parser.parse_args()


def validate_environment():
    """Validate that the Jina-v4 environment is properly set up."""
    try:
        import torch
        import transformers
        from sentence_transformers import SentenceTransformer

        print("✅ Core dependencies available")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   Transformers: {transformers.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"   CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")

        return True

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("🔧 Please run the Jina-v4 setup script:")
        print("   bash src/config/jina-v4/setup_jina_env.sh")
        return False




def run_evaluation_for_model(evaluator: JinaV4Evaluator,
                            ground_truth_data: Dict,
                            model_predictions: Dict,
                            model_name: str,
                            config: Dict,
                            result_manager: SharedResultManager,
                            max_samples: int = None) -> Dict[str, Any]:
    """Run evaluation for a single model."""
    logger = logging.getLogger('jina_v4_evaluation')

    logger.info(f"🔍 Evaluating model: {model_name}")

    # Filter predictions for this model
    model_data = {
        video_id: data for video_id, data in model_predictions.items()
        if model_name in data.get('predictions', {})
    }

    if not model_data:
        logger.warning(f"No predictions found for model: {model_name}")
        return {}

    # Limit samples if specified
    if max_samples:
        model_data = dict(list(model_data.items())[:max_samples])
        logger.info(f"Limited to {len(model_data)} samples for testing")

    logger.info(f"Processing {len(model_data)} samples")

    # Process each video
    evaluation_results = []
    processed_count = 0
    start_time = time.time()

    for video_id, video_data in model_data.items():
        try:
            # Get ground truth text
            if video_id not in ground_truth_data:
                logger.warning(f"No ground truth for video: {video_id}")
                continue

            ground_truth_text = ground_truth_data[video_id]['summary']

            # Get model prediction
            if model_name not in video_data.get('predictions', {}):
                logger.warning(f"No prediction for {model_name} on video: {video_id}")
                continue

            prediction_text = video_data['predictions'][model_name]

            # Evaluate using Jina-v4
            jina_result = evaluator.evaluate_single(
                ground_truth_text=ground_truth_text,
                prediction_text=prediction_text,
                video_id=video_id
            )

            # Create EncoderEvaluationResult with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            encoder_result = EncoderEvaluationResult(
                video_id=video_id,
                model_name=model_name,
                ground_truth_text=ground_truth_text,
                prediction_text=prediction_text,
                encoder_similarities={"jina-v4": jina_result.similarity_score},
                timestamp=timestamp,
                success=jina_result.success,
                error_message=jina_result.error_message
            )

            # Save using SharedResultManager
            result_manager.save_individual_result(encoder_result)

            if jina_result.success:
                evaluation_results.append({
                    'video_id': video_id,
                    'model_name': model_name,
                    'cosine_similarity': jina_result.similarity_score.cosine_similarity,
                    'normalized_cosine': jina_result.similarity_score.normalized_cosine,
                    'ground_truth_length': len(ground_truth_text),
                    'prediction_length': len(prediction_text),
                    'metadata': jina_result.similarity_score.metadata
                })
            else:
                logger.error(f"Evaluation failed for {video_id}: {jina_result.error_message}")

            processed_count += 1

            # Progress reporting
            if processed_count % config['processing'].get('progress_interval', 10) == 0:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed
                eta = (len(model_data) - processed_count) / rate if rate > 0 else 0
                logger.info(f"Progress: {processed_count}/{len(model_data)} "
                          f"({processed_count/len(model_data)*100:.1f}%) "
                          f"Rate: {rate:.1f} samples/s ETA: {eta:.0f}s")

            # Clear cache periodically
            if processed_count % config['processing'].get('clear_cache_interval', 25) == 0:
                evaluator.model.clear_cache()
                logger.debug("Cleared GPU cache")

        except Exception as e:
            logger.error(f"Error processing {video_id}: {e}")
            continue

    # Calculate aggregated statistics
    if evaluation_results:
        cosine_scores = [r['cosine_similarity'] for r in evaluation_results]
        normalized_scores = [r['normalized_cosine'] for r in evaluation_results]

        stats = {
            'model_name': model_name,
            'encoder_name': 'jina-v4',
            'total_samples': len(evaluation_results),
            'avg_cosine_similarity': float(sum(cosine_scores) / len(cosine_scores)),
            'avg_normalized_cosine': float(sum(normalized_scores) / len(normalized_scores)),
            'min_cosine_similarity': float(min(cosine_scores)),
            'max_cosine_similarity': float(max(cosine_scores)),
            'min_normalized_cosine': float(min(normalized_scores)),
            'max_normalized_cosine': float(max(normalized_scores)),
            'evaluation_time': time.time() - start_time,
            'samples_per_second': len(evaluation_results) / (time.time() - start_time),
            'encoder_stats': evaluator.get_stats()
        }

        logger.info(f"✅ {model_name} evaluation completed:")
        logger.info(f"   Samples: {stats['total_samples']}")
        logger.info(f"   Avg Normalized Cosine: {stats['avg_normalized_cosine']:.4f}")
        logger.info(f"   Time: {stats['evaluation_time']:.1f}s")
        logger.info(f"   Rate: {stats['samples_per_second']:.1f} samples/s")

        # Save aggregated summary using SharedResultManager
        # Convert evaluation_results to EncoderEvaluationResult format for summary
        encoder_results = []
        for result_data in evaluation_results:
            # Create proper SimilarityScore
            sim_score = SimilarityScore(
                cosine_similarity=result_data['cosine_similarity'],
                normalized_cosine=result_data['normalized_cosine'],
                metadata=result_data['metadata']
            )

            # Create EncoderEvaluationResult with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            encoder_result = EncoderEvaluationResult(
                video_id=result_data['video_id'],
                model_name=result_data['model_name'],
                ground_truth_text="",  # Not needed for summary
                prediction_text="",   # Not needed for summary
                encoder_similarities={"jina-v4": sim_score},
                timestamp=timestamp,
                success=True,
                error_message=None
            )
            encoder_results.append(encoder_result)

        # Save per-model summary
        result_manager.save_per_model_summary(encoder_results, model_name)

        return stats
    else:
        logger.error(f"No successful evaluations for {model_name}")
        return {}


def main():
    """Main evaluation function."""
    args = parse_arguments()

    # Validate environment
    if not validate_environment():
        sys.exit(1)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))

    # Determine base directory for path resolution (project root)
    # From src/scripts/jina-v4/run_jina-v4_evaluation.py -> clip-cc-bench/
    base_dir = Path(__file__).parent.parent.parent.parent

    # Resolve relative paths in config relative to base_dir
    data_paths = config.get('data_paths', {})
    for key in ['ground_truth_file', 'predictions_dir', 'results_base_dir']:
        if key in data_paths:
            path_value = data_paths[key]
            if not Path(path_value).is_absolute():
                # Resolve relative path from base_dir
                resolved_path = base_dir / path_value
                config['data_paths'][key] = str(resolved_path)

    # Override config with command line arguments
    if args.device:
        config['processing']['device'] = args.device
    if args.batch_size:
        config['encoder']['batch_size'] = args.batch_size
    if args.output_dir:
        config['data_paths']['results_base_dir'] = args.output_dir

    # Setup logging
    logger = setup_logging(config)

    # Initialize SharedResultManager
    results_base = Path(config['data_paths']['results_base_dir'])
    result_manager = SharedResultManager(results_base, "jina-v4")

    logger.info("🚀 Starting Jina-v4 evaluation")
    logger.info(f"📁 Results directory: {results_base / 'encoders'}")
    logger.info(f"🔧 Device: {config['processing']['device']}")
    logger.info(f"📊 Embedding dimension: {config['encoder']['additional_params']['embedding_dimension']}")

    # Initialize evaluator
    logger.info("🔧 Initializing Jina-v4 evaluator...")
    evaluator = JinaV4Evaluator(config)

    if not evaluator.initialize():
        logger.error("❌ Failed to initialize Jina-v4 evaluator")
        sys.exit(1)

    # Load data
    logger.info("📂 Loading ground truth data...")
    ground_truth_file = Path(config['data_paths']['ground_truth_file'])
    ground_truth_data = load_ground_truth_data(str(ground_truth_file))
    logger.info(f"✅ Loaded {len(ground_truth_data)} ground truth samples")

    logger.info("📂 Loading model predictions...")
    model_predictions = load_model_predictions(config['data_paths']['predictions_dir'])
    logger.info(f"✅ Loaded predictions for {len(model_predictions)} videos")

    # Determine models to evaluate
    models_to_evaluate = args.models if args.models else config['models_to_evaluate']
    logger.info(f"🎯 Evaluating {len(models_to_evaluate)} models: {models_to_evaluate}")

    # Run evaluation for each model
    all_results = {}
    total_start_time = time.time()

    for i, model_name in enumerate(models_to_evaluate, 1):
        logger.info(f"\n📋 Processing model {i}/{len(models_to_evaluate)}: {model_name}")

        try:
            model_stats = run_evaluation_for_model(
                evaluator=evaluator,
                ground_truth_data=ground_truth_data,
                model_predictions=model_predictions,
                model_name=model_name,
                config=config,
                result_manager=result_manager,
                max_samples=args.max_samples
            )

            if model_stats:
                all_results[model_name] = model_stats

        except Exception as e:
            logger.error(f"❌ Failed to evaluate {model_name}: {e}")
            continue

    # Calculate final summary
    total_time = time.time() - total_start_time

    # Print final results
    logger.info("\n" + "="*80)
    logger.info("🎉 JINA-V4 EVALUATION COMPLETED")
    logger.info("="*80)
    logger.info(f"📊 Models evaluated: {len(all_results)}")
    logger.info(f"⏱️  Total time: {total_time:.1f}s")
    logger.info(f"📁 Results saved to: {results_base / 'encoders'}")
    logger.info(f"    Individual results: {results_base / 'encoders' / 'individual_results' / 'csv' / 'jina-v4'}")
    logger.info(f"    Aggregated results: {results_base / 'encoders' / 'aggregated_results' / 'jina-v4'}")
    logger.info(f"    Logs: {results_base / 'encoders' / 'logs' / 'jina-v4'}")

    if all_results:
        logger.info("\n📈 Model Performance Summary (Normalized Cosine Similarity):")
        sorted_results = sorted(
            all_results.items(),
            key=lambda x: x[1]['avg_normalized_cosine'],
            reverse=True
        )

        for model_name, stats in sorted_results:
            logger.info(f"   {model_name:20s}: {stats['avg_normalized_cosine']:.4f}")

    # Cleanup
    evaluator.cleanup()
    logger.info("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()