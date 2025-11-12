#!/usr/bin/env python3
"""
G-VEval LLaMA Evaluation Script (Standalone)

Replicates G-VEval methodology using LLaMA inference.
No dependencies on existing judge core - completely standalone.

Usage:
    # Evaluate all models from config
    python src/scripts/run_gveval_llama_evaluation.py

    # Evaluate specific models (for SLURM batching)
    python src/scripts/run_gveval_llama_evaluation.py --models internvl llava_next_video

Results are saved to:
    - Individual: results/GVEval_LLaMA/individual_results/
    - Aggregated: results/GVEval_LLaMA/aggregated_results/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.gveval_llama.gveval_llama_core import GVEvalLLaMAEvaluator

def main(models_override=None):
    """Main evaluation function.

    Args:
        models_override: Optional list of model names to evaluate instead of config
    """
    # Set memory optimization environment variables early
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    print("🎯 G-VEval LLaMA Evaluation Starting...")
    print("=" * 60)

    try:
        # Initialize G-VEval evaluator (standalone)
        # Use absolute path resolution
        script_dir = Path(__file__).parent.parent.parent  # Go back to clip-cc-bench directory
        config_path = script_dir / "src/config/gveval_llama_config.yaml"

        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            print("Please ensure the config file exists in the correct location.")
            sys.exit(1)

        evaluator = GVEvalLLaMAEvaluator(str(config_path))

        # Override models if specified via CLI
        if models_override:
            evaluator.config['models_to_evaluate'] = models_override
            print(f"📌 Evaluating specific models: {', '.join(models_override)}")

        # Load G-VEval LLaMA scorer
        evaluator.logger.info("🤖 Loading G-VEval LLaMA scorer...")
        print("🤖 Loading G-VEval LLaMA scorer...")
        evaluator.load_scorer()
        print("✅ G-VEval LLaMA scorer loaded successfully")

        # Run evaluation for all models
        evaluator.logger.info("🚀 Starting evaluation for all models...")
        print("🚀 Starting evaluation for all models...")

        all_summaries = evaluator.evaluate_all_models()
        
        # Print final results
        print("\n" + "=" * 60)
        print("🎉 G-VEVAL LLAMA EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"📊 Total models evaluated: {len(all_summaries)}")
        print(f"📁 Individual results: {evaluator.individual_results_dir}")
        print(f"📁 Aggregated results: {evaluator.aggregated_results_dir}")
        
        # Print model rankings
        if all_summaries:
            print("\n🏆 Model Rankings by G-VEval Score:")
            sorted_summaries = sorted(all_summaries, key=lambda x: x['average_gveval_score'], reverse=True)
            for i, summary in enumerate(sorted_summaries, 1):
                score = summary['average_gveval_score']
                success_rate = summary['success_rate']
                print(f"{i:2d}. {summary['model_name']:25s}: {score:6.2f}/100 (success: {success_rate:.1%})")
        
        print("\n🎯 Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ G-VEval evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import torch
        import transformers
        import yaml
        print(f"✅ Dependencies check passed:")
        print(f"   - PyTorch: {torch.__version__}")
        print(f"   - Transformers: {transformers.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def check_data_availability(config_path: str):
    """Check if required data files are available."""
    from utils.gveval_llama.gveval_config_loader import GVEvalConfigLoader
    
    # Resolve absolute path
    if not Path(config_path).is_absolute():
        script_dir = Path(__file__).parent.parent.parent
        config_path = str(script_dir / config_path)
    
    config = GVEvalConfigLoader.load_config(config_path)
    
    # Check ground truth file
    gt_file = Path(config['data_paths']['ground_truth_file'])
    if not gt_file.exists():
        print(f"❌ Ground truth file not found: {gt_file}")
        return False
    
    # Check predictions directory
    pred_dir = Path(config['data_paths']['predictions_dir'])
    if not pred_dir.exists():
        print(f"❌ Predictions directory not found: {pred_dir}")
        return False
    
    # Check for model prediction files
    models = config['models_to_evaluate']
    missing_models = []
    for model in models:
        pred_file = pred_dir / f"{model}.json"
        if not pred_file.exists():
            missing_models.append(model)
    
    if missing_models:
        print(f"⚠️  Prediction files not found for models: {missing_models}")
        print("   Evaluation will skip these models.")
    
    print(f"✅ Data availability check passed")
    print(f"   - Ground truth: {gt_file}")
    print(f"   - Predictions dir: {pred_dir}")
    print(f"   - Available models: {len(models) - len(missing_models)}/{len(models)}")
    
    return True

def run_quick_test():
    """Run a quick test to verify the system works."""
    print("\n🧪 Running quick test...")
    
    try:
        script_dir = Path(__file__).parent.parent.parent
        config_path = str(script_dir / "src/config/gveval_llama_config.yaml")
        evaluator = GVEvalLLaMAEvaluator(config_path)
        
        # Test scorer creation (don't load model for quick test)
        from utils.gveval_llama.gveval_llama_scorer import GVEvalLLaMAScorer
        scorer_config = evaluator.config['gveval_llama']
        
        print("✅ Quick test passed - system is ready")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="G-VEval LLaMA Evaluation - Evaluate video captioning models using LLaMA judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all models from config file
  python run_gveval_llama_evaluation.py

  # Evaluate specific models (for SLURM batching)
  python run_gveval_llama_evaluation.py --models internvl llava_next_video

  # Skip pre-flight checks (for repeated runs)
  python run_gveval_llama_evaluation.py --skip-checks --models Qwen2.5-72B
        """
    )
    parser.add_argument(
        '--models',
        nargs='+',
        help='Specific model(s) to evaluate (overrides config file)'
    )
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip pre-flight dependency and data checks'
    )

    args = parser.parse_args()

    print("🎯 G-VEval LLaMA Evaluation System")
    print("=" * 40)

    # Run pre-flight checks (unless skipped)
    if not args.skip_checks:
        if not check_dependencies():
            sys.exit(1)

        script_dir = Path(__file__).parent.parent.parent
        config_path = str(script_dir / "src/config/gveval_llama_config.yaml")
        if not check_data_availability(config_path):
            sys.exit(1)

        if not run_quick_test():
            sys.exit(1)
    else:
        print("⚠️  Skipping pre-flight checks as requested")

    # Run main evaluation
    main(models_override=args.models)