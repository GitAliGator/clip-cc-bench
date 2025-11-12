#!/usr/bin/env python3
"""
N-Gram Metrics Evaluation Script

Traditional text generation metrics evaluation for clip-cc-bench:
- BLEU-1, BLEU-4
- ROUGE-1, ROUGE-4, ROUGE-L, ROUGE-Lsum  
- METEOR

Usage:
    python src/scripts/run_ngram_evaluation.py

Results are saved to:
    - Individual: results/n-gram/individual_results/
    - Aggregated: results/n-gram/aggregated_results/
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.ngram_metrics.ngram_core import NgramEvaluator

def main():
    """Main evaluation function."""
    print("📊 N-Gram Metrics Evaluation Starting...")
    print("=" * 60)
    
    try:
        # Initialize N-Gram evaluator
        # Use absolute path resolution
        script_dir = Path(__file__).parent.parent.parent  # Go back to clip-cc-bench directory
        config_path = script_dir / "src/config/ngram_config.yaml"
        
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            print("Please ensure the config file exists in the correct location.")
            sys.exit(1)
        
        evaluator = NgramEvaluator(str(config_path))
        
        # Run evaluation for all models
        evaluator.logger.info("🚀 Starting evaluation for all models...")
        print("🚀 Starting evaluation for all models...")
        
        all_summaries = evaluator.evaluate_all_models()
        
        # Print final results
        print("\n" + "=" * 60)
        print("🎉 N-GRAM METRICS EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"📊 Total models evaluated: {len(all_summaries)}")
        print(f"📁 Individual results: {evaluator.individual_results_dir}")
        print(f"📁 Aggregated results: {evaluator.aggregated_results_dir}")
        
        # Print model rankings
        if all_summaries:
            print("\n🏆 Model Rankings by Average Metrics:")
            
            # Sort by first metric (typically BLEU-1)
            target_metrics = evaluator.target_metrics
            if target_metrics:
                sort_metric = target_metrics[0]
                sorted_summaries = sorted(all_summaries, 
                                        key=lambda x: x.get('metric_averages', {}).get(sort_metric, 0), 
                                        reverse=True)
                
                # Print header
                print(f"{'Rank':>4} {'Model Name':<25} {' '.join(f'{m:>8}' for m in target_metrics)}")
                print("-" * (35 + 9 * len(target_metrics)))
                
                for i, summary in enumerate(sorted_summaries, 1):
                    model_name = summary['model_name']
                    averages = summary.get('metric_averages', {})
                    success_rate = summary.get('success_rate', 0)
                    
                    # Format metric scores
                    metric_strs = []
                    for metric in target_metrics:
                        score = averages.get(metric, 0.0)
                        metric_strs.append(f"{score:8.3f}")
                    
                    print(f"{i:>4} {model_name:<25} {' '.join(metric_strs)} (success: {success_rate:.1%})")
        
        print("\n📊 Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ N-gram evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import nltk
        import rouge_score
        import yaml
        print(f"✅ Dependencies check passed:")
        print(f"   - NLTK: {nltk.__version__}")
        print(f"   - ROUGE Score: Available")
        print(f"   - PyYAML: Available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages:")
        print("   pip install nltk rouge-score pyyaml")
        return False

def check_nltk_data():
    """Check and download required NLTK data."""
    try:
        import nltk
        required_data = ['punkt', 'wordnet', 'averaged_perceptron_tagger']
        
        missing_data = []
        for data_name in required_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}')
            except LookupError:
                missing_data.append(data_name)
        
        if missing_data:
            print(f"📥 Downloading required NLTK data: {missing_data}")
            for data_name in missing_data:
                try:
                    nltk.download(data_name, quiet=True)
                    print(f"   ✅ Downloaded: {data_name}")
                except:
                    print(f"   ❌ Failed to download: {data_name}")
                    return False
        
        print("✅ NLTK data check passed")
        return True
        
    except Exception as e:
        print(f"❌ NLTK data check failed: {e}")
        return False

def check_data_availability(config_path: str):
    """Check if required data files are available."""
    from utils.ngram_metrics.ngram_config_loader import NgramConfigLoader
    
    # Resolve absolute path
    if not Path(config_path).is_absolute():
        script_dir = Path(__file__).parent.parent.parent
        config_path = str(script_dir / config_path)
    
    config = NgramConfigLoader.load_config(config_path)
    
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
    available_models = []
    for model in models:
        pred_file = pred_dir / f"{model}.json"
        if not pred_file.exists():
            missing_models.append(model)
        else:
            available_models.append(model)
    
    if missing_models:
        print(f"⚠️  Prediction files not found for models: {missing_models}")
        print("   Evaluation will skip these models.")
    
    if not available_models:
        print(f"❌ No prediction files found for any models")
        return False
    
    print(f"✅ Data availability check passed")
    print(f"   - Ground truth: {gt_file}")
    print(f"   - Predictions dir: {pred_dir}")
    print(f"   - Available models: {len(available_models)}/{len(models)}")
    
    return True

def run_quick_test():
    """Run a quick test to verify the system works."""
    print("\n🧪 Running quick test...")
    
    try:
        script_dir = Path(__file__).parent.parent.parent
        config_path = str(script_dir / "src/config/ngram_config.yaml")
        
        # Test config loading
        from utils.ngram_metrics.ngram_config_loader import NgramConfigLoader
        config = NgramConfigLoader.load_config(config_path)
        
        # Test traditional metrics import
        from utils.ngram_metrics.traditional_metrics import evaluate_traditional_metrics
        
        # Quick metric test
        test_ref = "This is a test reference sentence."
        test_gen = "This is a test generated sentence."
        metrics = evaluate_traditional_metrics(test_ref, test_gen)
        
        if 'BLEU-1' in metrics and 'ROUGE-1' in metrics:
            print("✅ Quick test passed - system is ready")
            return True
        else:
            print("❌ Quick test failed - metrics not computed correctly")
            return False
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

if __name__ == "__main__":
    print("📊 N-Gram Metrics Evaluation System")
    print("=" * 40)
    
    # Run pre-flight checks
    if not check_dependencies():
        sys.exit(1)
    
    if not check_nltk_data():
        sys.exit(1)
    
    script_dir = Path(__file__).parent.parent.parent
    config_path = str(script_dir / "src/config/ngram_config.yaml")
    if not check_data_availability(config_path):
        sys.exit(1)
    
    if not run_quick_test():
        sys.exit(1)
    
    # Run main evaluation
    main()