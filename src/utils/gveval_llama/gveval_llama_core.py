"""
G-VEval LLaMA Core

Main evaluator class that orchestrates the G-VEval evaluation process using LLaMA.
Handles data loading, evaluation execution, and results saving.
"""

from pathlib import Path
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import torch
import csv
from .gveval_llama_scorer import GVEvalLLaMAScorer, GVEvalResult
from .gveval_config_loader import GVEvalConfigLoader

class GVEvalLLaMAEvaluator:
    """Standalone G-VEval evaluator using LLaMA (no external dependencies)."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path  # Store for parallel worker creation
        self.config = GVEvalConfigLoader.load_config(config_path)
        GVEvalConfigLoader.validate_config(self.config)
        
        self.scorer = None
        
        # Set up results directories with the new structure
        self.base_results_dir = Path(self.config['data_paths']['results_base_dir'])
        self.gveval_results_dir = self.base_results_dir / "g-veval"
        self.individual_results_dir = self.gveval_results_dir / "individual_results"
        self.individual_json_dir = self.individual_results_dir / "json"
        self.individual_csv_dir = self.individual_results_dir / "csv"
        self.aggregated_results_dir = self.gveval_results_dir / "aggregated_results"
        
        # Create directories
        self.individual_json_dir.mkdir(parents=True, exist_ok=True)
        self.individual_csv_dir.mkdir(parents=True, exist_ok=True)
        self.aggregated_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging for G-VEval evaluation."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use configuration-based log directory
        log_dir = Path(self.config.get('logging', {}).get('log_dir', 'results/g-veval/logs'))
        if not log_dir.is_absolute():
            # Make it relative to the project root
            log_dir = Path(__file__).parent.parent.parent.parent / log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"gveval_llama_evaluation_{timestamp}.log"
        
        # Create logger
        logger = logging.getLogger('gveval_llama')
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        logger.info(f"G-VEval LLaMA evaluation started. Log file: {log_file}")
        return logger
        
    def load_scorer(self):
        """Load G-VEval LLaMA scorer."""
        scorer_config = self.config['gveval_llama']
        self.scorer = GVEvalLLaMAScorer(scorer_config, self.logger)
        self.scorer.load_model()
        
        # Load prompt template
        template_path = self._get_prompt_template_path()
        self.scorer.load_prompt_template(str(template_path))
        
    def _get_prompt_template_path(self) -> Path:
        """Get the full path to the prompt template."""
        base_path = Path(__file__).parent
        template_relative_path = self.config['gveval_llama']['prompt_template_path']
        return base_path / template_relative_path
        
    def load_ground_truth(self) -> List[Dict]:
        """Load ground truth data."""
        ground_truth_file = Path(self.config['data_paths']['ground_truth_file'])
        if not ground_truth_file.exists():
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_file}")
        
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
        
        self.logger.info(f"Loaded {len(ground_truth)} ground truth samples")
        return ground_truth
        
    def load_model_predictions(self, model_name: str) -> List[Dict]:
        """Load model predictions."""
        predictions_dir = Path(self.config['data_paths']['predictions_dir'])
        pred_file = predictions_dir / f"{model_name}.json"
        
        if not pred_file.exists():
            raise FileNotFoundError(f"Predictions file not found: {pred_file}")
        
        with open(pred_file, 'r') as f:
            predictions_raw = json.load(f)
        
        # Convert dictionary format to list of dicts
        if isinstance(predictions_raw, dict):
            predictions = [
                {"video_id": video_id, "caption": caption}
                for video_id, caption in predictions_raw.items()
            ]
        else:
            predictions = predictions_raw
        
        self.logger.info(f"Loaded {len(predictions)} predictions for {model_name}")
        return predictions
        
    def evaluate_model_predictions(self, model_name: str, predictions: List[Dict], 
                                 ground_truth: List[Dict]) -> Dict[str, Any]:
        """Evaluate a model's predictions using G-VEval methodology."""
        if self.logger:
            self.logger.info(f"Starting G-VEval evaluation for model: {model_name}")
        
        results = []
        total_score = 0.0
        criterion_totals = {criterion: 0.0 for criterion in self.scorer.accr_criteria}
        successful_evaluations = 0
        
        checkpoint_interval = self.config['processing'].get('checkpoint_interval', 25)
        
        # Align predictions and ground truth by video_id if available
        aligned_data = self._align_predictions_and_ground_truth(predictions, ground_truth)
        
        for i, (pred, gt) in enumerate(aligned_data):
            if self.logger and (i + 1) % checkpoint_interval == 0:
                self.logger.info(f"Processed {i + 1}/{len(aligned_data)} samples")
            
            # Extract data with fallbacks for different formats
            video_id = pred.get('video_id', gt.get('id', gt.get('video_id', str(i))))
            generated_caption = pred.get('caption', pred.get('description', pred.get('summary', '')))
            reference_caption = gt.get('summary', gt.get('caption', gt.get('description', '')))
            
            if not generated_caption or not reference_caption:
                self.logger.warning(f"Skipping sample {video_id}: missing caption data")
                continue
            
            # Evaluate using G-VEval
            result = self.scorer.evaluate_ref_only(
                video_id=video_id,
                reference_caption=reference_caption,
                generated_caption=generated_caption,
                rubric_type=self.config['gveval_llama']['rubric_type']
            )
            
            results.append(result)
            
            if result.success:
                successful_evaluations += 1
                total_score += result.overall_score
                
                # Accumulate criterion scores
                for criterion, score in result.accr_scores.items():
                    criterion_totals[criterion] += score
            else:
                self.logger.warning(f"Evaluation failed for {video_id}: {result.error_message}")
        
        # Calculate averages
        num_samples = len(results)
        success_rate = successful_evaluations / num_samples if num_samples > 0 else 0
        average_score = total_score / successful_evaluations if successful_evaluations > 0 else 0
        criterion_averages = {
            criterion: total / successful_evaluations if successful_evaluations > 0 else 0
            for criterion, total in criterion_totals.items()
        }
        
        # Create summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = {
            "model_name": model_name,
            "total_samples": num_samples,
            "successful_evaluations": successful_evaluations,
            "success_rate": success_rate,
            "average_gveval_score": average_score,
            "accr_criterion_averages": criterion_averages,
            "timestamp": timestamp,
            "evaluation_method": "gveval_llama_ref_only",
            "rubric_type": self.config['gveval_llama']['rubric_type'],
            "score_range": f"{self.config['gveval_llama']['score_range'][0]}-{self.config['gveval_llama']['score_range'][1]}",
            "model_path": self.config['gveval_llama']['model_path']
        }
        
        # Save results
        self._save_results(model_name, results, summary, timestamp)
        
        if self.logger:
            self.logger.info(f"G-VEval evaluation completed for {model_name}")
            self.logger.info(f"Success rate: {success_rate:.2%}")
            self.logger.info(f"Average score: {average_score:.2f}/{self.config['gveval_llama']['score_range'][1]}")
        
        return summary
    
    def _align_predictions_and_ground_truth(self, predictions: List[Dict], 
                                          ground_truth: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """Align predictions with ground truth by video_id or index."""
        aligned_data = []
        
        # Try to align by video_id first
        gt_by_id = {}
        for gt in ground_truth:
            video_id = gt.get('id') or gt.get('video_id')
            if video_id:
                gt_by_id[video_id] = gt
        
        if gt_by_id:
            # Align by video_id
            for pred in predictions:
                video_id = pred.get('video_id')
                if video_id and video_id in gt_by_id:
                    aligned_data.append((pred, gt_by_id[video_id]))
                else:
                    self.logger.warning(f"No ground truth found for video_id: {video_id}")
        else:
            # Align by index
            min_length = min(len(predictions), len(ground_truth))
            aligned_data = list(zip(predictions[:min_length], ground_truth[:min_length]))
            
            if len(predictions) != len(ground_truth):
                self.logger.warning(f"Prediction and ground truth lengths differ: {len(predictions)} vs {len(ground_truth)}")
        
        self.logger.info(f"Aligned {len(aligned_data)} prediction-ground truth pairs")
        return aligned_data
    
    def _save_results(self, model_name: str, results: List[GVEvalResult], 
                     summary: Dict[str, Any], timestamp: str):
        """Save individual and aggregated results in both JSON and CSV formats."""
        
        # Save individual results JSON
        individual_json_file = self.individual_json_dir / f"{model_name}_gveval_individual_{timestamp}.json"
        individual_data = [self._result_to_dict(result) for result in results]
        
        with open(individual_json_file, 'w') as f:
            json.dump(individual_data, f, indent=2)
        
        # Save individual results CSV
        individual_csv_file = self.individual_csv_dir / f"{model_name}_gveval_individual_{timestamp}.csv"
        self._save_individual_csv(individual_csv_file, results)
        
        # Save aggregated summary JSON
        summary_file = self.aggregated_results_dir / f"{model_name}_gveval_summary_{timestamp}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Update aggregated CSV
        self._update_aggregated_csv(model_name, summary)
        
        self.logger.info(f"Individual JSON results saved to: {individual_json_file}")
        self.logger.info(f"Individual CSV results saved to: {individual_csv_file}")
        self.logger.info(f"Summary saved to: {summary_file}")
        self.logger.info(f"Aggregated CSV updated")
    
    def _result_to_dict(self, result: GVEvalResult) -> Dict[str, Any]:
        """Convert GVEvalResult to dictionary for JSON serialization."""
        return {
            "video_id": result.video_id,
            "reference_caption": result.reference_caption,
            "generated_caption": result.generated_caption,
            "accr_scores": result.accr_scores,
            "overall_score": result.overall_score,
            "reasoning": result.reasoning,
            "raw_response": result.raw_response,
            "success": result.success,
            "error_message": result.error_message
        }
    
    def _save_individual_csv(self, csv_file: Path, results: List[GVEvalResult]):
        """Save individual results to CSV format."""
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['video_id', 'accuracy', 'completeness', 'conciseness', 'relevance', 'overall_score'])
            
            # Write data rows
            for result in results:
                if result.success:
                    row = [
                        result.video_id,
                        round(result.accr_scores.get('accuracy', 0.0), 2),
                        round(result.accr_scores.get('completeness', 0.0), 2),
                        round(result.accr_scores.get('conciseness', 0.0), 2),
                        round(result.accr_scores.get('relevance', 0.0), 2),
                        round(result.overall_score, 2)
                    ]
                else:
                    # For failed evaluations, use 0.0 scores
                    row = [result.video_id, 0.0, 0.0, 0.0, 0.0, 0.0]
                writer.writerow(row)
    
    def _update_aggregated_csv(self, model_name: str, summary: Dict[str, Any]):
        """Update the aggregated results CSV file."""
        aggr_csv_file = self.aggregated_results_dir / "aggr_results.csv"
        
        # Read existing data if file exists
        existing_data = []
        if aggr_csv_file.exists():
            with open(aggr_csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_data = [row for row in reader if row['Model Name'] != model_name]
        
        # Add/update current model data
        new_row = {
            'Model Name': model_name,
            'accuracy': round(summary['accr_criterion_averages'].get('accuracy', 0.0), 2),
            'completeness': round(summary['accr_criterion_averages'].get('completeness', 0.0), 2),
            'conciseness': round(summary['accr_criterion_averages'].get('conciseness', 0.0), 2),
            'relevance': round(summary['accr_criterion_averages'].get('relevance', 0.0), 2),
            'overall_score': round(summary['average_gveval_score'], 2)
        }
        existing_data.append(new_row)
        
        # Sort by overall_score descending
        existing_data.sort(key=lambda x: float(x['overall_score']), reverse=True)
        
        # Write updated data
        with open(aggr_csv_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Model Name', 'accuracy', 'completeness', 'conciseness', 'relevance', 'overall_score']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_data)
    
    def evaluate_all_models(self) -> List[Dict[str, Any]]:
        """Evaluate all models specified in configuration."""
        ground_truth = self.load_ground_truth()
        models_to_evaluate = self.config['models_to_evaluate']
        
        self.logger.info(f"Starting G-VEval evaluation for {len(models_to_evaluate)} models")
        
        # Check if parallel processing is enabled
        max_workers = self.config['processing'].get('max_workers', 1)
        use_parallel = max_workers > 1 and self.config['processing'].get('model_level_parallelization', False)
        
        # Force sequential processing for 70B models due to memory constraints
        if use_parallel:
            self.logger.warning("Parallel processing requested but disabled due to 70B model memory constraints.")
            self.logger.info("Using memory-optimized sequential processing")
            all_summaries = self._evaluate_models_sequential(models_to_evaluate, ground_truth)
        else:
            self.logger.info("Using sequential processing")
            all_summaries = self._evaluate_models_sequential(models_to_evaluate, ground_truth)
        
        # Save overall summary
        self._save_overall_summary(all_summaries)
        
        return all_summaries
    
    def _evaluate_models_sequential(self, models_to_evaluate: List[str], ground_truth: List[Dict]) -> List[Dict[str, Any]]:
        """Memory-optimized sequential model evaluation."""
        all_summaries = []
        for i, model_name in enumerate(models_to_evaluate, 1):
            self.logger.info(f"Evaluating model {i}/{len(models_to_evaluate)}: {model_name}")
            
            try:
                # Clear GPU cache before each model evaluation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.info(f"GPU memory cleared for {model_name}")
                
                predictions = self.load_model_predictions(model_name)
                summary = self.evaluate_model_predictions(model_name, predictions, ground_truth)
                all_summaries.append(summary)
                
                self.logger.info(f"Completed {model_name}: Avg Score = {summary['average_gveval_score']:.2f}")
                
                # Force garbage collection after each model
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {model_name}: {e}")
                # Clean up memory even on failure
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
                continue
        
        return all_summaries
    
    def _evaluate_models_parallel(self, models_to_evaluate: List[str], ground_truth: List[Dict], max_workers: int) -> List[Dict[str, Any]]:
        """Memory-optimized sequential evaluation (parallel processing disabled due to memory constraints)."""
        self.logger.warning("Parallel processing disabled due to 70B model memory constraints. Using sequential processing.")
        return self._evaluate_models_sequential(models_to_evaluate, ground_truth)
    
    def _save_overall_summary(self, all_summaries: List[Dict[str, Any]]):
        """Save overall evaluation summary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        overall_summary_file = self.aggregated_results_dir / f"overall_gveval_summary_{timestamp}.json"
        
        overall_summary = {
            "evaluation_method": "gveval_llama_ref_only",
            "total_models_evaluated": len(all_summaries),
            "timestamp": timestamp,
            "rubric_type": self.config['gveval_llama']['rubric_type'],
            "model_summaries": all_summaries,
            "results_structure": {
                "individual_results_dir": str(self.individual_results_dir),
                "aggregated_results_dir": str(self.aggregated_results_dir)
            },
            "configuration": {
                "llama_model_path": self.config['gveval_llama']['model_path'],
                "evaluation_mode": self.config['gveval_llama']['evaluation_mode'],
                "score_range": self.config['gveval_llama']['score_range'],
                "criteria": self.config['gveval_llama']['criteria']
            }
        }
        
        with open(overall_summary_file, 'w') as f:
            json.dump(overall_summary, f, indent=2)
        
        self.logger.info(f"Overall summary saved to: {overall_summary_file}")
        
        # Log model rankings
        if all_summaries:
            sorted_summaries = sorted(all_summaries, key=lambda x: x['average_gveval_score'], reverse=True)
            self.logger.info("\nModel Rankings by G-VEval Score:")
            for i, summary in enumerate(sorted_summaries, 1):
                self.logger.info(f"{i}. {summary['model_name']}: {summary['average_gveval_score']:.2f}")