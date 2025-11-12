"""
N-Gram Metrics Core

Main evaluator class that orchestrates the n-gram evaluation process.
Handles data loading, evaluation execution, and results saving following clip-cc-bench patterns.
"""

from pathlib import Path
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
import os
import csv
from dataclasses import dataclass
import gc

from .traditional_metrics import evaluate_traditional_metrics
from .ngram_config_loader import NgramConfigLoader

@dataclass
class NgramResult:
    """Container for n-gram evaluation results from a single video."""
    video_id: str
    reference_summary: str
    generated_summary: str
    metrics: Dict[str, float]
    success: bool = True
    error_message: str = ""

class NgramEvaluator:
    """N-Gram metrics evaluator for clip-cc-bench."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = NgramConfigLoader.load_config(config_path)
        
        # Set up results directories with the clip-cc-bench structure
        self.base_results_dir = Path(self.config['data_paths']['results_base_dir'])
        self.ngram_results_dir = self.base_results_dir / "n-gram"
        self.individual_results_dir = self.ngram_results_dir / "individual_results"
        self.individual_json_dir = self.individual_results_dir / "json"
        self.individual_csv_dir = self.individual_results_dir / "csv"
        self.aggregated_results_dir = self.ngram_results_dir / "aggregated_results"
        
        # Create directories
        self.individual_json_dir.mkdir(parents=True, exist_ok=True)
        self.individual_csv_dir.mkdir(parents=True, exist_ok=True)
        self.aggregated_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Get metrics configuration
        self.target_metrics = self.config['ngram_metrics']['metrics']
        self.decimal_precision = self.config['ngram_metrics'].get('decimal_precision', 2)
        
    def _setup_logger(self):
        """Setup logging for n-gram evaluation."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use configuration-based log directory
        log_dir = Path(self.config.get('logging', {}).get('log_dir', 'results/n-gram/logs'))
        if not log_dir.is_absolute():
            # Make it relative to the project root
            log_dir = Path(__file__).parent.parent.parent.parent / log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"ngram_evaluation_{timestamp}.log"
        
        # Create logger
        logger = logging.getLogger('ngram_metrics')
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
        
        logger.info(f"N-Gram evaluation started. Log file: {log_file}")
        return logger
        
    def load_ground_truth(self) -> List[Dict]:
        """Load ground truth data from clip-cc dataset."""
        ground_truth_file = Path(self.config['data_paths']['ground_truth_file'])
        
        if not ground_truth_file.exists():
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_file}")
        
        self.logger.info(f"Loading ground truth from: {ground_truth_file}")
        
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.logger.info(f"Loaded {len(data)} ground truth entries")
        return data
    
    def load_model_predictions(self, model_name: str) -> List[Dict]:
        """Load model predictions from JSON file."""
        predictions_dir = Path(self.config['data_paths']['predictions_dir'])
        prediction_file = predictions_dir / f"{model_name}.json"
        
        if not prediction_file.exists():
            raise FileNotFoundError(f"Prediction file not found: {prediction_file}")
        
        self.logger.info(f"Loading predictions for {model_name} from: {prediction_file}")
        
        with open(prediction_file, 'r', encoding='utf-8') as f:
            raw_predictions = json.load(f)
        
        # Convert from {video_id: summary_text} format to list of dictionaries
        predictions = []
        for video_id, summary_text in raw_predictions.items():
            predictions.append({
                "video_id": video_id,
                "summary": summary_text
            })
        
        self.logger.info(f"Loaded {len(predictions)} predictions for {model_name}")
        return predictions
    
    def evaluate_model_predictions(self, model_name: str, predictions: List[Dict], 
                                 ground_truth: List[Dict]) -> Dict[str, Any]:
        """Evaluate a single model's predictions using n-gram metrics."""
        
        self.logger.info(f"Starting n-gram evaluation for model: {model_name}")
        
        # Align predictions with ground truth
        aligned_data = self._align_predictions_and_ground_truth(predictions, ground_truth)
        
        if not aligned_data:
            self.logger.error(f"No aligned data found for {model_name}")
            return {}
        
        # Evaluate each video
        results = []
        successful_evaluations = 0
        criterion_totals = {metric: 0.0 for metric in self.target_metrics}
        progress_interval = self.config['ngram_metrics'].get('progress_interval', 10)
        
        for i, (pred, gt) in enumerate(aligned_data, 1):
            # Extract video ID and summaries
            video_id = pred.get('video_id') or gt.get('id') or str(i)
            reference_summary = gt.get('summary', '')
            generated_summary = pred.get('summary', '')
            
            if not reference_summary or not generated_summary:
                self.logger.warning(f"Missing summary for {video_id}, skipping")
                continue
            
            # Evaluate using traditional metrics
            try:
                metrics = evaluate_traditional_metrics(reference_summary, generated_summary)
                
                # Filter to only requested metrics
                filtered_metrics = {k: v for k, v in metrics.items() if k in self.target_metrics}
                
                result = NgramResult(
                    video_id=video_id,
                    reference_summary=reference_summary,
                    generated_summary=generated_summary,
                    metrics=filtered_metrics,
                    success=True
                )
                
                # Accumulate totals for averaging
                for metric in self.target_metrics:
                    if metric in filtered_metrics:
                        criterion_totals[metric] += filtered_metrics[metric]
                
                successful_evaluations += 1
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {video_id}: {e}")
                result = NgramResult(
                    video_id=video_id,
                    reference_summary=reference_summary,
                    generated_summary=generated_summary,
                    metrics={metric: 0.0 for metric in self.target_metrics},
                    success=False,
                    error_message=str(e)
                )
            
            results.append(result)
            
            # Progress reporting
            if i % progress_interval == 0:
                self.logger.info(f"Processed {i}/{len(aligned_data)} videos for {model_name}")
            
            # Memory cleanup
            clear_interval = self.config['processing'].get('clear_cache_interval', 50)
            if i % clear_interval == 0:
                gc.collect()
        
        # Calculate averages
        num_samples = len(results)
        success_rate = successful_evaluations / num_samples if num_samples > 0 else 0
        criterion_averages = {
            metric: total / successful_evaluations if successful_evaluations > 0 else 0
            for metric, total in criterion_totals.items()
        }
        
        # Create summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = {
            "model_name": model_name,
            "total_samples": num_samples,
            "successful_evaluations": successful_evaluations,
            "success_rate": success_rate,
            "metric_averages": criterion_averages,
            "timestamp": timestamp,
            "evaluation_method": "ngram_traditional_metrics",
            "metrics_computed": self.target_metrics
        }
        
        # Save results
        self._save_results(model_name, results, summary, timestamp)
        
        self.logger.info(f"N-gram evaluation completed for {model_name}")
        self.logger.info(f"Success rate: {success_rate:.2%}")
        
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
    
    def _save_results(self, model_name: str, results: List[NgramResult], 
                     summary: Dict[str, Any], timestamp: str):
        """Save individual and aggregated results in both JSON and CSV formats."""
        
        # Save individual results JSON
        individual_json_file = self.individual_json_dir / f"{model_name}_ngram_individual_{timestamp}.json"
        individual_data = [self._result_to_dict(result) for result in results]
        
        with open(individual_json_file, 'w') as f:
            json.dump(individual_data, f, indent=2)
        
        # Save individual results CSV
        individual_csv_file = self.individual_csv_dir / f"{model_name}_ngram_individual_{timestamp}.csv"
        self._save_individual_csv(individual_csv_file, results)
        
        # Save aggregated summary JSON
        summary_file = self.aggregated_results_dir / f"{model_name}_ngram_summary_{timestamp}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Update aggregated CSV
        self._update_aggregated_csv(model_name, summary)
        
        self.logger.info(f"Individual JSON results saved to: {individual_json_file}")
        self.logger.info(f"Individual CSV results saved to: {individual_csv_file}")
        self.logger.info(f"Summary saved to: {summary_file}")
        self.logger.info(f"Aggregated CSV updated")
    
    def _result_to_dict(self, result: NgramResult) -> Dict[str, Any]:
        """Convert NgramResult to dictionary for JSON serialization."""
        return {
            "video_id": result.video_id,
            "reference_summary": result.reference_summary,
            "generated_summary": result.generated_summary,
            "metrics": result.metrics,
            "success": result.success,
            "error_message": result.error_message
        }
    
    def _save_individual_csv(self, csv_file: Path, results: List[NgramResult]):
        """Save individual results to CSV format."""
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            header = ['video_id'] + self.target_metrics
            writer.writerow(header)
            
            # Write data rows
            for result in results:
                if result.success:
                    row = [result.video_id]
                    for metric in self.target_metrics:
                        score = result.metrics.get(metric, 0.0)
                        row.append(round(score, self.decimal_precision))
                else:
                    # For failed evaluations, use 0.0 scores
                    row = [result.video_id] + [0.0] * len(self.target_metrics)
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
        metric_averages = summary['metric_averages']
        new_row = {'Model Name': model_name}
        
        for metric in self.target_metrics:
            avg_score = metric_averages.get(metric, 0.0)
            new_row[metric] = round(avg_score, self.decimal_precision)
        
        existing_data.append(new_row)
        
        # Sort by first metric (typically BLEU-1) descending
        if self.target_metrics:
            sort_metric = self.target_metrics[0]
            existing_data.sort(key=lambda x: float(x.get(sort_metric, 0)), reverse=True)
        
        # Write updated data
        with open(aggr_csv_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Model Name'] + self.target_metrics
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_data)
    
    def evaluate_all_models(self) -> List[Dict[str, Any]]:
        """Evaluate all models specified in configuration."""
        ground_truth = self.load_ground_truth()
        models_to_evaluate = self.config['models_to_evaluate']
        
        self.logger.info(f"Starting n-gram evaluation for {len(models_to_evaluate)} models")
        
        all_summaries = []
        for i, model_name in enumerate(models_to_evaluate, 1):
            self.logger.info(f"Evaluating model {i}/{len(models_to_evaluate)}: {model_name}")
            
            try:
                predictions = self.load_model_predictions(model_name)
                summary = self.evaluate_model_predictions(model_name, predictions, ground_truth)
                all_summaries.append(summary)
                
                # Log completion for this model
                if 'metric_averages' in summary:
                    avg_scores = summary['metric_averages']
                    score_str = ", ".join([f"{k}={v:.3f}" for k, v in avg_scores.items()])
                    self.logger.info(f"Completed {model_name}: {score_str}")
                
                # Force garbage collection after each model
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {model_name}: {e}")
                continue
        
        # Save overall summary
        self._save_overall_summary(all_summaries)
        
        return all_summaries
    
    def _save_overall_summary(self, all_summaries: List[Dict[str, Any]]):
        """Save overall evaluation summary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        overall_summary = {
            "evaluation_type": "ngram_traditional_metrics",
            "timestamp": timestamp,
            "total_models_evaluated": len(all_summaries),
            "metrics_computed": self.target_metrics,
            "model_summaries": all_summaries
        }
        
        summary_file = self.aggregated_results_dir / f"ngram_overall_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(overall_summary, f, indent=2)
        
        self.logger.info(f"Overall summary saved to: {summary_file}")