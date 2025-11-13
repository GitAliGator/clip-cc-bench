"""
Shared Result Management System

Thread-safe result management for concurrent decoder evaluations.
Handles CSV and JSON file updates with proper locking.
"""

import csv
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from base_types import DecoderEvaluationResult, SimilarityScore


class SharedResultManager:
    """Thread-safe result manager for concurrent decoder evaluations."""

    def __init__(self, results_base_dir: Path, decoder_name: str):
        self.results_base_dir = Path(results_base_dir)
        self.decoder_name = decoder_name
        self.logger = logging.getLogger(f'result_manager.{decoder_name}')

        # Result directories - decoder-isolated structure
        self.individual_csv_dir = self.results_base_dir / "decoders" / "individual_results" / "csv" / self.decoder_name
        self.individual_json_dir = self.results_base_dir / "decoders" / "individual_results" / "json" / self.decoder_name
        self.aggregated_results_dir = self.results_base_dir / "decoders" / "aggregated_results"
        self.decoder_aggregated_dir = self.aggregated_results_dir / self.decoder_name
        self.logs_dir = self.results_base_dir / "decoders" / "logs" / self.decoder_name

        # Ensure directories exist
        for dir_path in [self.individual_csv_dir, self.individual_json_dir, self.aggregated_results_dir, self.decoder_aggregated_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def save_individual_result(self, result: DecoderEvaluationResult) -> bool:
        """Save individual evaluation result to both CSV and JSON."""
        try:
            # Save to CSV (thread-safe)
            self._save_individual_result_csv(result)

            # Save to JSON (thread-safe)
            self._save_individual_result_json(result)

            return True
        except Exception as e:
            self.logger.error(f"Failed to save individual result for {result.video_id}: {e}")
            return False

    def _save_individual_result_csv(self, result: DecoderEvaluationResult):
        """Save individual result to CSV file in decoder-specific directory with fine-grained metrics."""
        model_csv_file = self.individual_csv_dir / f"{result.model_name}.csv"
        score = result.decoder_similarities.get(self.decoder_name, SimilarityScore(0.5, 0.5))

        # Read existing data if file exists
        data = []
        if model_csv_file.exists():
            with open(model_csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                data = list(reader)

        # Initialize with header if empty
        if not data:
            data = [[
                'video_id',
                'coarse_similarity',
                'fine_precision',
                'fine_recall',
                'fine_f1',
                'hm_cf'
            ]]

        # Find existing row or add new one
        video_row_idx = None
        for i in range(1, len(data)):
            if len(data[i]) > 0 and data[i][0] == result.video_id:
                video_row_idx = i
                break

        row_data = [
            result.video_id,
            round(score.normalized_cosine, 4),
            round(score.fine_grained_precision or 0.0, 4),
            round(score.fine_grained_recall or 0.0, 4),
            round(score.fine_grained_f1 or 0.0, 4),
            round(score.hm_cf or 0.0, 4)
        ]

        if video_row_idx is not None:
            # Update existing row
            data[video_row_idx] = row_data
        else:
            # Add new row
            data.append(row_data)

        # Write updated data
        with open(model_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(data)

    def _save_individual_result_json(self, result: DecoderEvaluationResult):
        """Save individual result to JSON file in decoder-specific directory with fine-grained metrics."""
        model_json_file = self.individual_json_dir / f"{result.model_name}.json"

        # Load existing data if file exists
        data = {}
        if model_json_file.exists():
            try:
                with open(model_json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                self.logger.warning(f"JSON file corrupted for {result.model_name}, starting fresh: {e}")
                data = {}

        # Ensure video_id entry exists
        if result.video_id not in data:
            data[result.video_id] = {
                'reference': result.ground_truth_text,
                'candidate': result.prediction_text,
                'reference_length': len(result.ground_truth_text),
                'candidate_length': len(result.prediction_text)
            }

        # Update with decoder's results
        if self.decoder_name in result.decoder_similarities:
            sim_score = result.decoder_similarities[self.decoder_name]

            # Check if this is a fallback score and mark as failure
            is_fallback = sim_score.metadata.get('fallback_score', False) if sim_score.metadata else False
            final_success = result.success and not is_fallback

            # Add fallback information to error message if applicable
            error_msg = result.error_message or ""
            if is_fallback and not error_msg:
                error_msg = "Fallback score used due to evaluation failure"

            data[result.video_id]['similarity_data'] = {
                'coarse_grained': {
                    'cosine_similarity': sim_score.cosine_similarity,
                    'normalized_cosine': sim_score.normalized_cosine
                },
                'fine_grained': {
                    'precision': sim_score.fine_grained_precision,
                    'recall': sim_score.fine_grained_recall,
                    'f1_score': sim_score.fine_grained_f1,
                    'num_gt_chunks': sim_score.metadata.get('num_gt_chunks', 0) if sim_score.metadata else 0,
                    'num_pred_chunks': sim_score.metadata.get('num_pred_chunks', 0) if sim_score.metadata else 0
                },
                'hybrid': {
                    'hm_cf': sim_score.hm_cf
                },
                'metadata': {
                    'success': final_success,
                    'error_message': error_msg,
                    'computation_time': sim_score.metadata.get('computation_time', 0.0) if sim_score.metadata else 0.0,
                    'is_fallback_score': is_fallback
                }
            }

        # Write updated content
        with open(model_json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def create_decoder_summary(self, all_results: List[DecoderEvaluationResult]) -> Dict[str, Any]:
        """Create aggregated summary for this decoder."""
        if not all_results:
            return {}

        # Calculate aggregate statistics
        similarities = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        hm_cf_scores = []
        successful_evaluations = 0

        for result in all_results:
            if result.success and self.decoder_name in result.decoder_similarities:
                sim_score = result.decoder_similarities[self.decoder_name]
                # Check if this is a fallback score - if so, don't count as successful
                is_fallback = sim_score.metadata.get('fallback_score', False) if sim_score.metadata else False
                if not is_fallback:
                    similarities.append(sim_score.normalized_cosine)
                    if sim_score.fine_grained_precision is not None:
                        precision_scores.append(sim_score.fine_grained_precision)
                    if sim_score.fine_grained_recall is not None:
                        recall_scores.append(sim_score.fine_grained_recall)
                    if sim_score.fine_grained_f1 is not None:
                        f1_scores.append(sim_score.fine_grained_f1)
                    if sim_score.hm_cf is not None:
                        hm_cf_scores.append(sim_score.hm_cf)
                    successful_evaluations += 1

        if not similarities:
            return {
                'decoder_name': self.decoder_name,
                'total_evaluations': len(all_results),
                'successful_evaluations': 0,
                'error_rate': 1.0,
                'timestamp': datetime.now().isoformat()
            }

        # Calculate statistics
        summary = {
            'decoder_name': self.decoder_name,
            'total_evaluations': len(all_results),
            'successful_evaluations': successful_evaluations,
            'error_rate': (len(all_results) - successful_evaluations) / len(all_results),
            'coarse_grained_stats': {
                'mean': np.mean(similarities),
                'min': np.min(similarities),
                'max': np.max(similarities),
                'std': np.std(similarities)
            },
            'timestamp': datetime.now().isoformat()
        }

        # Add fine-grained stats if available
        if precision_scores:
            summary['fine_grained_stats'] = {
                'precision': {
                    'mean': np.mean(precision_scores),
                    'std': np.std(precision_scores),
                    'min': np.min(precision_scores),
                    'max': np.max(precision_scores)
                },
                'recall': {
                    'mean': np.mean(recall_scores),
                    'std': np.std(recall_scores),
                    'min': np.min(recall_scores),
                    'max': np.max(recall_scores)
                },
                'f1': {
                    'mean': np.mean(f1_scores),
                    'std': np.std(f1_scores),
                    'min': np.min(f1_scores),
                    'max': np.max(f1_scores)
                }
            }

        # Add hybrid stats if available
        if hm_cf_scores:
            summary['hybrid_stats'] = {
                'hm_cf': {
                    'mean': np.mean(hm_cf_scores),
                    'std': np.std(hm_cf_scores),
                    'min': np.min(hm_cf_scores),
                    'max': np.max(hm_cf_scores)
                }
            }

        return summary

    def get_all_model_results(self) -> List[str]:
        """Get list of all model names that have result files."""
        model_names = set()

        # Check CSV files in decoder-specific directory
        for csv_file in self.individual_csv_dir.glob("*.csv"):
            model_names.add(csv_file.stem)

        # Check JSON files in decoder-specific directory
        for json_file in self.individual_json_dir.glob("*.json"):
            model_names.add(json_file.stem)

        return sorted(list(model_names))

    def load_model_results(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load existing results for a model from decoder-specific directory."""
        json_file = self.individual_json_dir / f"{model_name}.json"

        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load results for {model_name}: {e}")

        return None

    def create_per_model_summary(self, model_results: List[DecoderEvaluationResult], model_name: str) -> Dict[str, Any]:
        """Create summary statistics for a specific model with fine-grained metrics."""
        if not model_results:
            return {}

        # Calculate per-model statistics
        similarities = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        hm_cf_scores = []
        successful_evaluations = 0

        for result in model_results:
            if result.success and self.decoder_name in result.decoder_similarities:
                sim_score = result.decoder_similarities[self.decoder_name]
                # Check if this is a fallback score - if so, don't count as successful
                is_fallback = sim_score.metadata.get('fallback_score', False) if sim_score.metadata else False
                if not is_fallback:
                    similarities.append(sim_score.normalized_cosine)
                    if sim_score.fine_grained_precision is not None:
                        precision_scores.append(sim_score.fine_grained_precision)
                    if sim_score.fine_grained_recall is not None:
                        recall_scores.append(sim_score.fine_grained_recall)
                    if sim_score.fine_grained_f1 is not None:
                        f1_scores.append(sim_score.fine_grained_f1)
                    if sim_score.hm_cf is not None:
                        hm_cf_scores.append(sim_score.hm_cf)
                    successful_evaluations += 1

        if not similarities:
            return {
                'decoder_name': self.decoder_name,
                'model_name': model_name,
                'total_evaluations': len(model_results),
                'successful_evaluations': 0,
                'error_rate': 1.0,
                'timestamp': datetime.now().isoformat()
            }

        # Calculate statistics
        summary = {
            'decoder_name': self.decoder_name,
            'model_name': model_name,
            'total_evaluations': len(model_results),
            'successful_evaluations': successful_evaluations,
            'error_rate': (len(model_results) - successful_evaluations) / len(model_results),
            'coarse_grained_stats': {
                'mean': round(np.mean(similarities), 6),
                'min': round(np.min(similarities), 6),
                'max': round(np.max(similarities), 6),
                'std': round(np.std(similarities), 6)
            },
            'timestamp': datetime.now().isoformat()
        }

        # Add fine-grained stats if available
        if precision_scores:
            summary['fine_grained_stats'] = {
                'precision': {
                    'mean': round(np.mean(precision_scores), 6),
                    'std': round(np.std(precision_scores), 6),
                    'min': round(np.min(precision_scores), 6),
                    'max': round(np.max(precision_scores), 6)
                },
                'recall': {
                    'mean': round(np.mean(recall_scores), 6),
                    'std': round(np.std(recall_scores), 6),
                    'min': round(np.min(recall_scores), 6),
                    'max': round(np.max(recall_scores), 6)
                },
                'f1': {
                    'mean': round(np.mean(f1_scores), 6),
                    'std': round(np.std(f1_scores), 6),
                    'min': round(np.min(f1_scores), 6),
                    'max': round(np.max(f1_scores), 6)
                }
            }

        # Add hybrid stats if available
        if hm_cf_scores:
            summary['hybrid_stats'] = {
                'hm_cf': {
                    'mean': round(np.mean(hm_cf_scores), 6),
                    'std': round(np.std(hm_cf_scores), 6),
                    'min': round(np.min(hm_cf_scores), 6),
                    'max': round(np.max(hm_cf_scores), 6)
                }
            }

        return summary

    def save_per_model_summary(self, model_results: List[DecoderEvaluationResult], model_name: str):
        """Save per-model summary to decoder-specific aggregated results."""
        summary = self.create_per_model_summary(model_results, model_name)

        if not summary:
            return

        # Save to decoder-specific aggregated directory
        model_summary_file = self.decoder_aggregated_dir / f"{model_name}.json"
        with open(model_summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Update aggregated CSV
        self.update_aggregated_csv(model_name, summary)

        # Update cross-decoder stats
        self.update_cross_decoder_stats(model_name, summary)

    def update_aggregated_csv(self, model_name: str, summary: Dict[str, Any]):
        """Update the aggregated results CSV file by reading from all decoder directories."""
        csv_file = self.aggregated_results_dir / "aggregated_results.csv"

        # Discover all decoder directories
        decoders_base_dir = self.results_base_dir / "decoders" / "aggregated_results"
        decoder_dirs = [d for d in decoders_base_dir.iterdir() if d.is_dir() and d.name != '__pycache__']

        # Collect data from all decoder directories
        data = {}
        decoders = set()

        for decoder_dir in decoder_dirs:
            decoder_name = decoder_dir.name
            decoders.add(decoder_name)

            # Read all model summaries for this decoder
            for model_file in decoder_dir.glob("*.json"):
                model = model_file.stem
                try:
                    with open(model_file, 'r', encoding='utf-8') as f:
                        model_summary = json.load(f)
                        if model not in data:
                            data[model] = {}
                        # Store both coarse and fine_f1 and hm_cf
                        data[model][f"{decoder_name}_coarse"] = f"{model_summary['coarse_grained_stats']['mean']:.4f}"
                        if 'fine_grained_stats' in model_summary:
                            data[model][f"{decoder_name}_fine_f1"] = f"{model_summary['fine_grained_stats']['f1']['mean']:.4f}"
                        if 'hybrid_stats' in model_summary:
                            data[model][f"{decoder_name}_hm_cf"] = f"{model_summary['hybrid_stats']['hm_cf']['mean']:.4f}"
                except Exception as e:
                    self.logger.warning(f"Could not read {model_file}: {e}")

        # Build column names
        fieldnames = ['Model Name']
        for decoder in sorted(decoders):
            fieldnames.extend([f"{decoder}_coarse", f"{decoder}_fine_f1", f"{decoder}_hm_cf"])

        # Write updated CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for model in sorted(data.keys()):
                row = {'Model Name': model}
                for col in fieldnames[1:]:  # Skip 'Model Name'
                    row[col] = data[model].get(col, '')
                writer.writerow(row)

    def update_cross_decoder_stats(self, model_name: str, summary: Dict[str, Any]):
        """Update cross-decoder comparison statistics."""
        cross_stats_file = self.aggregated_results_dir / "cross_decoder_stats.json"

        # Load existing cross-decoder stats
        cross_stats = {}
        if cross_stats_file.exists():
            try:
                with open(cross_stats_file, 'r', encoding='utf-8') as f:
                    cross_stats = json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load cross-decoder stats: {e}")

        # Initialize model entry if it doesn't exist
        if model_name not in cross_stats:
            cross_stats[model_name] = {
                'decoder_comparisons': {},
                'timestamp': datetime.now().isoformat()
            }

        # Update this decoder's stats for the model
        cross_stats[model_name]['decoder_comparisons'][self.decoder_name] = {
            'coarse_grained': {
                'mean': summary['coarse_grained_stats']['mean'],
                'std': summary['coarse_grained_stats']['std'],
                'min': summary['coarse_grained_stats']['min'],
                'max': summary['coarse_grained_stats']['max']
            }
        }

        # Add fine-grained and hybrid stats if available
        if 'fine_grained_stats' in summary:
            cross_stats[model_name]['decoder_comparisons'][self.decoder_name]['fine_grained'] = {
                'precision': summary['fine_grained_stats']['precision'],
                'recall': summary['fine_grained_stats']['recall'],
                'f1': summary['fine_grained_stats']['f1']
            }

        if 'hybrid_stats' in summary:
            cross_stats[model_name]['decoder_comparisons'][self.decoder_name]['hybrid'] = summary['hybrid_stats']

        # Calculate rankings based on hm_cf (or coarse if hm_cf not available)
        decoder_scores = {}
        for dec, stats in cross_stats[model_name]['decoder_comparisons'].items():
            if 'hybrid' in stats and 'hm_cf' in stats['hybrid']:
                decoder_scores[dec] = stats['hybrid']['hm_cf']['mean']
            else:
                decoder_scores[dec] = stats['coarse_grained']['mean']

        if decoder_scores:
            sorted_decoders = sorted(decoder_scores.items(), key=lambda x: x[1], reverse=True)
            cross_stats[model_name]['best_decoder'] = sorted_decoders[0][0]
            cross_stats[model_name]['ranking'] = [dec for dec, _ in sorted_decoders]

            # Calculate performance gap (difference between best and worst)
            if len(sorted_decoders) > 1:
                cross_stats[model_name]['performance_gap'] = round(sorted_decoders[0][1] - sorted_decoders[-1][1], 6)

        # Update overall summary statistics
        cross_stats['summary'] = self._calculate_summary_stats(cross_stats)

        # Save updated cross-decoder stats
        with open(cross_stats_file, 'w', encoding='utf-8') as f:
            json.dump(cross_stats, f, indent=2, ensure_ascii=False)

    def _calculate_summary_stats(self, cross_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics across all models and decoders."""
        model_entries = {k: v for k, v in cross_stats.items() if k != 'summary' and isinstance(v, dict) and 'decoder_comparisons' in v}

        if not model_entries:
            return {'total_models': 0, 'timestamp': datetime.now().isoformat()}

        # Collect all decoder performance data
        decoder_performances = {}
        for model_name, model_data in model_entries.items():
            for decoder_name, decoder_stats in model_data.get('decoder_comparisons', {}).items():
                if decoder_name not in decoder_performances:
                    decoder_performances[decoder_name] = []
                # Use hm_cf if available, otherwise coarse
                if 'hybrid' in decoder_stats and 'hm_cf' in decoder_stats['hybrid']:
                    decoder_performances[decoder_name].append(decoder_stats['hybrid']['hm_cf']['mean'])
                else:
                    decoder_performances[decoder_name].append(decoder_stats['coarse_grained']['mean'])

        # Calculate overall best decoder
        overall_best_decoder = None
        if decoder_performances:
            decoder_averages = {dec: sum(scores) / len(scores) for dec, scores in decoder_performances.items()}
            overall_best_decoder = max(decoder_averages.items(), key=lambda x: x[1])[0]

        # Calculate average performance gaps
        performance_gaps = []
        for model_data in model_entries.values():
            if 'performance_gap' in model_data:
                performance_gaps.append(model_data['performance_gap'])

        avg_performance_gap = sum(performance_gaps) / len(performance_gaps) if performance_gaps else 0.0

        return {
            'total_models': len(model_entries),
            'total_decoders': len(decoder_performances),
            'overall_best_decoder': overall_best_decoder,
            'avg_performance_gap': round(avg_performance_gap, 6),
            'decoder_averages': {dec: round(sum(scores) / len(scores), 6) for dec, scores in decoder_performances.items()},
            'timestamp': datetime.now().isoformat()
        }
