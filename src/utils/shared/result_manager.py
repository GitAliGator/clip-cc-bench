"""
Shared Result Management System

Thread-safe result management for concurrent encoder evaluations.
Handles CSV and JSON file updates with proper locking.
"""

import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from base_types import EncoderEvaluationResult, SimilarityScore


class SharedResultManager:
    """Thread-safe result manager for concurrent encoder evaluations."""

    def __init__(self, results_base_dir: Path, encoder_name: str):
        self.results_base_dir = Path(results_base_dir)
        self.encoder_name = encoder_name
        self.logger = logging.getLogger(f'result_manager.{encoder_name}')

        # Result directories - encoder-isolated structure
        self.individual_csv_dir = self.results_base_dir / "encoders" / "individual_results" / "csv" / self.encoder_name
        self.individual_json_dir = self.results_base_dir / "encoders" / "individual_results" / "json" / self.encoder_name
        self.aggregated_results_dir = self.results_base_dir / "encoders" / "aggregated_results"
        self.encoder_aggregated_dir = self.aggregated_results_dir / self.encoder_name
        self.logs_dir = self.results_base_dir / "encoders" / "logs" / self.encoder_name

        # Ensure directories exist
        for dir_path in [self.individual_csv_dir, self.individual_json_dir, self.aggregated_results_dir, self.encoder_aggregated_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def save_individual_result(self, result: EncoderEvaluationResult) -> bool:
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

    def _save_individual_result_csv(self, result: EncoderEvaluationResult):
        """Save individual result to CSV file in encoder-specific directory."""
        model_csv_file = self.individual_csv_dir / f"{result.model_name}.csv"
        score = result.encoder_similarities.get(self.encoder_name, SimilarityScore(0.5, 0.5))

        # Read existing data if file exists
        data = []
        if model_csv_file.exists():
            with open(model_csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                data = list(reader)

        # Initialize with header if empty
        if not data:
            data = [['video_id', 'similarity_score']]

        # Find existing row or add new one
        video_row_idx = None
        for i in range(1, len(data)):
            if len(data[i]) > 0 and data[i][0] == result.video_id:
                video_row_idx = i
                break

        if video_row_idx is not None:
            # Update existing row
            data[video_row_idx][1] = round(score.normalized_cosine, 3)
        else:
            # Add new row
            data.append([result.video_id, round(score.normalized_cosine, 3)])

        # Write updated data
        with open(model_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(data)

    def _save_individual_result_json(self, result: EncoderEvaluationResult):
        """Save individual result to JSON file in encoder-specific directory."""
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

        # Update with encoder's results
        if self.encoder_name in result.encoder_similarities:
            sim_score = result.encoder_similarities[self.encoder_name]

            # Check if this is a fallback score and mark as failure
            is_fallback = sim_score.metadata.get('fallback_score', False)
            final_success = result.success and not is_fallback

            # Add fallback information to error message if applicable
            error_msg = result.error_message or ""
            if is_fallback and not error_msg:
                error_msg = "Fallback score used due to evaluation failure"

            data[result.video_id]['similarity_data'] = {
                'cosine_similarity': sim_score.cosine_similarity,
                'normalized_cosine': sim_score.normalized_cosine,
                'metadata': {
                    'success': final_success,
                    'error_message': error_msg,
                    'computation_time': sim_score.metadata.get('computation_time', 0.0),
                    'is_fallback_score': is_fallback
                }
            }

        # Write updated content
        with open(model_json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def create_encoder_summary(self, all_results: List[EncoderEvaluationResult]) -> Dict[str, Any]:
        """Create aggregated summary for this encoder."""
        if not all_results:
            return {}

        # Calculate aggregate statistics
        similarities = []
        successful_evaluations = 0

        for result in all_results:
            if result.success and self.encoder_name in result.encoder_similarities:
                sim_score = result.encoder_similarities[self.encoder_name]
                # Check if this is a fallback score - if so, don't count as successful
                is_fallback = sim_score.metadata.get('fallback_score', False)
                if not is_fallback:
                    similarities.append(sim_score.normalized_cosine)
                    successful_evaluations += 1

        if not similarities:
            return {
                'encoder_name': self.encoder_name,
                'total_evaluations': len(all_results),
                'successful_evaluations': 0,
                'error_rate': 1.0,
                'timestamp': datetime.now().isoformat()
            }

        # Calculate statistics
        avg_similarity = sum(similarities) / len(similarities)
        min_similarity = min(similarities)
        max_similarity = max(similarities)

        summary = {
            'encoder_name': self.encoder_name,
            'total_evaluations': len(all_results),
            'successful_evaluations': successful_evaluations,
            'error_rate': (len(all_results) - successful_evaluations) / len(all_results),
            'similarity_stats': {
                'mean': avg_similarity,
                'min': min_similarity,
                'max': max_similarity,
                'std': (sum((s - avg_similarity) ** 2 for s in similarities) / len(similarities)) ** 0.5
            },
            'timestamp': datetime.now().isoformat()
        }

        return summary

    def get_all_model_results(self) -> List[str]:
        """Get list of all model names that have result files."""
        model_names = set()

        # Check CSV files in encoder-specific directory
        for csv_file in self.individual_csv_dir.glob("*.csv"):
            model_names.add(csv_file.stem)

        # Check JSON files in encoder-specific directory
        for json_file in self.individual_json_dir.glob("*.json"):
            model_names.add(json_file.stem)

        return sorted(list(model_names))

    def load_model_results(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load existing results for a model from encoder-specific directory."""
        json_file = self.individual_json_dir / f"{model_name}.json"

        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load results for {model_name}: {e}")

        return None

    def create_per_model_summary(self, model_results: List[EncoderEvaluationResult], model_name: str) -> Dict[str, Any]:
        """Create summary statistics for a specific model."""
        if not model_results:
            return {}

        # Calculate per-model statistics
        similarities = []
        successful_evaluations = 0

        for result in model_results:
            if result.success and self.encoder_name in result.encoder_similarities:
                sim_score = result.encoder_similarities[self.encoder_name]
                # Check if this is a fallback score - if so, don't count as successful
                is_fallback = sim_score.metadata.get('fallback_score', False)
                if not is_fallback:
                    similarities.append(sim_score.normalized_cosine)
                    successful_evaluations += 1

        if not similarities:
            return {
                'encoder_name': self.encoder_name,
                'model_name': model_name,
                'total_evaluations': len(model_results),
                'successful_evaluations': 0,
                'error_rate': 1.0,
                'timestamp': datetime.now().isoformat()
            }

        # Calculate statistics
        avg_similarity = sum(similarities) / len(similarities)
        min_similarity = min(similarities)
        max_similarity = max(similarities)

        summary = {
            'encoder_name': self.encoder_name,
            'model_name': model_name,
            'total_evaluations': len(model_results),
            'successful_evaluations': successful_evaluations,
            'error_rate': (len(model_results) - successful_evaluations) / len(model_results),
            'similarity_stats': {
                'mean': round(avg_similarity, 6),
                'min': round(min_similarity, 6),
                'max': round(max_similarity, 6),
                'std': round((sum((s - avg_similarity) ** 2 for s in similarities) / len(similarities)) ** 0.5, 6)
            },
            'timestamp': datetime.now().isoformat()
        }

        return summary

    def save_per_model_summary(self, model_results: List[EncoderEvaluationResult], model_name: str):
        """Save per-model summary to encoder-specific aggregated results."""
        summary = self.create_per_model_summary(model_results, model_name)

        if not summary:
            return

        # Save to encoder-specific aggregated directory
        model_summary_file = self.encoder_aggregated_dir / f"{model_name}.json"
        with open(model_summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Update aggregated CSV
        self.update_aggregated_csv(model_name, summary['similarity_stats']['mean'])

        # Update cross-encoder stats
        self.update_cross_encoder_stats(model_name, summary['similarity_stats'])

    def update_aggregated_csv(self, model_name: str, mean_score: float):
        """Update the aggregated results CSV file by reading from all encoder directories."""
        csv_file = self.aggregated_results_dir / "aggregated_results.csv"

        # Discover all encoder directories
        encoders_base_dir = self.results_base_dir / "encoders" / "aggregated_results"
        encoder_dirs = [d for d in encoders_base_dir.iterdir() if d.is_dir() and d.name != '__pycache__']

        # Collect data from all encoder directories
        data = {}
        encoders = set()

        for encoder_dir in encoder_dirs:
            encoder_name = encoder_dir.name
            encoders.add(encoder_name)

            # Read all model summaries for this encoder
            for model_file in encoder_dir.glob("*.json"):
                model = model_file.stem
                try:
                    with open(model_file, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                        if model not in data:
                            data[model] = {}
                        data[model][encoder_name] = f"{summary['similarity_stats']['mean']:.3f}"
                except Exception as e:
                    self.logger.warning(f"Could not read {model_file}: {e}")

        # Write updated CSV
        fieldnames = ['Model Name'] + sorted(encoders)

        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for model in sorted(data.keys()):
                row = {'Model Name': model}
                for encoder in sorted(encoders):
                    row[encoder] = data[model].get(encoder, '')
                writer.writerow(row)

    def update_cross_encoder_stats(self, model_name: str, similarity_stats: Dict[str, float]):
        """Update cross-encoder comparison statistics."""
        cross_stats_file = self.aggregated_results_dir / "cross_encoder_stats.json"

        # Load existing cross-encoder stats
        cross_stats = {}
        if cross_stats_file.exists():
            try:
                with open(cross_stats_file, 'r', encoding='utf-8') as f:
                    cross_stats = json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load cross-encoder stats: {e}")

        # Initialize model entry if it doesn't exist
        if model_name not in cross_stats:
            cross_stats[model_name] = {
                'encoder_comparisons': {},
                'timestamp': datetime.now().isoformat()
            }

        # Update this encoder's stats for the model
        cross_stats[model_name]['encoder_comparisons'][self.encoder_name] = {
            'mean': round(similarity_stats['mean'], 6),
            'std': round(similarity_stats['std'], 6),
            'min': round(similarity_stats['min'], 6),
            'max': round(similarity_stats['max'], 6)
        }

        # Calculate rankings and best encoder for this model
        encoder_means = {enc: stats['mean'] for enc, stats in cross_stats[model_name]['encoder_comparisons'].items()}
        if encoder_means:
            sorted_encoders = sorted(encoder_means.items(), key=lambda x: x[1], reverse=True)
            cross_stats[model_name]['best_encoder'] = sorted_encoders[0][0]
            cross_stats[model_name]['ranking'] = [enc for enc, _ in sorted_encoders]

            # Calculate performance gap (difference between best and worst)
            if len(sorted_encoders) > 1:
                cross_stats[model_name]['performance_gap'] = round(sorted_encoders[0][1] - sorted_encoders[-1][1], 6)

        # Update overall summary statistics
        cross_stats['summary'] = self._calculate_summary_stats(cross_stats)

        # Save updated cross-encoder stats
        with open(cross_stats_file, 'w', encoding='utf-8') as f:
            json.dump(cross_stats, f, indent=2, ensure_ascii=False)

    def _calculate_summary_stats(self, cross_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics across all models and encoders."""
        model_entries = {k: v for k, v in cross_stats.items() if k != 'summary' and isinstance(v, dict) and 'encoder_comparisons' in v}

        if not model_entries:
            return {'total_models': 0, 'timestamp': datetime.now().isoformat()}

        # Collect all encoder performance data
        encoder_performances = {}
        for model_name, model_data in model_entries.items():
            for encoder_name, encoder_stats in model_data.get('encoder_comparisons', {}).items():
                if encoder_name not in encoder_performances:
                    encoder_performances[encoder_name] = []
                encoder_performances[encoder_name].append(encoder_stats['mean'])

        # Calculate overall best encoder
        overall_best_encoder = None
        if encoder_performances:
            encoder_averages = {enc: sum(scores) / len(scores) for enc, scores in encoder_performances.items()}
            overall_best_encoder = max(encoder_averages.items(), key=lambda x: x[1])[0]

        # Calculate average performance gaps
        performance_gaps = []
        for model_data in model_entries.values():
            if 'performance_gap' in model_data:
                performance_gaps.append(model_data['performance_gap'])

        avg_performance_gap = sum(performance_gaps) / len(performance_gaps) if performance_gaps else 0.0

        return {
            'total_models': len(model_entries),
            'total_encoders': len(encoder_performances),
            'overall_best_encoder': overall_best_encoder,
            'avg_performance_gap': round(avg_performance_gap, 6),
            'encoder_averages': {enc: round(sum(scores) / len(scores), 6) for enc, scores in encoder_performances.items()},
            'timestamp': datetime.now().isoformat()
        }