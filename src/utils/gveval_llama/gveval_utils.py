"""
G-VEval LLaMA Utilities

Utility functions for G-VEval LLaMA evaluation system.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

def load_json_file(file_path: str) -> Any:
    """Load JSON file with error handling."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(path, 'r') as f:
        return json.load(f)

def save_json_file(data: Any, file_path: str, indent: int = 2) -> None:
    """Save data to JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)

def extract_scores_from_text(text: str, criteria: List[str]) -> Dict[str, float]:
    """Extract scores from text using various patterns."""
    scores = {}
    
    # Greek letter patterns (G-VEval style)
    greek_patterns = {
        'accuracy': r'α(\d+(?:\.\d+)?)α',
        'completeness': r'β(\d+(?:\.\d+)?)β',
        'conciseness': r'ψ(\d+(?:\.\d+)?)ψ',
        'relevance': r'δ(\d+(?:\.\d+)?)δ'
    }
    
    # Standard patterns
    standard_patterns = {
        criterion: rf'{criterion}[^\d]*(\d+(?:\.\d+)?)'
        for criterion in criteria
    }
    
    for criterion in criteria:
        score = None
        
        # Try Greek letter pattern first
        if criterion in greek_patterns:
            match = re.search(greek_patterns[criterion], text)
            if match:
                score = float(match.group(1))
        
        # Try standard pattern
        if score is None:
            match = re.search(standard_patterns[criterion], text, re.IGNORECASE)
            if match:
                score = float(match.group(1))
        
        # Default score if nothing found
        scores[criterion] = score if score is not None else 50.0
    
    return scores

def validate_score_range(score: float, min_score: float = 0, max_score: float = 100) -> float:
    """Validate and clamp score to valid range."""
    return max(min_score, min(max_score, score))

def calculate_overall_score(criterion_scores: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    """Calculate overall score from criterion scores."""
    if not criterion_scores:
        return 0.0
    
    if weights is None:
        # Equal weights
        return sum(criterion_scores.values()) / len(criterion_scores)
    else:
        # Weighted average
        weighted_sum = sum(score * weights.get(criterion, 1.0) 
                          for criterion, score in criterion_scores.items())
        total_weight = sum(weights.get(criterion, 1.0) 
                          for criterion in criterion_scores.keys())
        return weighted_sum / total_weight if total_weight > 0 else 0.0

def format_evaluation_summary(results: List[Dict[str, Any]]) -> str:
    """Format evaluation results into a readable summary."""
    if not results:
        return "No results to summarize."
    
    total_samples = len(results)
    successful_results = [r for r in results if r.get('success', False)]
    success_rate = len(successful_results) / total_samples
    
    if not successful_results:
        return f"Evaluated {total_samples} samples, but no successful evaluations."
    
    # Calculate averages
    avg_overall = sum(r['overall_score'] for r in successful_results) / len(successful_results)
    
    # Calculate criterion averages
    criteria = list(successful_results[0]['accr_scores'].keys()) if successful_results[0].get('accr_scores') else []
    criterion_avgs = {}
    for criterion in criteria:
        scores = [r['accr_scores'][criterion] for r in successful_results if criterion in r.get('accr_scores', {})]
        criterion_avgs[criterion] = sum(scores) / len(scores) if scores else 0.0
    
    # Format summary
    summary = f"""G-VEval LLaMA Evaluation Summary
================================
Total Samples: {total_samples}
Successful Evaluations: {len(successful_results)}
Success Rate: {success_rate:.1%}
Average Overall Score: {avg_overall:.2f}/100

ACCR Criterion Averages:
"""
    
    for criterion, avg_score in criterion_avgs.items():
        summary += f"  {criterion.capitalize():12s}: {avg_score:6.2f}/100\n"
    
    return summary

def compare_models(model_summaries: List[Dict[str, Any]]) -> str:
    """Compare multiple model evaluation summaries."""
    if not model_summaries:
        return "No model summaries to compare."
    
    # Sort by average score
    sorted_summaries = sorted(model_summaries, 
                            key=lambda x: x.get('average_gveval_score', 0), 
                            reverse=True)
    
    comparison = "Model Comparison by G-VEval Score\n"
    comparison += "=" * 40 + "\n"
    
    for i, summary in enumerate(sorted_summaries, 1):
        model_name = summary.get('model_name', 'Unknown')
        avg_score = summary.get('average_gveval_score', 0)
        success_rate = summary.get('success_rate', 0)
        
        comparison += f"{i:2d}. {model_name:25s}: {avg_score:6.2f}/100 (success: {success_rate:.1%})\n"
    
    return comparison

def check_file_exists(file_path: str, description: str = "File") -> bool:
    """Check if file exists and log result."""
    path = Path(file_path)
    exists = path.exists()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {file_path}")
    return exists

def setup_directories(*dir_paths: str) -> None:
    """Create directories if they don't exist."""
    for dir_path in dir_paths:
        Path(dir_path).mkdir(parents=True, exist_ok=True)