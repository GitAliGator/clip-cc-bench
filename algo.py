#!/usr/bin/env python3
"""
VLM Ranking Algorithm using Borda Count and Mean Judge Score
Based on multi-embedding evaluation results
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


def load_evaluation_results(base_dir: str = "results/decoders/aggregated_results") -> Dict:
    """
    Load all evaluation results from the aggregated results directory.

    Returns:
        Dictionary with structure: {embedding_model: {vlm_name: stats}}
    """
    base_path = Path(base_dir)
    results = {}

    # Get all embedding model directories
    embedding_dirs = [d for d in base_path.iterdir() if d.is_dir()]

    for embed_dir in embedding_dirs:
        embedding_name = embed_dir.name
        results[embedding_name] = {}

        # Load all VLM JSON files in this embedding directory
        for json_file in embed_dir.glob("*.json"):
            vlm_name = json_file.stem  # filename without .json

            with open(json_file, 'r') as f:
                data = json.load(f)

                # Extract the metrics we need
                results[embedding_name][vlm_name] = {
                    'coarse': data['coarse_grained_stats']['mean'],
                    'fine': data['fine_grained_stats']['f1']['mean'],
                    'hm': data['hybrid_stats']['hm_cf']['mean'],
                    'std': data['hybrid_stats']['hm_cf']['std'],
                    'total_evals': data['total_evaluations']
                }

    return results


def compute_borda_scores(results: Dict) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Compute Borda scores and MeanJudge scores for each VLM.

    Args:
        results: Dictionary with structure {embedding_model: {vlm_name: stats}}

    Returns:
        Tuple of (borda_scores, mean_judge_scores)
        - borda_scores: {vlm_name: total_borda_points}
        - mean_judge_scores: {vlm_name: mean_hm_across_judges}
    """
    # Get list of all VLMs (assuming all embedding models evaluated the same VLMs)
    embedding_models = list(results.keys())
    first_embedding = embedding_models[0]
    vlm_names = list(results[first_embedding].keys())

    N_vlms = len(vlm_names)
    N_embeds = len(embedding_models)

    print(f"Found {N_vlms} VLMs and {N_embeds} embedding models")
    print(f"VLMs: {vlm_names}")
    print(f"Embedding models: {embedding_models}")
    print()

    # Initialize Borda scores
    borda_scores = {vlm: 0 for vlm in vlm_names}

    # Store all HM scores for computing MeanJudge
    hm_scores = {vlm: [] for vlm in vlm_names}

    # Step 2: Compute Borda points per judge (embedding model)
    for embedding_model in embedding_models:
        # Collect S[j, k] = μ_hm[j, k] for all VLMs under this judge
        vlm_scores = []
        for vlm in vlm_names:
            hm_score = results[embedding_model][vlm]['hm']
            vlm_scores.append((vlm, hm_score))
            hm_scores[vlm].append(hm_score)

        # Sort VLMs by HM score in descending order
        vlm_scores.sort(key=lambda x: x[1], reverse=True)

        # Assign Borda points
        print(f"Judge: {embedding_model}")
        for rank, (vlm, score) in enumerate(vlm_scores, start=1):
            borda_points = N_vlms - rank
            borda_scores[vlm] += borda_points
            print(f"  Rank {rank}: {vlm} (HM={score:.6f}) → +{borda_points} Borda points")
        print()

    # Step 4: Compute MeanJudge score per VLM
    mean_judge_scores = {
        vlm: np.mean(hm_scores[vlm])
        for vlm in vlm_names
    }

    return borda_scores, mean_judge_scores


def generate_table1(borda_scores: Dict[str, int],
                    mean_judge_scores: Dict[str, float]) -> pd.DataFrame:
    """
    Generate Table 1: Overall VLM ranking summary.

    Returns:
        DataFrame with columns: [Rank, VLM, Borda, MeanJudge]
    """
    # Create list of (vlm, borda, mean_judge)
    data = []
    for vlm in borda_scores.keys():
        data.append({
            'VLM': vlm,
            'Borda': borda_scores[vlm],
            'MeanJudge': mean_judge_scores[vlm]
        })

    # Sort by Borda (primary), then MeanJudge (secondary)
    df = pd.DataFrame(data)
    df = df.sort_values(by=['Borda', 'MeanJudge'], ascending=[False, False])
    df.insert(0, 'Rank', range(1, len(df) + 1))

    return df


def generate_table2(results: Dict) -> pd.DataFrame:
    """
    Generate Table 2: Per-judge detailed statistics.

    Creates a hierarchical column structure with:
    - Top level: Embedding model names
    - Second level: Coarse, Fine, HM, Std

    Returns:
        DataFrame with MultiIndex columns
    """
    embedding_models = sorted(results.keys())
    vlm_names = sorted(list(results[embedding_models[0]].keys()))

    # Build data dictionary
    data = []
    for vlm in vlm_names:
        row = {'VLM': vlm}
        for embed in embedding_models:
            stats = results[embed][vlm]
            row[f'{embed}_Coarse'] = stats['coarse']
            row[f'{embed}_Fine'] = stats['fine']
            row[f'{embed}_HM'] = stats['hm']
            row[f'{embed}_Std'] = stats['std']
        data.append(row)

    df = pd.DataFrame(data)

    # Create MultiIndex columns
    columns = [('', 'VLM')]
    for embed in embedding_models:
        columns.extend([
            (embed, 'Coarse'),
            (embed, 'Fine'),
            (embed, 'HM'),
            (embed, 'Std')
        ])

    # Reorder dataframe columns to match the MultiIndex structure
    ordered_data = {'VLM': df['VLM']}
    for embed in embedding_models:
        ordered_data[f'{embed}_Coarse'] = df[f'{embed}_Coarse']
        ordered_data[f'{embed}_Fine'] = df[f'{embed}_Fine']
        ordered_data[f'{embed}_HM'] = df[f'{embed}_HM']
        ordered_data[f'{embed}_Std'] = df[f'{embed}_Std']

    df_ordered = pd.DataFrame(ordered_data)
    df_ordered.columns = pd.MultiIndex.from_tuples(columns)

    return df_ordered


def save_results(table1: pd.DataFrame, table2: pd.DataFrame, output_dir: str = "results/ranking"):
    """
    Save the ranking tables to CSV files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save Table 1
    table1_file = output_path / "table1_vlm_ranking.csv"
    table1.to_csv(table1_file, index=False, float_format='%.6f')
    print(f"Table 1 saved to: {table1_file}")

    # Save Table 2
    table2_file = output_path / "table2_detailed_stats.csv"
    table2.to_csv(table2_file, float_format='%.6f')
    print(f"Table 2 saved to: {table2_file}")

    # Also save a formatted version for LaTeX (if jinja2 is available)
    try:
        table1_latex = output_path / "table1_vlm_ranking.tex"
        with open(table1_latex, 'w') as f:
            f.write(table1.to_latex(index=False, float_format='%.4f'))
        print(f"Table 1 (LaTeX) saved to: {table1_latex}")
    except ImportError:
        print("Skipping LaTeX output (jinja2 not installed)")


def print_summary(table1: pd.DataFrame):
    """
    Print a summary of the ranking results.
    """
    print("=" * 80)
    print("TABLE 1: VLM RANKING SUMMARY")
    print("=" * 80)
    print(table1.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print()

    print("=" * 80)
    print("TOP 5 VLMs:")
    print("=" * 80)
    top5 = table1.head(5)
    for idx, row in top5.iterrows():
        print(f"{row['Rank']}. {row['VLM']}")
        print(f"   Borda Score: {row['Borda']}")
        print(f"   MeanJudge: {row['MeanJudge']:.6f}")
        print()


def main():
    """
    Main execution function.
    """
    print("=" * 80)
    print("VLM RANKING ALGORITHM")
    print("Based on Borda Count and Mean Judge Score")
    print("=" * 80)
    print()

    # Step 1: Load evaluation results
    print("Step 1: Loading evaluation results...")
    results = load_evaluation_results()
    print(f"Loaded results for {len(results)} embedding models")
    print()

    # Step 2-4: Compute Borda scores and MeanJudge
    print("Step 2-4: Computing Borda scores and MeanJudge scores...")
    borda_scores, mean_judge_scores = compute_borda_scores(results)
    print()

    # Print Borda scores summary
    print("=" * 80)
    print("BORDA SCORES SUMMARY")
    print("=" * 80)
    for vlm in sorted(borda_scores.keys(), key=lambda x: borda_scores[x], reverse=True):
        print(f"{vlm}: {borda_scores[vlm]} (MeanJudge: {mean_judge_scores[vlm]:.6f})")
    print()

    # Step 5: Generate tables
    print("Step 5: Generating ranking tables...")
    table1 = generate_table1(borda_scores, mean_judge_scores)
    table2 = generate_table2(results)
    print()

    # Print summary
    print_summary(table1)

    # Save results
    print("Step 6: Saving results...")
    save_results(table1, table2)
    print()

    print("=" * 80)
    print("ALGORITHM COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
