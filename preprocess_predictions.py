#!/usr/bin/env python3
"""
Preprocess Model Predictions - Clean Newline Characters

This script removes \n and \n\n from model predictions to match
the ground truth format for fair evaluation.

Usage:
    python preprocess_predictions.py --backup           # Create backup and clean
    python preprocess_predictions.py --dry-run          # Preview changes only
    python preprocess_predictions.py --models m1 m2     # Clean specific models
"""

import json
import re
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple


def clean_text(text: str) -> Tuple[str, Dict[str, int]]:
    """
    Clean newline characters from text and normalize whitespace.

    Args:
        text: Input text to clean

    Returns:
        Tuple of (cleaned_text, statistics_dict)
    """
    original_text = text
    stats = {
        'double_newlines': text.count('\n\n'),
        'single_newlines': text.count('\n') - text.count('\n\n') * 2,  # Don't double-count \n\n
        'original_length': len(text)
    }

    # Step 1: Replace double newlines with single space
    text = text.replace('\n\n', ' ')

    # Step 2: Replace single newlines with single space
    text = text.replace('\n', ' ')

    # Step 3: Normalize multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)

    # Step 4: Strip leading and trailing whitespace
    text = text.strip()

    stats['cleaned_length'] = len(text)
    stats['chars_removed'] = stats['original_length'] - stats['cleaned_length']
    stats['changed'] = original_text != text

    return text, stats


def process_file(input_path: Path, output_path: Path, dry_run: bool = False) -> Dict:
    """
    Process a single JSON file, cleaning all descriptions.

    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        dry_run: If True, don't write output, just return statistics

    Returns:
        Dictionary with processing statistics
    """
    # Load JSON data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    file_stats = {
        'filename': input_path.name,
        'total_descriptions': len(data),
        'modified_descriptions': 0,
        'total_double_newlines': 0,
        'total_single_newlines': 0,
        'total_chars_removed': 0,
        'samples': []
    }

    # Clean each description
    cleaned_data = {}
    for video_id, description in data.items():
        cleaned_desc, desc_stats = clean_text(description)
        cleaned_data[video_id] = cleaned_desc

        # Update file statistics
        if desc_stats['changed']:
            file_stats['modified_descriptions'] += 1

        file_stats['total_double_newlines'] += desc_stats['double_newlines']
        file_stats['total_single_newlines'] += desc_stats['single_newlines']
        file_stats['total_chars_removed'] += desc_stats['chars_removed']

        # Save samples for first 3 changed descriptions
        if desc_stats['changed'] and len(file_stats['samples']) < 3:
            file_stats['samples'].append({
                'video_id': video_id,
                'original': description[:150] + '...' if len(description) > 150 else description,
                'cleaned': cleaned_desc[:150] + '...' if len(cleaned_desc) > 150 else cleaned_desc,
                'stats': desc_stats
            })

    # Write cleaned data if not dry run
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

        # Validate JSON integrity
        with open(output_path, 'r', encoding='utf-8') as f:
            validation_data = json.load(f)

        if len(validation_data) != len(data):
            raise ValueError(f"Validation failed: {input_path.name} - entry count mismatch")

        file_stats['validated'] = True
    else:
        file_stats['validated'] = False

    return file_stats


def create_backup(models_dir: Path, backup_dir: Path) -> bool:
    """
    Create backup of all model prediction files.

    Args:
        models_dir: Source directory with model predictions
        backup_dir: Destination directory for backup

    Returns:
        True if backup successful
    """
    if backup_dir.exists():
        print(f"⚠️  Backup directory already exists: {backup_dir}")
        response = input("Overwrite existing backup? (yes/no): ").strip().lower()
        if response != 'yes':
            print("Backup cancelled.")
            return False
        shutil.rmtree(backup_dir)

    backup_dir.mkdir(parents=True, exist_ok=True)

    # Copy all JSON files
    json_files = list(models_dir.glob("*.json"))
    for json_file in json_files:
        shutil.copy2(json_file, backup_dir / json_file.name)

    print(f"✅ Backup created: {backup_dir} ({len(json_files)} files)")
    return True


def main():
    """Main preprocessing workflow."""
    parser = argparse.ArgumentParser(
        description='Preprocess video model predictions by removing newline characters',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without modifying files')
    parser.add_argument('--backup', action='store_true',
                        help='Create backup before processing (recommended)')
    parser.add_argument('--no-backup', action='store_true',
                        help='Skip backup creation (use with caution)')
    parser.add_argument('--models', nargs='+',
                        help='Specific models to process (e.g., internvl minicpm)')

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent
    models_dir = base_dir / "data" / "models"
    backup_dir = base_dir / "data" / "models_backup"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = results_dir / f"preprocessing_log_{timestamp}.txt"

    # Determine mode
    if args.dry_run:
        mode = "DRY-RUN (Preview Only)"
        output_dir = models_dir  # Not actually written
    else:
        mode = "PRODUCTION (Files will be modified)"
        output_dir = models_dir

    # Header
    print("=" * 70)
    print("🔧 MODEL PREDICTIONS PREPROCESSING")
    print("=" * 70)
    print(f"Mode: {mode}")
    print(f"Source: {models_dir}")
    print(f"Backup: {backup_dir if args.backup else 'NONE'}")
    print(f"Log file: {log_file}")
    print()

    # Get list of files to process
    if args.models:
        json_files = []
        for model_name in args.models:
            json_file = models_dir / f"{model_name}.json"
            if json_file.exists():
                json_files.append(json_file)
            else:
                print(f"⚠️  Warning: {model_name}.json not found, skipping")
    else:
        json_files = sorted(models_dir.glob("*.json"))

    if not json_files:
        print("❌ No files to process!")
        return

    print(f"📁 Files to process: {len(json_files)}")
    print()

    # Create backup if requested
    if args.backup and not args.dry_run:
        if not create_backup(models_dir, backup_dir):
            return
        print()
    elif not args.dry_run and not args.no_backup and not args.backup:
        print("⚠️  WARNING: No backup will be created!")
        print("Use --backup to create backup, or --no-backup to suppress this warning")
        response = input("Continue without backup? (yes/no): ").strip().lower()
        if response != 'yes':
            print("Operation cancelled.")
            return
        print()

    # Process files
    all_stats = []
    total_modified = 0
    total_unchanged = 0

    for json_file in json_files:
        print(f"Processing: {json_file.name}...", end=' ')

        try:
            output_path = output_dir / json_file.name
            stats = process_file(json_file, output_path, dry_run=args.dry_run)
            all_stats.append(stats)

            if stats['modified_descriptions'] > 0:
                print(f"✅ Modified {stats['modified_descriptions']}/{stats['total_descriptions']} descriptions")
                total_modified += 1
            else:
                print("⭕ No changes needed")
                total_unchanged += 1

        except Exception as e:
            print(f"❌ ERROR: {e}")
            continue

    # Summary
    print()
    print("=" * 70)
    print("📊 PREPROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total files processed: {len(all_stats)}")
    print(f"Files modified: {total_modified}")
    print(f"Files unchanged: {total_unchanged}")
    print()

    # Detailed statistics
    print("Detailed Changes:")
    print("-" * 70)
    nn = "\\n\\n"
    n = "\\n"
    print(f"{'Model':<25} | {'Modified':<8} | {nn:<6} | {n:<6} | {'Chars':<8}")
    print("-" * 70)

    for stats in all_stats:
        if stats['modified_descriptions'] > 0:
            print(f"{stats['filename']:<25} | "
                  f"{stats['modified_descriptions']:>3}/{stats['total_descriptions']:<3} | "
                  f"{stats['total_double_newlines']:>6} | "
                  f"{stats['total_single_newlines']:>6} | "
                  f"{stats['total_chars_removed']:>8}")

    print("-" * 70)
    print()

    # Save log
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL PREDICTIONS PREPROCESSING LOG\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Files processed: {len(all_stats)}\n")
        f.write(f"Files modified: {total_modified}\n")
        f.write(f"Files unchanged: {total_unchanged}\n")
        f.write("\n")

        for stats in all_stats:
            f.write(f"\n{'='*70}\n")
            f.write(f"File: {stats['filename']}\n")
            f.write(f"{'='*70}\n")
            f.write(f"Total descriptions: {stats['total_descriptions']}\n")
            f.write(f"Modified: {stats['modified_descriptions']}\n")
            f.write(f"Double newlines removed: {stats['total_double_newlines']}\n")
            f.write(f"Single newlines removed: {stats['total_single_newlines']}\n")
            f.write(f"Total characters removed: {stats['total_chars_removed']}\n")
            f.write(f"Validated: {stats.get('validated', 'N/A')}\n")

            if stats['samples']:
                f.write(f"\nSamples:\n")
                for i, sample in enumerate(stats['samples'], 1):
                    f.write(f"\n  Sample {i} (Video ID: {sample['video_id']}):\n")
                    f.write(f"    Original: {sample['original']}\n")
                    f.write(f"    Cleaned:  {sample['cleaned']}\n")
                    f.write(f"    Stats: {sample['stats']}\n")

    print(f"📄 Detailed log saved to: {log_file}")
    print()

    if args.dry_run:
        print("ℹ️  DRY-RUN MODE: No files were modified")
        print("   Run without --dry-run to apply changes")
    else:
        print("✅ Preprocessing complete!")
        if args.backup:
            print(f"   Backup: {backup_dir}")
        print(f"   Modified files: {models_dir}")

    print("=" * 70)


if __name__ == "__main__":
    main()
