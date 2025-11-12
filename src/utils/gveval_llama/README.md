# G-VEval LLaMA Implementation

Standalone implementation of G-VEval using LLaMA inference, replicating the methodology from "G-VEval: A Versatile Metric for Evaluating Image and Video Captions Using GPT-4o" (arXiv:2412.13647).

## Overview

This implementation provides:
- **ACCR Rubric Evaluation**: Accuracy, Completeness, Conciseness, Relevance
- **Ref-Only Setting**: Evaluates generated captions against reference captions only
- **LLaMA-based Inference**: Uses local LLaMA-3.1-8B-Instruct instead of GPT-4o
- **Standalone Design**: No dependencies on existing multi-judge ensemble

## Directory Structure

```
src/utils/gveval_llama/
├── __init__.py                     # Package initialization
├── gveval_config_loader.py         # Configuration loading and validation
├── gveval_llama_scorer.py          # Core scorer implementation
├── gveval_llama_core.py            # Main evaluator orchestration
├── gveval_utils.py                 # Utility functions
├── README.md                       # This file
└── prompts/
    ├── __init__.py
    └── vid/
        ├── ref-only.txt            # Basic G-VEval prompt
        └── accr/
            └── ref-only.txt        # ACCR rubric prompt
```

## Key Components

### GVEvalLLaMAScorer
- Loads LLaMA model for evaluation
- Implements ACCR rubric scoring (0-100 scale)
- Parses responses using Greek letter notation (α, β, ψ, δ)
- Handles fallback score extraction

### GVEvalLLaMAEvaluator
- Orchestrates complete evaluation process
- Handles data loading and alignment
- Manages results saving in structured format
- Provides logging and progress tracking

### Configuration
- YAML-based configuration in `src/config/gveval_llama_config.yaml`
- Supports model path, generation parameters, data paths
- Validates configuration on load

## Usage

### Quick Test
```bash
python src/scripts/test_gveval_llama.py
```

### Full Evaluation
```bash
python src/scripts/run_gveval_llama_evaluation.py
```

### Results Location
- Individual results: `results/GVEval_LLaMA/individual_results/`
- Aggregated results: `results/GVEval_LLaMA/aggregated_results/`

## G-VEval Methodology

### ACCR Rubrics
1. **Accuracy (α)**: Factual correctness of entities, actions, events
2. **Completeness (β)**: Coverage of significant events and details
3. **Conciseness (ψ)**: Clarity and freedom from redundancy
4. **Relevance (δ)**: Pertinence to video content

### Scoring
- Scale: 0-100 for each criterion
- Greek letter notation for score extraction
- Overall score: average of ACCR scores
- Fallback patterns for robust parsing

### Ref-Only Setting
- Uses only reference captions (no direct video input)
- Compares generated caption against reference
- Chain-of-thought reasoning for evaluation
- Detailed reasoning provided for each criterion

## Configuration

Key settings in `gveval_llama_config.yaml`:

```yaml
gveval_llama:
  model_path: "/path/to/Llama-3.1-8B-Instruct/"
  evaluation_mode: "ref-only"
  rubric_type: "accr"
  temperature: 0.1
  score_range: [0, 100]
  criteria: ["accuracy", "completeness", "conciseness", "relevance"]
```

## Dependencies

- PyTorch
- Transformers
- PyYAML
- Local LLaMA-3.1-8B-Instruct model

## Output Format

### Individual Results
```json
{
  "video_id": "001",
  "reference_caption": "...",
  "generated_caption": "...", 
  "accr_scores": {
    "accuracy": 75.0,
    "completeness": 80.0,
    "conciseness": 70.0,
    "relevance": 85.0
  },
  "overall_score": 77.5,
  "reasoning": "...",
  "success": true
}
```

### Summary Results
```json
{
  "model_name": "example_model",
  "total_samples": 199,
  "average_gveval_score": 77.5,
  "accr_criterion_averages": {...},
  "success_rate": 0.95,
  "evaluation_method": "gveval_llama_ref_only"
}
```

## Troubleshooting

### Common Issues
1. **Model not found**: Check `model_path` in config
2. **CUDA out of memory**: Reduce batch size or use CPU
3. **Missing predictions**: Ensure prediction files exist in correct format
4. **Parsing failures**: Check prompt template and response format

### Debugging
- Check logs in `src/logs/gveval_llama_evaluation_*.log`
- Use test script to verify individual components
- Examine raw responses for parsing issues

## Comparison with Original G-VEval

| Aspect | Original G-VEval | This Implementation |
|--------|------------------|-------------------|
| Model | GPT-4o | LLaMA-3.1-8B-Instruct |
| API | OpenAI API | Local inference |
| Scoring | 0-100 | 0-100 (maintained) |
| Rubric | ACCR | ACCR (replicated) |
| Parsing | Token probabilities | Greek letter + fallback |
| Setting | Ref-only | Ref-only (replicated) |