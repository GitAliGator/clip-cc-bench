"""
Configuration Loader for N-Gram Metrics Evaluation

Handles YAML configuration loading and validation for the n-gram metrics system.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

class NgramConfigLoader:
    """Configuration loader for n-gram metrics evaluation."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        NgramConfigLoader.validate_config(config)
        return config
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate configuration structure and required fields."""
        required_sections = ['ngram_metrics', 'processing', 'data_paths', 'models_to_evaluate', 'logging']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate ngram_metrics section
        ngram_config = config['ngram_metrics']
        if 'metrics' not in ngram_config:
            raise ValueError("Missing 'metrics' in ngram_metrics configuration")
        
        # Validate data paths
        data_paths = config['data_paths']
        required_paths = ['ground_truth_file', 'predictions_dir', 'results_base_dir']
        
        for path_key in required_paths:
            if path_key not in data_paths:
                raise ValueError(f"Missing required data path: {path_key}")
        
        # Validate models list
        if not config['models_to_evaluate']:
            raise ValueError("No models specified for evaluation")
        
        print("✅ Configuration validation passed")