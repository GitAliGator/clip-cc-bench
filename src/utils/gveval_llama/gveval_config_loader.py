"""
G-VEval Configuration Loader

Standalone configuration loader for G-VEval LLaMA (no external dependencies).
"""

import yaml
from pathlib import Path
from typing import Dict, Any

class GVEvalConfigLoader:
    """Standalone configuration loader for G-VEval LLaMA."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load G-VEval configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate G-VEval configuration."""
        required_keys = [
            'gveval_llama',
            'processing', 
            'data_paths',
            'models_to_evaluate'
        ]
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Validate gveval_llama section
        gveval_config = config['gveval_llama']
        required_gveval_keys = [
            'model_path',
            'evaluation_mode',
            'rubric_type',
            'criteria'
        ]
        
        for key in required_gveval_keys:
            if key not in gveval_config:
                raise ValueError(f"Missing required G-VEval configuration key: {key}")
        
        # Validate criteria
        expected_criteria = ['accuracy', 'completeness', 'conciseness', 'relevance']
        actual_criteria = gveval_config['criteria']
        if set(actual_criteria) != set(expected_criteria):
            raise ValueError(f"Invalid criteria. Expected {expected_criteria}, got {actual_criteria}")
        
        # Validate paths
        model_path = Path(gveval_config['model_path'])
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        
        return True
    
    @staticmethod
    def get_prompt_path(config: Dict[str, Any]) -> Path:
        """Get the full path to the prompt template."""
        base_path = Path(__file__).parent
        prompt_relative_path = config['gveval_llama']['prompt_template_path']
        return base_path / prompt_relative_path