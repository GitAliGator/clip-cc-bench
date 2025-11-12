"""
G-VEval LLaMA Scorer

LLaMA-based implementation of G-VEval scorer using ACCR rubrics.
Replicates G-VEval methodology but uses local LLaMA inference instead of GPT-4o.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class GVEvalResult:
    """Result from G-VEval LLaMA evaluation."""
    video_id: str
    reference_caption: str
    generated_caption: str
    accr_scores: Dict[str, float]  # accuracy, completeness, conciseness, relevance
    overall_score: float
    reasoning: str
    raw_response: str
    success: bool
    error_message: str = ""

class GVEvalLLaMAScorer:
    """G-VEval implementation using LLaMA instead of GPT-4o."""
    
    def __init__(self, model_config: Dict, logger=None):
        self.model_config = model_config
        self.logger = logger
        self.tokenizer = None
        self.model = None
        self.device = None
        
        # G-VEval specific settings
        self.accr_criteria = model_config.get('criteria', ['accuracy', 'completeness', 'conciseness', 'relevance'])
        self.score_range = model_config.get('score_range', [0, 100])
        self.prompt_template = None
        
    def load_model(self):
        """Load LLaMA model for G-VEval evaluation."""
        # Set memory optimization environment variables
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        if self.logger:
            self.logger.info(f"Loading LLaMA model for G-VEval: {self.model_config['model_path']}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['model_path'],
            trust_remote_code=self.model_config.get('trust_remote_code', True),
            local_files_only=self.model_config.get('local_files_only', True)
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config['model_path'],
            torch_dtype=getattr(torch, self.model_config.get('torch_dtype', 'bfloat16')),
            device_map=self.model_config.get('device_map', 'auto'),
            trust_remote_code=self.model_config.get('trust_remote_code', True),
            local_files_only=self.model_config.get('local_files_only', True)
        )
        
        self.device = next(self.model.parameters()).device
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.logger:
            self.logger.info(f"G-VEval LLaMA model loaded on device: {self.device}")
    
    def load_prompt_template(self, template_path: str):
        """Load G-VEval prompt template."""
        template_file = Path(template_path)
        if not template_file.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        
        with open(template_file, 'r') as f:
            self.prompt_template = f.read()
        
        if self.logger:
            self.logger.info(f"Loaded G-VEval prompt template: {template_path}")
    
    def evaluate_ref_only(self, video_id: str, reference_caption: str, generated_caption: str, 
                         rubric_type: str = 'accr') -> GVEvalResult:
        """Evaluate using G-VEval ref-only setting with ACCR rubrics."""
        try:
            # Ensure prompt template is loaded
            if self.prompt_template is None:
                template_path = self.model_config.get('prompt_template_path', 
                    'prompts/vid/accr/ref-only.txt')
                self.load_prompt_template(template_path)
            
            # Format prompt with captions
            formatted_prompt = self._format_gveval_prompt(reference_caption, generated_caption)
            
            # Generate evaluation using LLaMA
            raw_response = self._generate_llama_response(formatted_prompt)
            
            # Parse G-VEval response to extract ACCR scores
            accr_scores, reasoning = self._parse_gveval_response(raw_response)
            
            # Calculate overall score (average of ACCR scores)
            overall_score = sum(accr_scores.values()) / len(accr_scores) if accr_scores else 0.0
            
            return GVEvalResult(
                video_id=video_id,
                reference_caption=reference_caption,
                generated_caption=generated_caption,
                accr_scores=accr_scores,
                overall_score=overall_score,
                reasoning=reasoning,
                raw_response=raw_response,
                success=True
            )
            
        except torch.cuda.OutOfMemoryError as e:
            if self.logger:
                self.logger.error(f"G-VEval evaluation failed for {video_id}: CUDA out of memory. {e}")
                self.logger.warning(f"Evaluation failed for {video_id}: {e}")
            
            # Clean up GPU memory on OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Return default scores on memory failure
            default_scores = {criterion: 0.0 for criterion in self.accr_criteria}  # Use 0 for failed evaluations
            
            return GVEvalResult(
                video_id=video_id,
                reference_caption=reference_caption,
                generated_caption=generated_caption,
                accr_scores=default_scores,
                overall_score=0.0,
                reasoning="Evaluation failed due to CUDA out of memory",
                raw_response="",
                success=False,
                error_message=str(e)
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"G-VEval evaluation failed for {video_id}: {e}")
            
            # Return default scores on failure
            default_scores = {criterion: 50.0 for criterion in self.accr_criteria}
            
            return GVEvalResult(
                video_id=video_id,
                reference_caption=reference_caption,
                generated_caption=generated_caption,
                accr_scores=default_scores,
                overall_score=50.0,
                reasoning="Evaluation failed",
                raw_response="",
                success=False,
                error_message=str(e)
            )
    
    def _format_gveval_prompt(self, reference_caption: str, generated_caption: str) -> str:
        """Format the G-VEval prompt with reference and generated captions."""
        if self.prompt_template is None:
            raise ValueError("Prompt template not loaded")
        
        # Replace placeholders in the template
        formatted_prompt = self.prompt_template.replace('{reference_caption}', reference_caption)
        formatted_prompt = formatted_prompt.replace('{generated_caption}', generated_caption)
        
        return formatted_prompt
    
    def _generate_llama_response(self, prompt: str) -> str:
        """Generate response using LLaMA model."""
        if self.logger:
            self.logger.debug(f"G-VEval prompt length: {len(prompt)}")
        
        # Format prompt for LLaMA chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": "You are an expert evaluator for video captions. Provide detailed analysis and follow the exact format requested."},
                {"role": "user", "content": prompt}
            ]
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback if chat template fails
                formatted_prompt = prompt
        else:
            formatted_prompt = prompt
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Generation parameters optimized for G-VEval
        generation_config = {
            'max_new_tokens': self.model_config.get('max_new_tokens', 1024),
            'min_new_tokens': self.model_config.get('min_new_tokens', 100),
            'temperature': self.model_config.get('temperature', 0.1),
            'do_sample': self.model_config.get('do_sample', True),
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id
        }
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_config)
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up tensors immediately
            del inputs
            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except torch.cuda.OutOfMemoryError as e:
            # Clean up on CUDA OOM error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e
        
        # Remove input prompt from response
        if response.startswith(formatted_prompt):
            response = response[len(formatted_prompt):].strip()
        
        # If using chat template, extract only the assistant's response
        if "assistant" in response:
            parts = response.split("assistant")
            if len(parts) > 1:
                response = parts[-1].strip()
        
        return response
    
    def _parse_gveval_response(self, response: str) -> Tuple[Dict[str, float], str]:
        """Parse G-VEval response to extract ACCR scores using Greek letter notation."""
        accr_scores = {}
        reasoning_parts = []
        
        # Greek letter patterns for ACCR scores (as used in G-VEval)
        patterns = {
            'accuracy': r'α(\d+(?:\.\d+)?)α',
            'completeness': r'β(\d+(?:\.\d+)?)β', 
            'conciseness': r'ψ(\d+(?:\.\d+)?)ψ',
            'relevance': r'δ(\d+(?:\.\d+)?)δ'
        }
        
        # Extract scores using Greek letter notation
        for criterion, pattern in patterns.items():
            match = re.search(pattern, response)
            if match:
                score = float(match.group(1))
                # Ensure score is within valid range
                score = max(self.score_range[0], min(self.score_range[1], score))
                accr_scores[criterion] = score
            else:
                # Fallback: try to find score in text
                score = self._extract_score_fallback(response, criterion)
                accr_scores[criterion] = score
        
        # Extract reasoning (everything in the response)
        reasoning = response.strip()
        
        if self.logger:
            self.logger.debug(f"Parsed ACCR scores: {accr_scores}")
        
        return accr_scores, reasoning
    
    def _extract_score_fallback(self, response: str, criterion: str) -> float:
        """Fallback method to extract scores when Greek letters are not found."""
        # Look for patterns like "accuracy: 75" or "Accuracy Analysis: ... score of 80"
        patterns = [
            rf'{criterion}[^\d]*(\d+(?:\.\d+)?)',
            rf'{criterion.capitalize()}[^\d]*(\d+(?:\.\d+)?)',
            rf'score[^\d]*(\d+(?:\.\d+)?)'  # Generic score pattern
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                score = float(matches[0])
                # Ensure score is within valid range
                score = max(self.score_range[0], min(self.score_range[1], score))
                return score
        
        # Default score if nothing found
        return 50.0
    
    def batch_evaluate(self, evaluation_data: List[Dict]) -> List[GVEvalResult]:
        """Evaluate multiple samples in batch."""
        results = []
        
        for i, data in enumerate(evaluation_data):
            if self.logger and (i + 1) % 10 == 0:
                self.logger.info(f"Processed {i + 1}/{len(evaluation_data)} evaluations")
            
            result = self.evaluate_ref_only(
                video_id=data.get('video_id', str(i)),
                reference_caption=data['reference_caption'],
                generated_caption=data['generated_caption'],
                rubric_type=data.get('rubric_type', 'accr')
            )
            
            results.append(result)
        
        return results