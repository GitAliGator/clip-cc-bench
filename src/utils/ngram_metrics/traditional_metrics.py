# ======================================
# Traditional N-Gram Metrics: BLEU, ROUGE and METEOR
# Adapted for clip-cc-bench
# ======================================

import threading
import re
import string
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer

# Thread-safe lock for NLTK resources (shared across all function calls)
_nltk_lock = threading.Lock()

# Try to import METEOR, handle gracefully if it fails
try:
    from nltk.translate.meteor_score import meteor_score
    METEOR_AVAILABLE = True
except:
    METEOR_AVAILABLE = False

def ensure_nltk_data():
    """Ensure required NLTK data is downloaded."""
    required_data = ['punkt', 'wordnet', 'averaged_perceptron_tagger']
    
    with _nltk_lock:
        for data_name in required_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}')
            except LookupError:
                try:
                    nltk.download(data_name, quiet=True)
                except:
                    pass  # Fail silently if download fails

def evaluate_traditional_metrics(reference, generated):
    """
    Evaluate traditional n-gram metrics: BLEU, METEOR, ROUGE.
    
    Args:
        reference (str): Ground truth/reference text
        generated (str): Generated/candidate text
        
    Returns:
        dict: Dictionary containing all computed metrics
    """
    # Ensure NLTK data is available
    ensure_nltk_data()
    
    # 1) Preprocessing for BLEU: Lowercase, remove punctuation, use NLTK tokenizer
    def preprocess_for_bleu(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        return word_tokenize(text)  # Use tokenizer instead of split()

    # 2) Preprocessing for METEOR: Standard NLTK tokenization (lowercase)
    def preprocess_for_meteor(text):
        return word_tokenize(text.lower())  # No need to remove punctuation; METEOR handles it

    # 3) Preprocessing for ROUGE (especially rougeLsum): Add a newline after each sentence
    def preprocess_for_rouge_lsum(text):
        text = text.strip()
        text = re.sub(r'([.!?])\\s*', r'\\1\\n', text)  # Handle multiple sentence delimiters
        return text

    #################################
    # BLEU Scores
    #################################
    ref_bleu = [preprocess_for_bleu(reference)]
    gen_bleu = preprocess_for_bleu(generated)
    smoothing = SmoothingFunction().method1

    bleu1 = sentence_bleu(ref_bleu, gen_bleu,
                          weights=(1, 0, 0, 0),
                          smoothing_function=smoothing)
    bleu2 = sentence_bleu(ref_bleu, gen_bleu,
                          weights=(0.5, 0.5, 0, 0),
                          smoothing_function=smoothing)
    bleu3 = sentence_bleu(ref_bleu, gen_bleu,
                          weights=(0.33, 0.33, 0.33, 0),
                          smoothing_function=smoothing)
    bleu4 = sentence_bleu(ref_bleu, gen_bleu,
                          weights=(0.25, 0.25, 0.25, 0.25),
                          smoothing_function=smoothing)

    #################################
    # METEOR
    #################################
    if METEOR_AVAILABLE:
        try:
            with _nltk_lock:  # Thread-safe access to NLTK resources
                ref_meteor = [preprocess_for_meteor(reference)]
                gen_meteor = preprocess_for_meteor(generated)
                meteor = meteor_score(ref_meteor, gen_meteor)
        except Exception as e:
            # Handle any WordNet corpus or METEOR-related errors gracefully
            meteor = 0.0
    else:
        meteor = 0.0

    #################################
    # ROUGE (including rougeLsum)
    #################################
    ref_rouge = preprocess_for_rouge_lsum(reference)
    gen_rouge = preprocess_for_rouge_lsum(generated)

    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL', 'rougeLsum'],
        use_stemmer=True  # Stemming helps generalization
    )
    raw_rouge = scorer.score(ref_rouge, gen_rouge)

    rouge_scores = {
        rouge_type: score_val.fmeasure
        for rouge_type, score_val in raw_rouge.items()
    }

    return {
        'BLEU-1': bleu1,
        'BLEU-2': bleu2,
        'BLEU-3': bleu3,
        'BLEU-4': bleu4,
        'METEOR': meteor,
        'ROUGE-1': rouge_scores.get('rouge1', 0.0),
        'ROUGE-2': rouge_scores.get('rouge2', 0.0), 
        'ROUGE-3': rouge_scores.get('rouge3', 0.0),
        'ROUGE-4': rouge_scores.get('rouge4', 0.0),
        'ROUGE-L': rouge_scores.get('rougeL', 0.0),
        'ROUGE-Lsum': rouge_scores.get('rougeLsum', 0.0)
    }

# Legacy function name for backward compatibility
def evaluate_BRM(reference, generated):
    """Legacy wrapper for backward compatibility with MPII code."""
    results = evaluate_traditional_metrics(reference, generated)
    
    # Convert to MPII format for compatibility
    return {
        'bleu1': results['BLEU-1'],
        'bleu2': results['BLEU-2'], 
        'bleu3': results['BLEU-3'],
        'bleu4': results['BLEU-4'],
        'meteor': results['METEOR'],
        'rouge': {
            'rouge1': results['ROUGE-1'],
            'rouge2': results['ROUGE-2'],
            'rouge3': results['ROUGE-3'], 
            'rouge4': results['ROUGE-4'],
            'rougeL': results['ROUGE-L'],
            'rougeLsum': results['ROUGE-Lsum']
        }
    }