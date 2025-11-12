#!/bin/bash
# En-Vision Environment Activation Script

echo "⚡ Activating En-Vision environment..."
source "/mmfs2/home/jacks.local/mali9292/aaai_student_abstract/clip-cc-bench/src/config/envision/venv/bin/activate"

echo "✅ En-Vision environment activated"
echo "📁 Virtual environment: /mmfs2/home/jacks.local/mali9292/aaai_student_abstract/clip-cc-bench/src/config/envision/venv"

# Show Python and key package versions
python -c "
import sys
import torch
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# Set environment variables for better performance
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0

echo ""
echo "🚀 Ready to run En-Vision evaluation!"
echo "   Run: python /mmfs2/home/jacks.local/mali9292/aaai_student_abstract/clip-cc-bench/src/scripts/envision/run_envision_evaluation.py"
echo "   Deactivate: deactivate"
