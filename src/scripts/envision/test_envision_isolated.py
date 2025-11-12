#!/usr/bin/env python3
"""
En-Vision Isolated Test Script - Auto-approve custom code
"""

import subprocess
import sys
import os
from pathlib import Path

def run_envision_evaluation():
    """Run En-Vision evaluation with auto-approval of custom code."""

    # Set up environment
    script_dir = Path(__file__).parent
    config_dir = script_dir.parent.parent / "config" / "envision"
    activate_script = config_dir / "activate_env.sh"

    # Set environment variables
    env = os.environ.copy()
    env['TRANSFORMERS_TRUST_REMOTE_CODE'] = 'true'
    env['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = 'true'
    env['PYTHONUNBUFFERED'] = '1'

    # Command to run
    cmd = [
        'bash', '-c',
        f'source {activate_script} && python run_envision_evaluation.py --models internvl'
    ]

    print("🚀 Running En-Vision evaluation with auto-approval...")

    # Run with auto-response
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=script_dir
    )

    # Auto-respond "y" to prompts
    try:
        stdout, stderr = process.communicate(input="y\ny\ny\ny\n", timeout=300)

        print("STDOUT:")
        print(stdout)

        if stderr:
            print("STDERR:")
            print(stderr)

        return process.returncode == 0

    except subprocess.TimeoutExpired:
        process.kill()
        print("❌ Process timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        return False

if __name__ == "__main__":
    success = run_envision_evaluation()
    sys.exit(0 if success else 1)