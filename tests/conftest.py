"""
pytest configuration for tradebot tests.

Adds project root to sys.path so that all imports
(e.g. from utils.token_definitions) work without
requiring an installed package.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
