import sys
from pathlib import Path

# Add project root directory to sys.path so app and model modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app

# Vercel WSGI entry point
# Exports the Flask 'app' instance
