"""Shared location of the project environment file."""

from pathlib import Path


PROJECT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
