"""Configuration management for the twibot package."""

import os
import datetime
from pathlib import Path
from typing import Optional
import yaml


class Config:
    """
    Configuration class that loads settings from YAML files.

    API keys are loaded from environment variables for security.
    """

    def __init__(self, config_path: str):
        """
        Initialize configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(self.config_path, 'r') as f:
            self._data = yaml.safe_load(f)

        # API key from environment variable
        self._api_key = os.environ.get('OPENROUTER_API_KEY')

    @property
    def api_key(self) -> Optional[str]:
        """OpenRouter API key from environment variable."""
        return self._api_key

    def require_api_key(self) -> str:
        """Get API key or raise error if not set."""
        if not self._api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Please set it with: export OPENROUTER_API_KEY='your-key'"
            )
        return self._api_key

    # Dataset configuration
    @property
    def dataset_name(self) -> str:
        """Name of the dataset."""
        return self._data.get('dataset', {}).get('name', 'twibot22')

    @property
    def dataset_path(self) -> str:
        """Path to the dataset directory."""
        return self._data.get('dataset', {}).get('path', './data')

    @property
    def reference_date(self) -> datetime.datetime:
        """Reference date for account age calculation."""
        date_str = self._data.get('dataset', {}).get('reference_date')
        if date_str:
            return datetime.datetime.strptime(date_str, '%Y-%m-%d').replace(
                tzinfo=datetime.timezone.utc
            )
        # TwiBot-22 collection reference date
        return datetime.datetime(2022, 1, 26, tzinfo=datetime.timezone.utc)

    @property
    def date_format(self) -> str:
        """Date format string for parsing created_at."""
        return self._data.get('dataset', {}).get('date_format', '%Y-%m-%d %H:%M:%S')

    # Embedding configuration
    @property
    def embedding_model(self) -> str:
        """Sentence transformer model for embeddings."""
        return self._data.get('embedding', {}).get('model', 'jinaai/jina-embeddings-v3')

    @property
    def embedding_batch_size(self) -> int:
        """Batch size for embedding computation."""
        return self._data.get('embedding', {}).get('batch_size', 64)

    @property
    def checkpoint_interval(self) -> int:
        """Checkpoint interval for resumable processing."""
        return self._data.get('embedding', {}).get('checkpoint_interval', 100)

    def to_dict(self) -> dict:
        """Return the raw configuration dictionary."""
        return self._data.copy()
