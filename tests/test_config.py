import os
from unittest.mock import patch

from processor.config import Config


class TestConfig:
    """Tests for Config class."""

    def test_loads_input_and_output_dirs(self):
        """Should load INPUT_DIR and OUTPUT_DIR from env."""
        with patch.dict(os.environ, {"INPUT_DIR": "/in", "OUTPUT_DIR": "/out"}, clear=True):
            config = Config()
            assert config.INPUT_DIR == "/in"
            assert config.OUTPUT_DIR == "/out"

    def test_defaults_to_none_when_unset(self):
        """Should default to None when env vars are not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.INPUT_DIR is None
            assert config.OUTPUT_DIR is None
