import os


class Config:
    """Configuration management for the unzip processor."""

    def __init__(self):
        self.INPUT_DIR = os.getenv("INPUT_DIR")
        self.OUTPUT_DIR = os.getenv("OUTPUT_DIR")
