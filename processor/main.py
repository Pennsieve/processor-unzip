import logging

from processor.config import Config
from processor.extractor import ArchiveExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def main():
    """Main entry point for the unzip processor."""
    log.info("Starting unzip processor")

    config = Config()

    if not config.INPUT_DIR:
        raise ValueError("INPUT_DIR environment variable is required")
    if not config.OUTPUT_DIR:
        raise ValueError("OUTPUT_DIR environment variable is required")

    extractor = ArchiveExtractor(config.INPUT_DIR, config.OUTPUT_DIR)
    output_dir, files = extractor.process()

    log.info(f"Extracted {len(files)} files into {output_dir}")
    log.info("Unzip processor completed successfully")


if __name__ == "__main__":
    main()
