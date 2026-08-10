import logging
import os

from processor.utils import collect_files, extract_archive, get_archive_type
from processor.zarr_converter import process_zarr_conversions

log = logging.getLogger(__name__)


class ArchiveExtractor:
    """Extracts archive files into the output directory."""

    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the extractor.

        Args:
            input_dir: Directory containing the input archive file
            output_dir: Directory for extracted output
        """
        self.input_dir = input_dir
        self.output_dir = output_dir

    def find_input_file(self) -> str:
        """
        Find the input archive file in the input directory.

        Supports: .zip, .tar, .tar.gz, .tgz, .tar.bz2, .tbz2, .tar.xz, .txz

        Returns:
            Path to the archive file

        Raises:
            FileNotFoundError: If no archive file is found
            ValueError: If multiple archive files are found
        """
        archive_files = [f for f in os.listdir(self.input_dir) if get_archive_type(f) is not None]
        if len(archive_files) == 0:
            raise FileNotFoundError("Expected exactly one archive file, found 0")
        if len(archive_files) > 1:
            raise ValueError(f"Expected exactly one archive file, found {len(archive_files)}")
        return os.path.join(self.input_dir, archive_files[0])

    def extract(self, archive_path: str) -> str:
        """
        Extract an archive file directly into the output directory.

        Args:
            archive_path: Path to the archive file

        Returns:
            Path to the output directory containing extracted files

        Raises:
            ValueError: If the archive format is not supported
        """
        log.info(f"Extracting archive: {archive_path}")

        os.makedirs(self.output_dir, exist_ok=True)
        extract_archive(archive_path, self.output_dir)

        log.info(f"Extracted archive to: {self.output_dir}")
        return self.output_dir

    def collect_extracted_files(self) -> list[tuple[str, str]]:
        """
        Collect all files extracted into the output directory.

        Returns:
            List of tuples (absolute_path, relative_path within output_dir)
        """
        files = collect_files(self.output_dir)
        log.info(f"Collected {len(files)} files from output directory")
        return files

    def process(self) -> tuple[str, list[tuple[str, str]]]:
        """
        Main processing method: find the input archive and extract it.

        Returns:
            Tuple of (output_dir, list of (abs_path, rel_path) tuples)
        """
        archive_path = self.find_input_file()
        self.extract(archive_path)

        converted = process_zarr_conversions(self.output_dir)
        if converted:
            log.info(f"Converted {len(converted)} zarr v2 store(s) to v3")

        files = self.collect_extracted_files()

        return self.output_dir, files
