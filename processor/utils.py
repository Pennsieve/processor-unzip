import os
import tarfile
import zipfile

# Supported archive extensions (order matters - longer extensions first)
SUPPORTED_EXTENSIONS = [
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
    ".tar",
    ".zip",
]


def get_archive_type(filename: str) -> str | None:
    """
    Detect archive type from filename extension.

    Args:
        filename: Name of the file to check

    Returns:
        The matching extension (e.g., '.tar.gz', '.zip') or None if unsupported
    """
    lower = filename.lower()
    for ext in SUPPORTED_EXTENSIONS:
        if lower.endswith(ext):
            return ext
    return None


def extract_zip(zip_path: str, output_dir: str) -> str:
    """
    Extract a ZIP file to the specified directory.

    Args:
        zip_path: Path to the ZIP file
        output_dir: Directory to extract files to

    Returns:
        Path to the extraction directory
    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    return output_dir


def extract_tar(tar_path: str, output_dir: str) -> str:
    """
    Extract a tar archive to the specified directory.

    Handles .tar, .tar.gz, .tgz, .tar.bz2, .tbz2, .tar.xz, .txz formats
    automatically via tarfile's 'r:*' mode.

    Args:
        tar_path: Path to the tar archive
        output_dir: Directory to extract files to

    Returns:
        Path to the extraction directory
    """
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(output_dir)

    return output_dir


def extract_archive(archive_path: str, output_dir: str) -> str:
    """
    Extract an archive to the specified directory.

    Automatically detects format from filename and calls the appropriate
    extraction function.

    Args:
        archive_path: Path to the archive file
        output_dir: Directory to extract files to

    Returns:
        Path to the extraction directory

    Raises:
        ValueError: If archive format is not supported
    """
    archive_type = get_archive_type(os.path.basename(archive_path))
    if archive_type == ".zip":
        return extract_zip(archive_path, output_dir)
    elif archive_type is not None:
        # All other supported types are tar variants
        return extract_tar(archive_path, output_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def collect_files(directory: str) -> list[tuple[str, str]]:
    """
    Recursively collect all files in a directory.

    Args:
        directory: Root directory to collect files from

    Returns:
        List of tuples (absolute_path, relative_path)
    """
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            abs_path = os.path.join(root, filename)
            rel_path = os.path.relpath(abs_path, directory)
            files.append((abs_path, rel_path))

    return files
