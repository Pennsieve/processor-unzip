import os
import tarfile
import zipfile

import pytest

from processor.utils import (
    collect_files,
    extract_archive,
    extract_tar,
    extract_zip,
    get_archive_type,
)


class TestGetArchiveType:
    """Tests for get_archive_type function."""

    def test_returns_zip_for_zip_file(self):
        assert get_archive_type("data.zip") == ".zip"
        assert get_archive_type("nested.name.zip") == ".zip"

    def test_returns_tar_gz_for_gzipped_tarball(self):
        assert get_archive_type("data.tar.gz") == ".tar.gz"

    def test_returns_tgz_for_tgz_extension(self):
        assert get_archive_type("data.tgz") == ".tgz"

    def test_returns_tar_bz2_for_bzip2_tarball(self):
        assert get_archive_type("data.tar.bz2") == ".tar.bz2"
        assert get_archive_type("data.tbz2") == ".tbz2"

    def test_returns_tar_xz_for_xz_tarball(self):
        assert get_archive_type("data.tar.xz") == ".tar.xz"
        assert get_archive_type("data.txz") == ".txz"

    def test_returns_tar_for_plain_tarball(self):
        assert get_archive_type("data.tar") == ".tar"

    def test_returns_none_for_unsupported(self):
        assert get_archive_type("data.txt") is None
        assert get_archive_type("data.rar") is None
        assert get_archive_type("data.7z") is None

    def test_case_insensitive(self):
        assert get_archive_type("data.ZIP") == ".zip"
        assert get_archive_type("data.TAR.GZ") == ".tar.gz"
        assert get_archive_type("data.TGZ") == ".tgz"


class TestExtractZip:
    """Tests for extract_zip function."""

    def test_extracts_zip_contents(self, tmp_path):
        zip_path = tmp_path / "test.zip"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("file1.txt", "content1")
            zf.writestr("subdir/file2.txt", "content2")

        result = extract_zip(str(zip_path), str(output_dir))

        assert result == str(output_dir)
        assert (output_dir / "file1.txt").exists()
        assert (output_dir / "subdir" / "file2.txt").exists()
        assert (output_dir / "file1.txt").read_text() == "content1"


class TestExtractTar:
    """Tests for extract_tar function."""

    def test_extracts_tar_gz_contents(self, tmp_path):
        tar_path = tmp_path / "test.tar.gz"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with tarfile.open(tar_path, "w:gz") as tf:
            file1 = tmp_path / "file1.txt"
            file1.write_text("content1")
            tf.add(file1, arcname="file1.txt")

            subdir = tmp_path / "subdir"
            subdir.mkdir()
            file2 = subdir / "file2.txt"
            file2.write_text("content2")
            tf.add(file2, arcname="subdir/file2.txt")

        result = extract_tar(str(tar_path), str(output_dir))

        assert result == str(output_dir)
        assert (output_dir / "file1.txt").read_text() == "content1"
        assert (output_dir / "subdir" / "file2.txt").read_text() == "content2"

    def test_extracts_tar_bz2_contents(self, tmp_path):
        tar_path = tmp_path / "test.tar.bz2"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with tarfile.open(tar_path, "w:bz2") as tf:
            file1 = tmp_path / "file1.txt"
            file1.write_text("content1")
            tf.add(file1, arcname="file1.txt")

        result = extract_tar(str(tar_path), str(output_dir))

        assert result == str(output_dir)
        assert (output_dir / "file1.txt").read_text() == "content1"

    def test_extracts_plain_tar_contents(self, tmp_path):
        tar_path = tmp_path / "test.tar"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with tarfile.open(tar_path, "w") as tf:
            file1 = tmp_path / "file1.txt"
            file1.write_text("content1")
            tf.add(file1, arcname="file1.txt")

        result = extract_tar(str(tar_path), str(output_dir))

        assert result == str(output_dir)
        assert (output_dir / "file1.txt").read_text() == "content1"


class TestExtractArchive:
    """Tests for extract_archive function."""

    def test_extracts_zip_file(self, tmp_path):
        zip_path = tmp_path / "test.zip"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("file1.txt", "content1")

        result = extract_archive(str(zip_path), str(output_dir))

        assert result == str(output_dir)
        assert (output_dir / "file1.txt").read_text() == "content1"

    def test_extracts_tar_gz_file(self, tmp_path):
        tar_path = tmp_path / "test.tar.gz"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        file1 = tmp_path / "file1.txt"
        file1.write_text("content1")
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(file1, arcname="file1.txt")

        result = extract_archive(str(tar_path), str(output_dir))

        assert result == str(output_dir)
        assert (output_dir / "file1.txt").read_text() == "content1"

    def test_raises_for_unsupported_format(self, tmp_path):
        unsupported_path = tmp_path / "test.rar"
        unsupported_path.write_bytes(b"fake")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with pytest.raises(ValueError, match="Unsupported archive format"):
            extract_archive(str(unsupported_path), str(output_dir))


class TestCollectFiles:
    """Tests for collect_files function."""

    def test_collects_all_files_recursively(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.txt").write_text("b")
        nested = sub / "nested"
        nested.mkdir()
        (nested / "c.bin").write_bytes(b"\x00")

        files = collect_files(str(tmp_path))

        files_dict = {rel: abs_ for abs_, rel in files}
        assert "a.txt" in files_dict
        assert os.path.join("sub", "b.txt") in files_dict
        assert os.path.join("sub", "nested", "c.bin") in files_dict
        assert len(files) == 3

    def test_returns_empty_for_empty_directory(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        assert collect_files(str(empty_dir)) == []
