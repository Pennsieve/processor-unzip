import json
import os
import tarfile
import zipfile

import pytest

from processor.extractor import ArchiveExtractor


class TestArchiveExtractor:
    """Tests for ArchiveExtractor class."""

    def test_find_input_file_single_zip(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "data.zip").write_bytes(b"fake zip")

        extractor = ArchiveExtractor(str(input_dir), str(tmp_path / "output"))
        result = extractor.find_input_file()

        assert result == str(input_dir / "data.zip")

    def test_find_input_file_no_archive(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "data.txt").write_text("not an archive")

        extractor = ArchiveExtractor(str(input_dir), str(tmp_path / "output"))

        with pytest.raises(FileNotFoundError, match="Expected exactly one archive file"):
            extractor.find_input_file()

    def test_find_input_file_multiple_archives(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "data1.zip").write_bytes(b"fake zip")
        (input_dir / "data2.zip").write_bytes(b"fake zip")

        extractor = ArchiveExtractor(str(input_dir), str(tmp_path / "output"))

        with pytest.raises(ValueError, match="Expected exactly one archive file"):
            extractor.find_input_file()

    def test_find_input_file_tar_gz(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "data.tar.gz").write_bytes(b"fake tar")

        extractor = ArchiveExtractor(str(input_dir), str(tmp_path / "output"))
        result = extractor.find_input_file()

        assert result == str(input_dir / "data.tar.gz")

    def test_extract_zip_directly_into_output_dir(self, tmp_path):
        """Files should land directly in output dir, not under a folder named after the zip."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        zip_path = input_dir / "my-data.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("file1.txt", "hello")
            zf.writestr("subdir/file2.txt", "world")

        extractor = ArchiveExtractor(str(input_dir), str(output_dir))
        result = extractor.extract(str(zip_path))

        assert result == str(output_dir)
        assert (output_dir / "file1.txt").read_text() == "hello"
        assert (output_dir / "subdir" / "file2.txt").read_text() == "world"
        # Confirm no wrapper folder was created from the zip basename
        assert not (output_dir / "my-data").exists()

    def test_extract_creates_output_dir_if_missing(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        zip_path = input_dir / "data.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("file1.txt", "hello")

        extractor = ArchiveExtractor(str(input_dir), str(output_dir))
        extractor.extract(str(zip_path))

        assert output_dir.exists()
        assert (output_dir / "file1.txt").read_text() == "hello"

    def test_extract_tar_gz_directly_into_output_dir(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Build a directory tree to add to the archive
        src = tmp_path / "src"
        src.mkdir()
        (src / "file1.txt").write_text("hello")
        (src / "subdir").mkdir()
        (src / "subdir" / "file2.txt").write_text("world")

        tar_path = input_dir / "bundle.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            for item in src.iterdir():
                tf.add(item, arcname=item.name)

        extractor = ArchiveExtractor(str(input_dir), str(output_dir))
        extractor.extract(str(tar_path))

        assert (output_dir / "file1.txt").read_text() == "hello"
        assert (output_dir / "subdir" / "file2.txt").read_text() == "world"
        assert not (output_dir / "bundle").exists()

    def test_collect_extracted_files(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        zip_path = input_dir / "data.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("a.txt", "a")
            zf.writestr("sub/b.txt", "b")

        extractor = ArchiveExtractor(str(input_dir), str(output_dir))
        extractor.extract(str(zip_path))
        files = extractor.collect_extracted_files()

        rel_paths = {rel for _, rel in files}
        assert "a.txt" in rel_paths
        assert os.path.join("sub", "b.txt") in rel_paths
        assert len(files) == 2

    def test_process_full_workflow_zip(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        zip_path = input_dir / "sample.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("file1.txt", "hello")
            zf.writestr("nested/file2.bin", b"\x00\x01\x02")

        extractor = ArchiveExtractor(str(input_dir), str(output_dir))
        result_dir, files = extractor.process()

        assert result_dir == str(output_dir)
        assert (output_dir / "file1.txt").read_text() == "hello"
        assert (output_dir / "nested" / "file2.bin").read_bytes() == b"\x00\x01\x02"
        assert len(files) == 2

    def test_process_full_workflow_tar_gz(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        src = tmp_path / "src"
        src.mkdir()
        (src / "file1.txt").write_text("hello")
        (src / "sub").mkdir()
        (src / "sub" / "file2.txt").write_text("world")

        tar_path = input_dir / "sample.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            for item in src.iterdir():
                tf.add(item, arcname=item.name)

        extractor = ArchiveExtractor(str(input_dir), str(output_dir))
        result_dir, files = extractor.process()

        assert result_dir == str(output_dir)
        assert (output_dir / "file1.txt").read_text() == "hello"
        assert (output_dir / "sub" / "file2.txt").read_text() == "world"
        assert len(files) == 2

    def test_process_converts_zarr_v2_to_v3_at_root(self, tmp_path):
        """Zip with a Zarr v2 store should be flattened to root and converted to v3."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        zarray = {
            "zarr_format": 2,
            "shape": [64, 64],
            "chunks": [32, 32],
            "dtype": "<u1",
            "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0},
            "fill_value": 0,
            "order": "C",
            "filters": None,
            "dimension_separator": "/",
        }

        zip_path = input_dir / "data.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("img.zarr/.zgroup", json.dumps({"zarr_format": 2}))
            zf.writestr("img.zarr/.zattrs", json.dumps({"key": "value"}))
            zf.writestr("img.zarr/0/.zarray", json.dumps(zarray))

        extractor = ArchiveExtractor(str(input_dir), str(output_dir))
        result_dir, files = extractor.process()

        # zarr.json at output root, not inside img.zarr/
        assert (output_dir / "zarr.json").exists()
        meta = json.loads((output_dir / "zarr.json").read_text())
        assert meta["zarr_format"] == 3
        assert meta["node_type"] == "group"

        assert not (output_dir / "img.zarr").exists()
        assert not (output_dir / ".zgroup").exists()
