import json
import os
import zipfile

import pytest

from processor.zarr_converter import (
    _build_v3_array_metadata,
    _build_v3_group_metadata,
    _map_compressor_v2_to_v3,
    _parse_dtype,
    convert_zarr_v2_to_v3,
    find_zarr_stores,
    flatten_zarr_store,
    is_zarr_v2,
    process_zarr_conversions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _make_v2_store(root, zattrs=None, arrays=None):
    """Create a minimal Zarr v2 store on disk.

    Args:
        root: Path to the store root directory.
        zattrs: Optional dict for root .zattrs.
        arrays: Dict mapping array name → .zarray dict.
    """
    os.makedirs(root, exist_ok=True)
    _write_json(os.path.join(root, ".zgroup"), {"zarr_format": 2})
    if zattrs:
        _write_json(os.path.join(root, ".zattrs"), zattrs)
    for name, zarray in (arrays or {}).items():
        arr_dir = os.path.join(root, name)
        os.makedirs(arr_dir, exist_ok=True)
        _write_json(os.path.join(arr_dir, ".zarray"), zarray)


SAMPLE_ZARRAY = {
    "zarr_format": 2,
    "shape": [100, 200],
    "chunks": [10, 20],
    "dtype": "<i2",
    "compressor": {
        "id": "blosc",
        "cname": "lz4",
        "clevel": 5,
        "shuffle": 1,
        "blocksize": 0,
    },
    "fill_value": 0,
    "order": "C",
    "filters": None,
    "dimension_separator": "/",
}


# ---------------------------------------------------------------------------
# Flatten tests
# ---------------------------------------------------------------------------

class TestFlattenZarrStore:
    def test_flattens_v2_store(self, tmp_path):
        output = tmp_path / "output"
        store = output / "my.zarr"
        _make_v2_store(str(store), arrays={"0": SAMPLE_ZARRAY})

        result = flatten_zarr_store(str(output))

        assert result == str(store)
        assert (output / ".zgroup").exists()
        assert (output / "0" / ".zarray").exists()
        assert not store.exists()

    def test_flattens_v3_store(self, tmp_path):
        output = tmp_path / "output"
        store = output / "my.zarr"
        store.mkdir(parents=True)
        _write_json(str(store / "zarr.json"), {"zarr_format": 3, "node_type": "group", "attributes": {}})
        (store / "0").mkdir()
        _write_json(str(store / "0" / "zarr.json"), {"zarr_format": 3, "node_type": "array"})

        result = flatten_zarr_store(str(output))

        assert result is not None
        assert (output / "zarr.json").exists()
        assert (output / "0" / "zarr.json").exists()
        assert not store.exists()

    def test_noop_when_already_at_root(self, tmp_path):
        output = tmp_path / "output"
        _make_v2_store(str(output), arrays={"0": SAMPLE_ZARRAY})

        result = flatten_zarr_store(str(output))

        assert result is None
        assert (output / ".zgroup").exists()

    def test_noop_when_no_zarr(self, tmp_path):
        output = tmp_path / "output"
        output.mkdir()
        (output / "readme.txt").write_text("hello")

        result = flatten_zarr_store(str(output))

        assert result is None

    def test_preserves_chunk_data(self, tmp_path):
        output = tmp_path / "output"
        store = output / "data.zarr"
        _make_v2_store(str(store), arrays={"0": SAMPLE_ZARRAY})
        chunk_dir = store / "0" / "0"
        chunk_dir.mkdir(parents=True)
        (chunk_dir / "0").write_bytes(b"\xde\xad\xbe\xef")

        flatten_zarr_store(str(output))

        assert (output / "0" / "0" / "0").read_bytes() == b"\xde\xad\xbe\xef"


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------

class TestIsZarrV2:
    def test_valid_v2(self, tmp_path):
        store = tmp_path / "test.zarr"
        _make_v2_store(str(store))
        assert is_zarr_v2(str(store)) is True

    def test_already_v3(self, tmp_path):
        store = tmp_path / "test.zarr"
        _make_v2_store(str(store))
        _write_json(os.path.join(str(store), "zarr.json"), {"zarr_format": 3})
        assert is_zarr_v2(str(store)) is False

    def test_no_zgroup(self, tmp_path):
        store = tmp_path / "test.zarr"
        store.mkdir()
        assert is_zarr_v2(str(store)) is False

    def test_invalid_json(self, tmp_path):
        store = tmp_path / "test.zarr"
        store.mkdir()
        (store / ".zgroup").write_text("not json")
        assert is_zarr_v2(str(store)) is False


class TestFindZarrStores:
    def test_finds_root_store(self, tmp_path):
        store = tmp_path / "output" / "my.zarr"
        _make_v2_store(str(store))
        result = find_zarr_stores(str(tmp_path / "output"))
        assert result == [str(store)]

    def test_excludes_nested_subgroups(self, tmp_path):
        """Nested .zgroup (sub-group) should not appear as a separate store."""
        store = tmp_path / "output" / "my.zarr"
        _make_v2_store(str(store))
        # Add a nested sub-group
        sub = store / "labels"
        sub.mkdir()
        _write_json(str(sub / ".zgroup"), {"zarr_format": 2})

        result = find_zarr_stores(str(tmp_path / "output"))
        assert result == [str(store)]

    def test_multiple_stores(self, tmp_path):
        store_a = tmp_path / "output" / "a.zarr"
        store_b = tmp_path / "output" / "b.zarr"
        _make_v2_store(str(store_a))
        _make_v2_store(str(store_b))

        result = find_zarr_stores(str(tmp_path / "output"))
        assert sorted(result) == sorted([str(store_a), str(store_b)])

    def test_empty_dir(self, tmp_path):
        output = tmp_path / "output"
        output.mkdir()
        assert find_zarr_stores(str(output)) == []

    def test_non_zarr_content_ignored(self, tmp_path):
        output = tmp_path / "output"
        output.mkdir()
        (output / "readme.txt").write_text("hello")
        assert find_zarr_stores(str(output)) == []


# ---------------------------------------------------------------------------
# Dtype mapping
# ---------------------------------------------------------------------------

class TestParseDtype:
    @pytest.mark.parametrize(
        "dtype_str, expected",
        [
            ("<i2", ("int16", 2, "little")),
            ("<i4", ("int32", 4, "little")),
            (">f4", ("float32", 4, "big")),
            ("|u1", ("uint8", 1, "little")),
            ("<f8", ("float64", 8, "little")),
            (">i8", ("int64", 8, "big")),
            ("<u2", ("uint16", 2, "little")),
            ("|b1", ("bool", 1, "little")),
        ],
    )
    def test_known_dtypes(self, dtype_str, expected):
        assert _parse_dtype(dtype_str) == expected

    def test_unsupported_dtype_raises(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            _parse_dtype("<c16")


# ---------------------------------------------------------------------------
# Compressor mapping
# ---------------------------------------------------------------------------

class TestMapCompressor:
    def test_blosc(self):
        compressor = {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0}
        result = _map_compressor_v2_to_v3(compressor, typesize=2)
        assert len(result) == 1
        codec = result[0]
        assert codec["name"] == "blosc"
        assert codec["configuration"]["cname"] == "lz4"
        assert codec["configuration"]["shuffle"] == "shuffle"
        assert codec["configuration"]["typesize"] == 2

    def test_blosc_bitshuffle(self):
        compressor = {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 2, "blocksize": 0}
        result = _map_compressor_v2_to_v3(compressor, typesize=4)
        assert result[0]["configuration"]["shuffle"] == "bitshuffle"

    def test_blosc_noshuffle(self):
        compressor = {"id": "blosc", "cname": "lz4", "clevel": 1, "shuffle": 0, "blocksize": 0}
        result = _map_compressor_v2_to_v3(compressor, typesize=4)
        assert result[0]["configuration"]["shuffle"] == "noshuffle"

    def test_null_compressor(self):
        assert _map_compressor_v2_to_v3(None, typesize=4) == []

    def test_zlib(self):
        compressor = {"id": "zlib", "level": 6}
        result = _map_compressor_v2_to_v3(compressor, typesize=4)
        assert result[0]["name"] == "gzip"
        assert result[0]["configuration"]["level"] == 6

    def test_gzip(self):
        compressor = {"id": "gzip", "level": 3}
        result = _map_compressor_v2_to_v3(compressor, typesize=4)
        assert result[0]["name"] == "gzip"


# ---------------------------------------------------------------------------
# Metadata building
# ---------------------------------------------------------------------------

class TestBuildV3ArrayMetadata:
    def test_basic_array(self):
        meta = _build_v3_array_metadata(SAMPLE_ZARRAY)
        assert meta["zarr_format"] == 3
        assert meta["node_type"] == "array"
        assert meta["shape"] == [100, 200]
        assert meta["data_type"] == "int16"
        assert meta["chunk_grid"]["configuration"]["chunk_shape"] == [10, 20]
        assert meta["chunk_key_encoding"]["configuration"]["separator"] == "/"
        assert meta["fill_value"] == 0
        assert meta["attributes"] == {}

    def test_codecs_structure(self):
        meta = _build_v3_array_metadata(SAMPLE_ZARRAY)
        codecs = meta["codecs"]
        assert len(codecs) == 2  # bytes + blosc
        assert codecs[0]["name"] == "bytes"
        assert codecs[0]["configuration"]["endian"] == "little"
        assert codecs[1]["name"] == "blosc"

    def test_with_attributes(self):
        attrs = {"_ARRAY_DIMENSIONS": ["y", "x"]}
        meta = _build_v3_array_metadata(SAMPLE_ZARRAY, zattrs=attrs)
        assert meta["attributes"] == attrs

    def test_null_fill_value_becomes_zero(self):
        zarray = {**SAMPLE_ZARRAY, "fill_value": None}
        meta = _build_v3_array_metadata(zarray)
        assert meta["fill_value"] == 0

    def test_no_compressor(self):
        zarray = {**SAMPLE_ZARRAY, "compressor": None}
        meta = _build_v3_array_metadata(zarray)
        codecs = meta["codecs"]
        assert len(codecs) == 1  # bytes only
        assert codecs[0]["name"] == "bytes"

    def test_f_order_adds_transpose(self):
        zarray = {**SAMPLE_ZARRAY, "order": "F"}
        meta = _build_v3_array_metadata(zarray)
        codecs = meta["codecs"]
        assert codecs[0]["name"] == "transpose"
        assert codecs[0]["configuration"]["order"] == [1, 0]
        assert codecs[1]["name"] == "bytes"

    def test_default_dimension_separator(self):
        zarray = {k: v for k, v in SAMPLE_ZARRAY.items() if k != "dimension_separator"}
        meta = _build_v3_array_metadata(zarray)
        assert meta["chunk_key_encoding"]["configuration"]["separator"] == "."


class TestBuildV3GroupMetadata:
    def test_empty_group(self):
        meta = _build_v3_group_metadata()
        assert meta == {"zarr_format": 3, "node_type": "group", "attributes": {}}

    def test_with_attrs(self):
        attrs = {"multiscales": [{"version": "0.4"}]}
        meta = _build_v3_group_metadata(attrs)
        assert meta["attributes"] == attrs


# ---------------------------------------------------------------------------
# Full conversion
# ---------------------------------------------------------------------------

class TestConvertZarrV2ToV3:
    def test_converts_metadata(self, tmp_path):
        store = tmp_path / "test.zarr"
        _make_v2_store(
            str(store),
            zattrs={"multiscales": [{"version": "0.4"}]},
            arrays={"0": SAMPLE_ZARRAY},
        )
        # Add a fake chunk file to verify it's untouched
        chunk_dir = store / "0" / "0"
        chunk_dir.mkdir(parents=True)
        chunk_file = chunk_dir / "0"
        chunk_file.write_bytes(b"\x00\x01\x02\x03")

        convert_zarr_v2_to_v3(str(store))

        # v3 zarr.json files created
        root_meta = json.loads((store / "zarr.json").read_text())
        assert root_meta["zarr_format"] == 3
        assert root_meta["node_type"] == "group"
        assert root_meta["attributes"]["multiscales"][0]["version"] == "0.4"

        array_meta = json.loads((store / "0" / "zarr.json").read_text())
        assert array_meta["zarr_format"] == 3
        assert array_meta["node_type"] == "array"
        assert array_meta["data_type"] == "int16"

        # v2 files removed
        assert not (store / ".zgroup").exists()
        assert not (store / ".zattrs").exists()
        assert not (store / "0" / ".zarray").exists()

        # Chunk data untouched
        assert chunk_file.read_bytes() == b"\x00\x01\x02\x03"

    def test_multiple_arrays(self, tmp_path):
        store = tmp_path / "test.zarr"
        zarray_f4 = {**SAMPLE_ZARRAY, "dtype": "<f4"}
        _make_v2_store(str(store), arrays={"0": SAMPLE_ZARRAY, "1": zarray_f4})

        convert_zarr_v2_to_v3(str(store))

        meta_0 = json.loads((store / "0" / "zarr.json").read_text())
        meta_1 = json.loads((store / "1" / "zarr.json").read_text())
        assert meta_0["data_type"] == "int16"
        assert meta_1["data_type"] == "float32"


class TestProcessZarrConversions:
    def test_converts_v2_store(self, tmp_path):
        store = tmp_path / "output" / "my.zarr"
        _make_v2_store(str(store), arrays={"0": SAMPLE_ZARRAY})

        converted = process_zarr_conversions(str(tmp_path / "output"))

        assert converted == [str(store)]
        assert (store / "zarr.json").exists()
        assert not (store / ".zgroup").exists()

    def test_skips_v3_store(self, tmp_path):
        store = tmp_path / "output" / "my.zarr"
        _make_v2_store(str(store))
        _write_json(os.path.join(str(store), "zarr.json"), {"zarr_format": 3, "node_type": "group", "attributes": {}})

        converted = process_zarr_conversions(str(tmp_path / "output"))
        assert converted == []

    def test_non_zarr_unaffected(self, tmp_path):
        output = tmp_path / "output"
        output.mkdir()
        txt = output / "readme.txt"
        txt.write_text("hello")

        converted = process_zarr_conversions(str(output))

        assert converted == []
        assert txt.read_text() == "hello"

    def test_mixed_content(self, tmp_path):
        output = tmp_path / "output"
        store = output / "data.zarr"
        _make_v2_store(str(store), arrays={"0": SAMPLE_ZARRAY})
        (output / "notes.txt").write_text("some notes")

        converted = process_zarr_conversions(str(output))

        assert len(converted) == 1
        assert (store / "zarr.json").exists()
        assert (output / "notes.txt").read_text() == "some notes"


# ---------------------------------------------------------------------------
# Integration: zip → extract → convert
# ---------------------------------------------------------------------------

class TestZipIntegration:
    def test_zip_with_v2_zarr_flattened_to_root(self, tmp_path):
        """Zip containing a v2 zarr store → extract → flatten → convert → v3 at root."""
        from processor.extractor import ArchiveExtractor

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        zip_path = input_dir / "dataset.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("sample.zarr/.zgroup", json.dumps({"zarr_format": 2}))
            zf.writestr(
                "sample.zarr/.zattrs",
                json.dumps({"multiscales": [{"version": "0.4"}]}),
            )
            zf.writestr("sample.zarr/0/.zarray", json.dumps(SAMPLE_ZARRAY))

        extractor = ArchiveExtractor(str(input_dir), str(output_dir))
        result_dir, files = extractor.process()

        # zarr.json at output root (flattened from sample.zarr/)
        assert (output_dir / "zarr.json").exists()
        root_meta = json.loads((output_dir / "zarr.json").read_text())
        assert root_meta["zarr_format"] == 3
        assert root_meta["node_type"] == "group"

        assert (output_dir / "0" / "zarr.json").exists()
        arr_meta = json.loads((output_dir / "0" / "zarr.json").read_text())
        assert arr_meta["node_type"] == "array"

        # v2 files removed
        assert not (output_dir / ".zgroup").exists()
        assert not (output_dir / ".zattrs").exists()

        # Original subdirectory removed
        assert not (output_dir / "sample.zarr").exists()

        # Collected files reflect flattened v3 output
        rel_paths = {rel for _, rel in files}
        assert "zarr.json" in rel_paths
        assert os.path.join("0", "zarr.json") in rel_paths
