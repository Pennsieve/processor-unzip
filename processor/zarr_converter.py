"""
Zarr v2 → v3 metadata converter.

Converts Zarr v2 stores (.zgroup, .zarray, .zattrs) to Zarr v3 format
(zarr.json) without re-encoding chunk data. Blosc-compressed chunks are
binary-compatible between v2 and v3.

Self-contained module — no imports from other processor modules.
"""

import json
import logging
import os
import shutil

log = logging.getLogger(__name__)

# Shuffle mode mapping: v2 integer → v3 string
SHUFFLE_MAP = {0: "noshuffle", 1: "shuffle", 2: "bitshuffle"}

# (dtype kind, itemsize) → v3 data_type name
_DTYPE_KIND_MAP = {
    ("b", 1): "bool",
    ("i", 1): "int8",
    ("i", 2): "int16",
    ("i", 4): "int32",
    ("i", 8): "int64",
    ("u", 1): "uint8",
    ("u", 2): "uint16",
    ("u", 4): "uint32",
    ("u", 8): "uint64",
    ("f", 2): "float16",
    ("f", 4): "float32",
    ("f", 8): "float64",
}


def _is_zarr_root(path: str) -> bool:
    """Check if a directory is the root of a zarr store (v2 or v3)."""
    return os.path.isfile(os.path.join(path, ".zgroup")) or os.path.isfile(os.path.join(path, "zarr.json"))


def flatten_zarr_store(output_dir: str) -> str | None:
    """If a zarr store is nested in a subdirectory, move its contents to output_dir root.

    Handles the common case where a zip extracts to output_dir/name.zarr/.
    The downstream pipeline expects zarr.json at the output root.

    Returns the original subdirectory path if flattened, None otherwise.
    """
    if _is_zarr_root(output_dir):
        return None

    # Find zarr stores anywhere in the tree
    candidates = []
    for root, _dirs, files in os.walk(output_dir):
        if ".zgroup" in files or "zarr.json" in files:
            candidates.append(root)

    if not candidates:
        return None

    # Pick the shallowest store
    candidates.sort(key=lambda p: p.count(os.sep))
    zarr_root = candidates[0]

    log.info(f"Flattening zarr store {zarr_root} → {output_dir}")

    for item in os.listdir(zarr_root):
        shutil.move(os.path.join(zarr_root, item), os.path.join(output_dir, item))

    # Remove the now-empty directory and any empty parents up to (not including) output_dir
    current = zarr_root
    while current != output_dir:
        try:
            os.rmdir(current)
        except OSError:
            break
        current = os.path.dirname(current)

    return zarr_root


def find_zarr_stores(output_dir: str) -> list[str]:
    """Find root-level Zarr v2 stores (directories with .zgroup).

    Returns only top-level store roots, not nested sub-groups.
    """
    candidates = []
    for root, _dirs, files in os.walk(output_dir):
        if ".zgroup" in files:
            candidates.append(root)

    # Keep only roots — filter out any path that is a child of another candidate
    candidates.sort()
    roots = []
    for path in candidates:
        if not any(path.startswith(r + os.sep) for r in roots):
            roots.append(path)
    return roots


def is_zarr_v2(path: str) -> bool:
    """Check if path is a Zarr v2 store (.zgroup with zarr_format 2, no zarr.json)."""
    zgroup_path = os.path.join(path, ".zgroup")
    zarr_json_path = os.path.join(path, "zarr.json")

    if not os.path.isfile(zgroup_path):
        return False
    if os.path.isfile(zarr_json_path):
        return False

    try:
        with open(zgroup_path) as f:
            zgroup = json.load(f)
        return zgroup.get("zarr_format") == 2
    except (json.JSONDecodeError, OSError):
        return False


def _parse_dtype(dtype_str: str) -> tuple[str, int, str]:
    """Parse a NumPy dtype string into (v3_data_type, itemsize, endian).

    Examples:
        "<i2" → ("int16", 2, "little")
        ">f4" → ("float32", 4, "big")
        "|u1" → ("uint8", 1, "little")
    """
    if dtype_str[0] in "<>=|":
        order_char = dtype_str[0]
        type_str = dtype_str[1:]
    else:
        order_char = "="
        type_str = dtype_str

    kind = type_str[0]
    itemsize = int(type_str[1:])

    v3_type = _DTYPE_KIND_MAP.get((kind, itemsize))
    if v3_type is None:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    endian = "big" if order_char == ">" else "little"
    return v3_type, itemsize, endian


def _map_compressor_v2_to_v3(compressor: dict | None, typesize: int) -> list[dict]:
    """Convert v2 compressor config to v3 codec entries.

    Returns 0 or 1 codec dicts for the bytes→bytes compression stage.
    """
    if compressor is None:
        return []

    cid = compressor.get("id", "")

    if cid == "blosc":
        shuffle_int = compressor.get("shuffle", 0)
        shuffle_str = SHUFFLE_MAP.get(shuffle_int, "noshuffle")
        return [
            {
                "name": "blosc",
                "configuration": {
                    "cname": compressor.get("cname", "lz4"),
                    "clevel": compressor.get("clevel", 5),
                    "shuffle": shuffle_str,
                    "typesize": typesize,
                    "blocksize": compressor.get("blocksize", 0),
                },
            }
        ]
    elif cid in ("zlib", "gzip"):
        return [
            {
                "name": "gzip",
                "configuration": {"level": compressor.get("level", 1)},
            }
        ]
    else:
        log.warning(f"Unknown compressor '{cid}', passing through as-is")
        return [{"name": cid, "configuration": {k: v for k, v in compressor.items() if k != "id"}}]


def _build_v3_array_metadata(zarray: dict, zattrs: dict | None = None) -> dict:
    """Build a v3 zarr.json dict from v2 .zarray (and optional .zattrs)."""
    v3_type, itemsize, endian = _parse_dtype(zarray["dtype"])

    codecs = []

    # Fortran order requires a transpose codec
    if zarray.get("order", "C") == "F":
        ndim = len(zarray["shape"])
        codecs.append({"name": "transpose", "configuration": {"order": list(reversed(range(ndim)))}})

    # Array → bytes codec
    codecs.append({"name": "bytes", "configuration": {"endian": endian}})

    # Compression codec(s)
    codecs.extend(_map_compressor_v2_to_v3(zarray.get("compressor"), itemsize))

    # Warn about filters (not mapped)
    filters = zarray.get("filters")
    if filters:
        log.warning(f"v2 filters present and will be skipped: {filters}")

    separator = zarray.get("dimension_separator", ".")
    fill_value = zarray.get("fill_value", 0)
    if fill_value is None:
        fill_value = 0

    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": zarray["shape"],
        "data_type": v3_type,
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": zarray["chunks"]},
        },
        "chunk_key_encoding": {
            "name": "v2",
            "configuration": {"separator": separator},
        },
        "fill_value": fill_value,
        "codecs": codecs,
        "attributes": zattrs if zattrs else {},
    }


def _build_v3_group_metadata(zattrs: dict | None = None) -> dict:
    """Build a v3 zarr.json dict for a group node."""
    return {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": zattrs if zattrs else {},
    }


def _read_json(path: str) -> dict:
    """Read and parse a JSON file."""
    with open(path) as f:
        return json.load(f)


def _write_json(path: str, data: dict) -> None:
    """Write a dict as formatted JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def convert_zarr_v2_to_v3(zarr_path: str) -> None:
    """Convert a Zarr v2 store to v3 format in-place.

    Rewrites metadata files (.zgroup, .zarray, .zattrs → zarr.json)
    without touching chunk data.
    """
    log.info(f"Converting Zarr v2 → v3: {zarr_path}")

    v2_files_to_delete = []

    for root, _dirs, files in os.walk(zarr_path):
        is_array = ".zarray" in files
        is_group = ".zgroup" in files

        if not is_array and not is_group:
            continue

        # Read optional .zattrs
        zattrs = None
        zattrs_path = os.path.join(root, ".zattrs")
        if os.path.isfile(zattrs_path):
            zattrs = _read_json(zattrs_path)
            v2_files_to_delete.append(zattrs_path)

        if is_array:
            zarray_path = os.path.join(root, ".zarray")
            zarray = _read_json(zarray_path)
            v3_meta = _build_v3_array_metadata(zarray, zattrs)
            v2_files_to_delete.append(zarray_path)
        else:
            v3_meta = _build_v3_group_metadata(zattrs)

        if is_group:
            v2_files_to_delete.append(os.path.join(root, ".zgroup"))

        _write_json(os.path.join(root, "zarr.json"), v3_meta)
        log.info(f"  Wrote {os.path.join(root, 'zarr.json')}")

    # Remove old v2 metadata files
    for path in v2_files_to_delete:
        os.remove(path)
        log.debug(f"  Removed {path}")

    # Verify with zarr-python v3
    try:
        import zarr

        zarr.open_group(zarr_path, mode="r")
        log.info(f"  Verified: zarr.open_group succeeded for {zarr_path}")
    except Exception as e:
        log.error(f"  Verification failed for {zarr_path}: {e}")
        raise


def process_zarr_conversions(output_dir: str) -> list[str]:
    """Find and convert all Zarr v2 stores in output_dir.

    Returns list of paths to converted stores.
    """
    stores = find_zarr_stores(output_dir)
    converted = []

    for store_path in stores:
        if is_zarr_v2(store_path):
            convert_zarr_v2_to_v3(store_path)
            converted.append(store_path)
        else:
            log.debug(f"Skipping non-v2 store: {store_path}")

    return converted
