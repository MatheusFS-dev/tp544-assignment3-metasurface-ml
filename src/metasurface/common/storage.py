"""Atomic, fsync-backed checkpoints with one previous committed snapshot."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np


def canonical_json(value):
    """Serialize configuration deterministically and reject nonfinite numbers.

    Args:
        value (object): JSON-compatible configuration or metadata.
    Returns:
        str: Sorted, compact JSON.
    Raises:
        ValueError: If a number is not finite.
        TypeError: If a value cannot be serialized.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def array_digest(arrays):
    """Fingerprint arrays, including their names, shapes, types, and contents.

    Args:
        arrays (dict): Named NumPy arrays, optionally including checksum.
    Returns:
        str: SHA-256 digest excluding the checksum field itself.
    Raises:
        TypeError: If an array contains Python objects.
    """
    digest = hashlib.sha256()
    for name in sorted(k for k in arrays if k != "checksum"):
        value = np.ascontiguousarray(arrays[name])
        if value.dtype.hasobject:
            raise TypeError("Object arrays are forbidden in checkpoints.")
        digest.update(canonical_json([name, value.shape, value.dtype.str]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def atomic_write(path, writer, rotate_backup=True):
    """Commit a file atomically while retaining its previous complete snapshot.

    Args:
        path (Path-like): Destination file.
        writer (callable): Function receiving a writable binary file object.
        rotate_backup (bool): Preserve the existing backup during recovery if false.
    Returns:
        None.
    Raises:
        OSError: If writing, syncing, or replacement fails.
        Exception: Any exception raised by writer.
    Notes:
        A crash can lose the active operation, but not the previous snapshot.
        Run only one writer for each experiment directory at a time.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() and rotate_backup:
            backup_temp = Path(temporary + ".backup")
            shutil.copyfile(path, backup_temp)
            with backup_temp.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(backup_temp, path.with_suffix(path.suffix + ".bak"))
        os.replace(temporary, path)
        if os.name == "posix":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)
        Path(temporary + ".backup").unlink(missing_ok=True)


def save_npz(path, arrays):
    """Save named arrays with an embedded integrity fingerprint.

    Args:
        path (Path-like): Destination checkpoint.
        arrays (dict): NumPy-compatible values to persist.
    Returns:
        None.
    Raises:
        OSError: If the checkpoint cannot be committed.
        TypeError: If an object array is present.
    """
    values = {k: np.asarray(v) for k, v in arrays.items() if k != "checksum"}
    values["checksum"] = np.asarray(array_digest(values))
    atomic_write(path, lambda stream: np.savez_compressed(stream, **values))


def load_npz(path):
    """Read a checkpoint and recover a structurally interrupted file if possible.

    Args:
        path (Path-like): Checkpoint path.
    Returns:
        dict: Copied arrays with a verified checksum.
    Raises:
        ValueError: If content integrity fails or no valid snapshot is readable.
        FileNotFoundError: If no checkpoint exists.
    Notes:
        Semantic tampering is rejected rather than silently rolling back.
    """
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            result = {k: archive[k].copy() for k in archive.files}
    except (ValueError, EOFError, zipfile.BadZipFile) as error:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            raise ValueError("Unreadable checkpoint without a backup.") from error
        result = load_npz(backup)
        # Restore the last good snapshot before a later write rotates backups.
        atomic_write(path, lambda stream: stream.write(backup.read_bytes()), rotate_backup=False)
        print(f"Recovered previous checkpoint: {path.name}")
    if "checksum" not in result or result["checksum"].item() != array_digest(result):
        raise ValueError("Checkpoint integrity fingerprint mismatch.")
    return result
