"""Regressions reproducing consequential findings from independent review."""

import numpy as np
import pytest


@pytest.mark.parametrize("kind", ["npz", "pt"])
def test_recovery_preserves_backup_for_a_second_crash(tmp_path, kind):
    """Require the good backup to survive repeated primary corruption.

    Args:
        tmp_path (Path): Isolated test folder.
        kind (str): Checkpoint format.
    Returns:
        None.
    Raises:
        AssertionError: If recovery destroys the known-good snapshot.
    """
    from storage import save_npz, load_npz
    from learning import save_checkpoint, load_checkpoint
    save, load = (save_npz, load_npz) if kind == "npz" else (save_checkpoint, load_checkpoint)
    path = tmp_path / ("checkpoint." + kind)
    save(path, {"value": np.asarray(1)} if kind == "npz" else {"value": 1})
    save(path, {"value": np.asarray(2)} if kind == "npz" else {"value": 2})
    backup = path.with_suffix(path.suffix + ".bak")
    expected = backup.read_bytes()
    for _ in range(2):
        path.write_bytes(b"crash")
        assert int(load(path)["value"]) == 1
        assert backup.read_bytes() == expected


def test_real_lumapi_vector_fields_on_nonuniform_mesh():
    """Exercise the documented vector E result and spatial quadrature.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: If component extraction or quadrature is wrong.
    """
    from lumerical import plane_field
    x, y = np.array([0., 0.2, 1.]), np.array([0., 0.7, 2.])
    electric = np.zeros((3, 3, 1, 10, 3), dtype=complex)
    electric[:, :, 0, :, 0] = (x[:, None, None] + 2*y[None, :, None]) * (1 + 2j)
    electric[..., 1] = 3 - 4j
    result = {"x": x, "y": y, "z": [0.], "f": np.arange(10), "E": electric, "Lumerical_dataset": {}}
    assert np.allclose(plane_field(result, "Ex"), 2.5 * (1 + 2j))
    assert np.allclose(plane_field(result, "Ey"), 3 - 4j)


def test_crash_during_recovery_keeps_backup(tmp_path, monkeypatch):
    """Inject failure just before installing the recovered primary checkpoint.

    Args:
        tmp_path (Path): Isolated folder.
        monkeypatch (object): Pytest patch manager.
    Returns:
        None.
    Raises:
        AssertionError: If the backup is lost during recovery.
    """
    import storage
    path = tmp_path / "recovery.npz"
    storage.save_npz(path, {"value": np.asarray(1)})
    storage.save_npz(path, {"value": np.asarray(2)})
    backup = path.with_suffix(".npz.bak")
    good = backup.read_bytes()
    path.write_bytes(b"crash")
    original_replace = storage.os.replace
    def interrupted_replace(source, destination):
        """Fail primary replacement while allowing any unrelated operation.

        Args:
            source (Path-like): Temporary file.
            destination (Path-like): Target file.
        Returns:
            None.
        Raises:
            OSError: At primary restoration.
        """
        if destination == path:
            raise OSError("Injected restoration crash")
        original_replace(source, destination)
    monkeypatch.setattr(storage.os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="Injected"):
        storage.load_npz(path)
    assert backup.read_bytes() == good
    monkeypatch.setattr(storage.os, "replace", original_replace)
    assert int(storage.load_npz(path)["value"]) == 1


def test_cuda_default_does_not_silently_use_cpu(tmp_path, monkeypatch):
    """Reject unavailable CUDA despite a usable generated test dataset.

    Args:
        tmp_path (Path): Isolated folder.
        monkeypatch (object): Pytest patch manager.
    Returns:
        None.
    Raises:
        AssertionError: If CUDA default silently falls back to CPU.
    """
    import config
    import torch
    from dataset import generate
    from learning import train
    assert config.TRAINING["device"] == "cuda"
    data = generate(tmp_path / "data", config.PHYSICS, "synthetic", limit=32)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="device is unavailable"):
        train(data, tmp_path / "train", {**config.TRAINING, "allow_synthetic": True})
