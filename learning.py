"""Small PyTorch inverse classifier with atomic minibatch-level recovery."""

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import zipfile

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn

from dataset import load_dataset
from storage import atomic_write, canonical_json


class InverseNet(nn.Module):
    """Classify four pattern choices using two hidden layers."""

    def __init__(self):
        """Construct the approved 8-to-32-to-64-to-16 network.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 16))

    def forward(self, inputs):
        """Return four independent categorical-logit vectors for each sample.

        Args:
            inputs (Tensor): Scaled descriptors of shape (batch, 8).
        Returns:
            Tensor: Logits of shape (batch, 4 positions, 4 classes).
        Raises:
            RuntimeError: If input dimensions do not match the model.
        """
        return self.layers(inputs).reshape(-1, 4, 4)


def save_checkpoint(path, state):
    """Atomically persist a safe-loadable training state with integrity checking.

    Args:
        path (Path-like): Destination .pt file.
        state (dict): Tensors and basic Python values describing training.
    Returns:
        None.
    Raises:
        OSError: If the snapshot cannot be committed.
    Notes:
        An outer digest protects the inner serialized state. Both loads use
        weights_only=True, and do not require arbitrary Python classes.
    """
    buffer = io.BytesIO()
    torch.save(state, buffer)
    payload = buffer.getvalue()
    envelope = {"payload": payload, "sha256": hashlib.sha256(payload).hexdigest()}
    atomic_write(path, lambda stream: torch.save(envelope, stream))


def load_checkpoint(path):
    """Load a training snapshot, recovering a structurally damaged primary file.

    Args:
        path (Path-like): Training checkpoint.
    Returns:
        dict: Verified CPU-resident training state.
    Raises:
        ValueError: If an integrity digest fails or no snapshot can be recovered.
        FileNotFoundError: If no checkpoint exists.
    """
    path = Path(path)
    try:
        envelope = torch.load(path, map_location="cpu", weights_only=True)
    except (RuntimeError, EOFError, pickle.UnpicklingError, zipfile.BadZipFile, IndexError) as error:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            raise ValueError("Unreadable training checkpoint without a backup.") from error
        state = load_checkpoint(backup)
        atomic_write(path, lambda stream: stream.write(backup.read_bytes()), rotate_backup=False)
        print("Recovered previous training checkpoint.")
        return state
    if not isinstance(envelope, dict) or "payload" not in envelope or "sha256" not in envelope:
        raise ValueError("Invalid training checkpoint envelope.")
    if hashlib.sha256(envelope["payload"]).hexdigest() != envelope["sha256"]:
        raise ValueError("Training checkpoint integrity mismatch.")
    return torch.load(io.BytesIO(envelope["payload"]), map_location="cpu", weights_only=True)


def metrics(model, x, labels, indices):
    """Compute categorical loss, per-position accuracy, and complete-layout accuracy.

    Args:
        model (InverseNet): Classifier to evaluate.
        x (Tensor): Scaled descriptor matrix on the model device.
        labels (Tensor): Geometry codes with shape (samples, 4).
        indices (list[int]): Disjoint row indices to evaluate.
    Returns:
        dict: Scalar loss, position accuracy, and exact layout accuracy.
    Raises:
        ValueError: If the subset is empty.
    """
    if not len(indices):
        raise ValueError("Cannot evaluate an empty split.")
    model.eval()
    with torch.no_grad():
        logits = model(x[indices])
        y = labels[indices]
        prediction = logits.argmax(dim=-1)
        return {"loss": nn.functional.cross_entropy(logits.reshape(-1, 4), y.reshape(-1)).item(),
                "position_accuracy": (prediction == y).float().mean().item(),
                "exact_accuracy": (prediction == y).all(dim=1).float().mean().item()}


def train(dataset_path, output_dir, options, max_steps=None):
    """Train or resume from the last committed minibatch using stored spectra only.

    Args:
        dataset_path (Path-like): Generated NPZ checkpoint.
        output_dir (Path-like): Training directory, including last.pt and backup.
        options (dict): Seed, batch size, learning rate, epochs, patience, device,
            and explicit synthetic-data permission.
        max_steps (int or None): Optional additional-step limit for recovery tests.
    Returns:
        Path: Authoritative checkpoint containing latest and best model weights.
    Raises:
        ValueError: If data, settings, provenance, or resume identity are invalid.
        RuntimeError: If the requested device is unavailable or loss is nonfinite.
        OSError: If a snapshot cannot be saved.
    Notes:
        CPU continuation is tested for bitwise equality on the same runtime.
        CUDA RNG is saved, but cross-device/version bitwise equality is not
        promised. A hard crash repeats at most one uncommitted minibatch.
    Examples:
        train('runs/data/dataset_checkpoint.npz', 'runs/training', TRAINING)
    """
    options = dict(options)
    if (options["batch_size"] < 1 or options["max_epochs"] < 1 or options["patience"] < 1
            or not np.isfinite(options["learning_rate"]) or options["learning_rate"] <= 0):
        raise ValueError("Invalid training configuration.")
    state_data = load_dataset(dataset_path)
    metadata = json.loads(state_data["metadata"].item())
    if metadata["backend"] == "synthetic" and not options["allow_synthetic"]:
        raise ValueError("Refusing synthetic data. Enable allow_synthetic only for a software demonstration.")
    eligible = np.flatnonzero(state_data["completed"] & (state_data["reason"] == "ok"))
    if len(eligible) < 24:
        raise ValueError(f"Only {len(eligible)} eligible layouts. Need at least 24. Inspect spectra and the selected band.")
    device = torch.device(options["device"])
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise RuntimeError("Requested training device is unavailable.")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(options["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(options["seed"])
    model = InverseNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
    path = Path(output_dir) / "last.pt"
    data_id = state_data["checksum"].item()
    option_id = canonical_json(options)
    if path.exists():
        state = load_checkpoint(path)
        if state["dataset_id"] != data_id:
            raise ValueError("The dataset changed. Use a new training directory.")
        if state["options_json"] != option_id or state["torch_version"] != str(torch.__version__):
            raise ValueError("Training configuration or PyTorch version changed. Use a new training directory.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    else:
        shuffled = np.random.default_rng(options["seed"]).permutation(eligible)
        train_end = int(0.70 * len(shuffled))
        val_end = train_end + max(1, int(0.15 * len(shuffled)))
        splits = {"train": shuffled[:train_end].tolist(), "val": shuffled[train_end:val_end].tolist(), "test": shuffled[val_end:].tolist()}
        rows = state_data["features"][splits["train"]]
        mean = rows.mean(axis=0)
        std = rows.std(axis=0)
        std[std < 1e-8] = 1
        state = {"schema": 1, "dataset_id": data_id, "options_json": option_id, "torch_version": str(torch.__version__),
                 "metadata": metadata, "splits": splits, "mean": torch.tensor(mean, dtype=torch.float32),
                 "std": torch.tensor(std, dtype=torch.float32), "epoch": 0, "step": 0, "cursor": 0,
                 "order": torch.empty(0, dtype=torch.long), "best_loss": float("inf"), "best_model": None,
                 "stale": 0, "history": [], "sum_loss": 0.0, "seen": 0, "finished": False}
    x = ((torch.tensor(state_data["features"], dtype=torch.float32) - state["mean"]) / state["std"]).to(device)
    y = torch.tensor(state_data["codes"], dtype=torch.long, device=device)
    additional_steps = 0
    while not state["finished"] and state["epoch"] < options["max_epochs"]:
        if not len(state["order"]):
            indices = torch.tensor(state["splits"]["train"], dtype=torch.long)
            state["order"] = indices[torch.randperm(len(indices))]
            state["cursor"] = 0
        while state["cursor"] < len(state["order"]):
            batch = state["order"][state["cursor"]:state["cursor"] + options["batch_size"]].to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[batch])
            loss = nn.functional.cross_entropy(logits.reshape(-1, 4), y[batch].reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss became nonfinite. Last committed minibatch remains recoverable.")
            loss.backward()
            optimizer.step()
            state["cursor"] += len(batch)
            state["step"] += 1
            state["sum_loss"] += loss.item() * len(batch)
            state["seen"] += len(batch)
            state["model"] = model.state_dict()
            state["optimizer"] = optimizer.state_dict()
            state["rng"] = torch.get_rng_state()
            state["cuda_rng"] = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
            save_checkpoint(path, state)
            additional_steps += 1
            if max_steps is not None and additional_steps >= max_steps:
                return path
        val = metrics(model, x, y, state["splits"]["val"])
        state["epoch"] += 1
        state["history"].append({"epoch": state["epoch"], "train_loss": state["sum_loss"] / state["seen"], **val})
        if val["loss"] < state["best_loss"]:
            state["best_loss"] = val["loss"]
            state["best_model"] = copy.deepcopy(model.state_dict())
            state["stale"] = 0
        else:
            state["stale"] += 1
        state["order"] = torch.empty(0, dtype=torch.long)
        state["cursor"] = 0
        state["sum_loss"] = 0.0
        state["seen"] = 0
        state["finished"] = state["stale"] >= options["patience"] or state["epoch"] >= options["max_epochs"]
        save_checkpoint(path, state)
        print(f"Epoch {state['epoch']}: validation loss {val['loss']:.5f}, position accuracy {val['position_accuracy']:.3f}")
    if state["best_model"] is None:
        raise ValueError("No validation epoch completed. Resume training before evaluation.")
    print(f"Checkpoint: {path}. Best validation loss: {state['best_loss']:.5f}")
    return path
