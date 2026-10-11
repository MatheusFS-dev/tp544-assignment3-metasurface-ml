"""Paper-scale restricted inverse network."""

import torch
from torch import nn


class RestrictedNet(nn.Module):
    """Approximate the reported 24-to-48 sigmoid restricted architecture."""

    def __init__(self, dropout=0.5):
        """Build reported layer widths with an explicit assumed dropout rate.

        Args:
            dropout (float): Dropout probability omitted by the paper.
        """
        super().__init__()
        layers = []
        for index, (inputs, outputs) in enumerate(((24, 24), (24, 500), (500, 500),
                                                    (500, 500), (500, 500))):
            layers.extend((nn.Linear(inputs, outputs), nn.ReLU()))
            if index < 4:
                layers.append(nn.Dropout(dropout))
        layers.extend((nn.Linear(500, 48), nn.Sigmoid()))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        """Predict sixteen three-bit pattern codes.

        Args:
            inputs (Tensor): Descriptors with shape (batch, 24).

        Returns:
            Tensor: Sigmoid outputs with shape (batch, 48).
        """
        return self.layers(inputs)


def encode_bits(codes):
    """Convert 16 tile IDs to 48 binary target values.

    Args:
        codes (array-like): Shape (batch, 16) integer tile IDs.

    Returns:
        Tensor: Float targets with shape (batch, 48).
    """
    values = torch.as_tensor(codes, dtype=torch.long)
    return ((values.unsqueeze(-1) >> torch.tensor([2, 1, 0])) & 1).reshape(-1, 48).float()


def decode_bits(outputs):
    """Threshold outputs and decode sixteen tile IDs.

    Args:
        outputs (Tensor): Sigmoid values with shape (batch, 48).

    Returns:
        Tensor: Integer tile IDs with shape (batch, 16).
    """
    bits = (outputs.reshape(-1, 16, 3) >= 0.5).long()
    return (bits * torch.tensor([4, 2, 1], device=outputs.device)).sum(dim=-1)


def train(dataset_path, output_dir, options, max_steps=None):
    """Train or resume the restricted model from saved paper-scale spectra.

    Args:
        dataset_path (Path-like): Prepared paper dataset.
        output_dir (Path-like): Independent paper training directory.
        options (dict): Seed, device, epochs, rate, batch, and fixture permission.
        max_steps (int or None): Stop after additional minibatches for tests.

    Returns:
        Path: Last committed training checkpoint.

    Raises:
        ValueError: If dataset, options, or resume identity are invalid.
        RuntimeError: If the requested device is unavailable.
    """
    import json
    import time
    from pathlib import Path

    import numpy as np

    from metasurface.common.progress import Progress
    from metasurface.common.storage import canonical_json
    from metasurface.paper.dataset import load_dataset
    from metasurface.reduced.learning import load_checkpoint, save_checkpoint

    options = dict(options)
    if (options["batch_size"] < 1 or options["max_epochs"] < 1 or
            options["learning_rate"] <= 0 or not 0 <= options["dropout"] < 1):
        raise ValueError("Invalid paper training options.")
    data = load_dataset(dataset_path)
    metadata = json.loads(data["metadata"].item())
    if metadata["backend"] == "test_fixture" and not options.get("allow_test_fixture", False):
        raise ValueError("Paper training refuses test fixture data by default.")
    eligible = np.flatnonzero(data["completed"] & (data["reason"] == "ok"))
    if len(eligible) < 24:
        raise ValueError(f"Only {len(eligible)} eligible paper layouts. Need at least 24.")
    device = torch.device(options["device"])
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise RuntimeError("Requested paper training device is unavailable.")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(options["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(options["seed"])
    model = RestrictedNet(options["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
    path = Path(output_dir) / "last.pt"
    option_id = canonical_json(options)
    if path.exists():
        state = load_checkpoint(path)
        if (state["dataset_id"] != data["checksum"].item() or state["options_json"] != option_id
                or state["torch_version"] != str(torch.__version__)):
            raise ValueError("Paper training identity changed. Use a new directory.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    else:
        shuffled = np.random.default_rng(options["seed"]).permutation(eligible)
        cutoff = int(0.70 * len(shuffled))
        train_ids, test_ids = shuffled[:cutoff].tolist(), shuffled[cutoff:].tolist()
        rows = data["features"][train_ids]
        mean, std = rows.mean(axis=0), rows.std(axis=0)
        std[std < 1e-8] = 1
        state = {"schema": 1, "dataset_id": data["checksum"].item(), "options_json": option_id,
                 "torch_version": str(torch.__version__), "metadata": metadata,
                 "splits": {"train": train_ids, "test": test_ids},
                 "mean": torch.tensor(mean, dtype=torch.float32),
                 "std": torch.tensor(std, dtype=torch.float32), "epoch": 0, "step": 0,
                 "cursor": 0, "order": torch.empty(0, dtype=torch.long), "history": [],
                 "batch_history": [],
                 "sum_loss": 0.0, "seen": 0, "finished": False}
    state.setdefault("batch_history", [])
    x = ((torch.tensor(data["features"], dtype=torch.float32) - state["mean"]) / state["std"]).to(device)
    y = encode_bits(data["codes"]).to(device)
    additional_steps = 0
    print(f"Paper training: {len(eligible)} eligible, epoch {state['epoch'] + 1}, {state['step']} saved minibatches", flush=True)
    while not state["finished"] and state["epoch"] < options["max_epochs"]:
        if not len(state["order"]):
            ids = torch.tensor(state["splits"]["train"], dtype=torch.long)
            state["order"] = ids[torch.randperm(len(ids))]
            state["cursor"] = 0
        progress = Progress(f"Paper epoch {state['epoch'] + 1}", len(state["order"]), state["cursor"])
        while state["cursor"] < len(state["order"]):
            batch = state["order"][state["cursor"]:state["cursor"] + options["batch_size"]].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_started = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x[batch]), y[batch])
            if not torch.isfinite(loss):
                raise RuntimeError("Paper training loss became nonfinite.")
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            compute_seconds = time.perf_counter() - batch_started
            state["cursor"] += len(batch)
            state["step"] += 1
            state["sum_loss"] += loss.item() * len(batch)
            state["seen"] += len(batch)
            state["batch_history"].append({"epoch": state["epoch"] + 1, "step": state["step"],
                                           "cursor": state["cursor"], "batch_size": len(batch),
                                           "loss": float(loss.item()), "compute_seconds": compute_seconds})
            state["model"] = model.state_dict()
            state["optimizer"] = optimizer.state_dict()
            state["rng"] = torch.get_rng_state()
            state["cuda_rng"] = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
            save_checkpoint(path, state)
            progress.committed(state["cursor"], f"step {state['step']}")
            additional_steps += 1
            if max_steps is not None and additional_steps >= max_steps:
                return path
        model.eval()
        with torch.no_grad():
            train_ids = state["splits"]["train"]
            output = model(x[train_ids])
            target = y[train_ids]
            predicted_codes = decode_bits(output)
            target_codes = decode_bits(target)
            match = predicted_codes == target_codes
            train_eval_loss = nn.functional.mse_loss(output, target).item()
            train_bit_accuracy = ((output >= 0.5) == (target >= 0.5)).float().mean().item()
            train_tile_accuracy = match.float().mean().item()
            train_exact_accuracy = match.all(dim=1).float().mean().item()
        state["epoch"] += 1
        record = {"epoch": state["epoch"], "train_loss": state["sum_loss"] / state["seen"],
                  "train_eval_loss": train_eval_loss, "train_bit_accuracy": train_bit_accuracy,
                  "train_tile_accuracy": train_tile_accuracy, "train_exact_accuracy": train_exact_accuracy}
        state["history"].append(record)
        state["order"] = torch.empty(0, dtype=torch.long)
        state["cursor"] = 0
        state["sum_loss"] = 0.0
        state["seen"] = 0
        state["finished"] = state["epoch"] >= options["max_epochs"]
        save_checkpoint(path, state)
        print(f"Paper epoch {state['epoch']}: train_loss={record['train_loss']:.5f} "
              f"train_bit_accuracy={train_bit_accuracy:.3f} "
              f"train_tile_accuracy={train_tile_accuracy:.3f} "
              f"train_exact_accuracy={train_exact_accuracy:.3f}", flush=True)
    return path
