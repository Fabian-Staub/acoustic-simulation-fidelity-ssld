import argparse
import gc
import itertools
import json
import os
import random
import resource
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def log_info(*args, **kwargs) -> None:
    """Write diagnostics and detailed progress to stderr (.err under Slurm)."""
    kwargs.setdefault("file", sys.stderr)
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def log_result(*args, **kwargs) -> None:
    """Write concise training results to stdout (.out under Slurm)."""
    kwargs.setdefault("file", sys.stdout)
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


# ============================================================
# Reproducibility / CPU resources
# ============================================================

def available_cpu_count() -> int:
    """Return CPUs actually available to this Slurm task/process."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")

    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass

    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass

    return max(1, os.cpu_count() or 1)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


# ============================================================
# Dataset split / normalization loading
# ============================================================

def load_distribution(distribution_path: str) -> dict:
    """Load and validate train-scope z-score statistics."""
    distribution_path = os.path.abspath(distribution_path)

    if not os.path.isfile(distribution_path):
        raise FileNotFoundError(
            f"Normalization statistics not found: {distribution_path}"
        )

    with open(distribution_path, "r", encoding="utf-8") as handle:
        distribution = json.load(handle)

    statistics = distribution.get("statistics")
    if not isinstance(statistics, dict):
        raise ValueError("distribution.json is missing a 'statistics' object")

    normalized_statistics: dict[str, tuple[float, float]] = {}
    for feature_name in ("hopiv", "logmel"):
        feature_stats = statistics.get(feature_name)
        if not isinstance(feature_stats, dict):
            raise ValueError(
                f"distribution.json is missing statistics for {feature_name!r}"
            )

        mean = float(feature_stats["mean"])
        stdev = float(feature_stats["stdev"])
        if not torch.isfinite(torch.tensor(mean)):
            raise ValueError(f"Non-finite mean for {feature_name}: {mean}")
        if not torch.isfinite(torch.tensor(stdev)) or stdev <= 0.0:
            raise ValueError(
                f"Invalid standard deviation for {feature_name}: {stdev}"
            )

        normalized_statistics[feature_name] = (mean, stdev)

    if distribution.get("scope") != "train":
        raise ValueError(
            "Normalization statistics must have scope='train' to avoid leakage"
        )

    distribution["normalized_statistics"] = normalized_statistics
    return distribution


def discover_split_shards(dataset_root: str, split: str) -> tuple[str, list[str]]:
    """Return split/pt and its sorted .pt shard names."""
    split = str(split).lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown dataset split: {split!r}")

    pt_folder = os.path.abspath(os.path.join(dataset_root, split, "pt"))
    if not os.path.isdir(pt_folder):
        raise FileNotFoundError(f"Split PT folder not found: {pt_folder}")

    filenames = sorted(
        entry.name
        for entry in os.scandir(pt_folder)
        if entry.is_file() and entry.name.lower().endswith(".pt")
    )
    if not filenames:
        raise RuntimeError(f"No .pt shards found in {pt_folder}")

    return pt_folder, filenames


# ============================================================
# Lightweight performance diagnostics
# ============================================================

def format_bytes(num_bytes: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(num_bytes)

    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0

    return f"{value:.2f} TiB"


def cuda_sync(device: torch.device) -> None:
    """Synchronize only when CUDA timing must be made accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def print_cuda_memory(prefix: str, device: torch.device) -> None:
    if device.type != "cuda":
        return

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak = torch.cuda.max_memory_allocated(device)

    log_info(
        f"{prefix} CUDA memory: "
        f"allocated={format_bytes(allocated)}, "
        f"reserved={format_bytes(reserved)}, "
        f"peak_allocated={format_bytes(peak)}",
        flush=True,
    )


def get_process_rss_bytes() -> int:
    """Return resident memory of the current process."""
    status_path = f"/proc/{os.getpid()}/status"

    try:
        with open(status_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass

    # Linux reports ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def get_system_memory() -> tuple[int | None, int | None]:
    """Return (total_bytes, available_bytes) from /proc/meminfo."""
    values: dict[str, int] = {}

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                key, raw_value = line.split(":", maxsplit=1)
                fields = raw_value.strip().split()

                if fields:
                    values[key] = int(fields[0]) * 1024
    except (OSError, ValueError):
        return None, None

    return values.get("MemTotal"), values.get("MemAvailable")


def read_int_file(path: str) -> int | None:
    try:
        value = Path(path).read_text(encoding="utf-8").strip()
        if value == "max":
            return None
        return int(value)
    except (OSError, ValueError):
        return None


def get_cgroup_memory() -> tuple[int | None, int | None]:
    """Return (job_limit_bytes, job_usage_bytes) for cgroup v2/v1."""
    try:
        lines = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None, None

    # cgroup v2
    if os.path.isfile("/sys/fs/cgroup/cgroup.controllers"):
        for line in lines:
            hierarchy, controllers, relative = line.split(":", maxsplit=2)
            if hierarchy == "0" and controllers == "":
                base = os.path.join("/sys/fs/cgroup", relative.lstrip("/"))
                return (
                    read_int_file(os.path.join(base, "memory.max")),
                    read_int_file(os.path.join(base, "memory.current")),
                )

    # cgroup v1
    for line in lines:
        _hierarchy, controllers, relative = line.split(":", maxsplit=2)
        if "memory" in controllers.split(","):
            base = os.path.join(
                "/sys/fs/cgroup/memory",
                relative.lstrip("/"),
            )
            return (
                read_int_file(os.path.join(base, "memory.limit_in_bytes")),
                read_int_file(os.path.join(base, "memory.usage_in_bytes")),
            )

    return None, None


def print_ram_status(prefix: str) -> None:
    node_total, node_available = get_system_memory()
    rss = get_process_rss_bytes()
    job_limit, job_usage = get_cgroup_memory()

    parts = [f"process_RSS={format_bytes(rss)}"]

    if job_usage is not None:
        parts.append(f"job_usage={format_bytes(job_usage)}")

    if job_limit is not None and job_limit < (1 << 60):
        parts.append(f"job_limit={format_bytes(job_limit)}")
        if job_usage is not None:
            parts.append(
                f"job_remaining={format_bytes(max(0, job_limit - job_usage))}"
            )

    if node_available is not None:
        parts.append(f"node_available={format_bytes(node_available)}")

    if node_total is not None:
        parts.append(f"node_total={format_bytes(node_total)}")

    log_info(f"{prefix} RAM: " + ", ".join(parts), flush=True)


# ============================================================
# Dataset
# ============================================================

class HOARoomDataset(Dataset):
    """Merge one folder-defined split into normalized contiguous RAM tensors."""

    def __init__(
        self,
        pt_folder: str,
        pt_files,
        normalization_statistics: dict[str, tuple[float, float]],
        hoa_order: int,
        dtype: torch.dtype = torch.float32,
        preload_label: str = "dataset",
        feature_layout: str = "hopiv_then_logmel",
    ) -> None:
        super().__init__()

        self.pt_folder = os.path.abspath(pt_folder)
        self.dtype = dtype
        self.preload_label = str(preload_label)
        self.hoa_order = int(hoa_order)
        self.feature_layout = str(feature_layout)

        if self.feature_layout not in {
            "hopiv_then_logmel",
            "logmel_then_hopiv",
        }:
            raise ValueError(
                "feature_layout must be 'hopiv_then_logmel' or "
                "'logmel_then_hopiv'"
            )

        try:
            self.hopiv_mean, self.hopiv_stdev = normalization_statistics["hopiv"]
            self.logmel_mean, self.logmel_stdev = normalization_statistics["logmel"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "normalization_statistics must contain (mean, stdev) pairs "
                "for 'hopiv' and 'logmel'"
            ) from exc

        selected_filenames = [
            os.path.basename(os.fspath(filename))
            for filename in pt_files
        ]
        if not selected_filenames:
            raise ValueError("pt_files must not be empty")
        if len(selected_filenames) != len(set(selected_filenames)):
            raise ValueError("pt_files contains duplicate shard names")

        self.pt_files = [
            os.path.join(self.pt_folder, filename)
            for filename in selected_filenames
        ]
        for path in self.pt_files:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Shard does not exist: {path}")

        self._load_into_contiguous_tensors()

    @staticmethod
    def _validate_shard(data, path: str) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not isinstance(data, (tuple, list))
            or len(data) != 2
            or not all(torch.is_tensor(item) for item in data)
        ):
            raise ValueError(
                f"Expected {path} to contain the tensor tuple (X, accdoa)"
            )

        X_all, accdoa_all = data
        if X_all.ndim != 4:
            raise ValueError(
                f"Expected X shape (N, T, M, C), got {tuple(X_all.shape)}"
            )
        if accdoa_all.ndim != 4 or accdoa_all.shape[-2:] != (3, 3):
            raise ValueError(
                f"Expected accdoa shape (N, T, 3, 3), "
                f"got {tuple(accdoa_all.shape)}"
            )
        if X_all.shape[:2] != accdoa_all.shape[:2]:
            raise ValueError(f"Shard X/target dimensions do not match in {path}")
        return X_all, accdoa_all

    def _normalize_features_(self, X: torch.Tensor) -> None:
        """Apply train-statistics z-score normalization in-place."""
        num_logmel_channels = (self.hoa_order + 1) ** 2
        num_channels = X.shape[-1]

        if num_channels <= num_logmel_channels:
            raise ValueError(
                f"Expected more than {num_logmel_channels} feature channels, "
                f"got {num_channels}"
            )

        if self.feature_layout == "hopiv_then_logmel":
            hopiv = X[..., :-num_logmel_channels]
            logmel = X[..., -num_logmel_channels:]
        else:
            logmel = X[..., :num_logmel_channels]
            hopiv = X[..., num_logmel_channels:]

        hopiv.sub_(self.hopiv_mean).div_(self.hopiv_stdev)
        logmel.sub_(self.logmel_mean).div_(self.logmel_stdev)

    def _load_into_contiguous_tensors(self) -> None:
        total_disk_bytes = sum(os.path.getsize(path) for path in self.pt_files)

        shard_sample_counts: list[int] = []
        feature_shape: tuple[int, ...] | None = None
        target_shape: tuple[int, ...] | None = None

        # First pass obtains exact allocation sizes without retaining raw tensors.
        for path in self.pt_files:
            X_shard, target_shard = self._validate_shard(
                torch.load(path, map_location="cpu", weights_only=True),
                path,
            )
            if feature_shape is None:
                feature_shape = tuple(X_shard.shape[1:])
                target_shape = tuple(target_shard.shape[1:])
            elif tuple(X_shard.shape[1:]) != feature_shape:
                raise ValueError(f"Inconsistent X shape in {os.path.basename(path)}")
            elif tuple(target_shard.shape[1:]) != target_shape:
                raise ValueError(
                    f"Inconsistent target shape in {os.path.basename(path)}"
                )

            shard_sample_counts.append(int(X_shard.shape[0]))
            del X_shard, target_shard

        total_samples = sum(shard_sample_counts)
        if total_samples == 0 or feature_shape is None or target_shape is None:
            raise RuntimeError("No samples found in selected shards")

        preload_start = time.perf_counter()
        log_info(
            f"[RAM MERGE {self.preload_label}] starting {len(self.pt_files)} shards, "
            f"samples={total_samples}, disk_size={format_bytes(total_disk_bytes)}"
        )
        print_ram_status(f"[RAM MERGE {self.preload_label} BEFORE]")

        # Only these normalized features and targets persist in RAM.
        self.X = torch.empty(
            (total_samples, *feature_shape),
            dtype=self.dtype,
        )
        self.target = torch.empty(
            (total_samples, *target_shape),
            dtype=self.dtype,
        )

        offset = 0
        for position, (path, count) in enumerate(
            zip(self.pt_files, shard_sample_counts),
            start=1,
        ):
            shard_start = time.perf_counter()
            X_shard, target_shard = self._validate_shard(
                torch.load(path, map_location="cpu", weights_only=True),
                path,
            )

            # Convert directly into the final dtype, normalize that temporary,
            # and copy only normalized values into the persistent RAM tensor.
            X_normalized = X_shard.to(dtype=self.dtype)
            self._normalize_features_(X_normalized)

            self.X[offset:offset + count].copy_(X_normalized)
            self.target[offset:offset + count].copy_(
                target_shard.to(dtype=self.dtype)
            )
            offset += count

            del X_shard, target_shard, X_normalized

            elapsed = time.perf_counter() - shard_start
            log_info(
                f"[RAM MERGE {self.preload_label} "
                f"{position}/{len(self.pt_files)}] "
                f"{os.path.basename(path)} | normalized+copied={count} | "
                f"load+normalize+copy={elapsed:.3f}s"
            )
            if position == 1 or position == len(self.pt_files) or position % 5 == 0:
                print_ram_status(
                    f"[RAM MERGE {self.preload_label} "
                    f"{position}/{len(self.pt_files)}]"
                )

        if offset != total_samples:
            raise RuntimeError(
                f"RAM merge incomplete: copied {offset} of {total_samples} samples"
            )

        total_seconds = time.perf_counter() - preload_start
        tensor_bytes = self.X.numel() * self.X.element_size()
        tensor_bytes += self.target.numel() * self.target.element_size()
        log_info(
            f"[RAM MERGE {self.preload_label} DONE] samples={total_samples}, "
            f"tensor_size={format_bytes(tensor_bytes)}, time={total_seconds:.2f}s"
        )
        print_ram_status(f"[RAM MERGE {self.preload_label} AFTER]")

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.target[index]


# ============================================================
# Model
# ============================================================

class SDLCRNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 46,
        num_speakers: int = 3,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.num_speakers = num_speakers
        self.output_size = num_speakers * 3

        self.conv1 = nn.Conv2d(
            input_channels,
            128,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(128)
        self.elu1 = nn.ELU()
        self.pool1 = nn.MaxPool2d((1, 2))
        self.drop1 = nn.Dropout2d(dropout_rate)

        self.conv2 = nn.Conv2d(
            128,
            128,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(128)
        self.elu2 = nn.ELU()
        self.pool2 = nn.MaxPool2d((1, 8))
        self.drop2 = nn.Dropout2d(dropout_rate)

        self.conv3 = nn.Conv2d(
            128,
            128,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.elu3 = nn.ELU()
        self.pool3 = nn.MaxPool2d((1, 4))
        self.drop3 = nn.Dropout2d(dropout_rate)

        self.bilstm1 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.bilstm2 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(256, 256)
        self.fc_elu = nn.ELU()
        self.fc_dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected model input shape (B, T, M, C), "
                f"got {tuple(x.shape)}"
            )

        if x.shape[-1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, "
                f"got {x.shape[-1]}"
            )

        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.drop1(self.pool1(self.elu1(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(self.elu2(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(self.elu3(self.bn3(self.conv3(x)))))

        if x.shape[-1] != 2:
            raise ValueError(
                "Unexpected pooled mel dimension. "
                f"Expected 2, got {x.shape[-1]}. "
                "The current model expects 128 input mel bins."
            )

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.flatten(start_dim=2)

        x, _ = self.bilstm1(x)
        x, _ = self.bilstm2(x)

        x = self.fc1(x)
        x = self.fc_elu(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)

        return x


# ============================================================
# Framewise PIT loss
# ============================================================

class FramewisePITLoss(nn.Module):
    def __init__(self, nb_tracks: int = 3) -> None:
        super().__init__()

        self.nb_tracks = int(nb_tracks)

        permutations = list(
            itertools.permutations(range(self.nb_tracks))
        )

        self.register_buffer(
            "permutations",
            torch.tensor(permutations, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if pred.ndim == 3:
            expected_output_size = self.nb_tracks * 3

            if pred.shape[-1] != expected_output_size:
                raise ValueError(
                    f"Expected prediction output size "
                    f"{expected_output_size}, got {pred.shape[-1]}"
                )

            pred = pred.reshape(
                pred.shape[0],
                pred.shape[1],
                self.nb_tracks,
                3,
            )

        if pred.ndim != 4:
            raise ValueError(
                f"Expected pred shape (B, T, tracks, 3), "
                f"got {tuple(pred.shape)}"
            )

        if target.ndim != 4:
            raise ValueError(
                f"Expected target shape (B, T, tracks, 3), "
                f"got {tuple(target.shape)}"
            )

        if pred.shape != target.shape:
            raise ValueError(
                f"pred shape {tuple(pred.shape)} does not match "
                f"target shape {tuple(target.shape)}"
            )

        permutation_losses = []

        for permutation in self.permutations:
            permuted_target = target[:, :, permutation, :]

            loss = (
                pred - permuted_target
            ).pow(2).mean(dim=(-1, -2))

            permutation_losses.append(loss)

        permutation_losses = torch.stack(
            permutation_losses,
            dim=0,
        )

        min_loss = permutation_losses.min(dim=0).values

        return min_loss.mean()


# ============================================================
# Train / validation / test loops
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    use_amp: bool,
    grad_clip=None,
    epoch: int | None = None,
    log_every: int = 20,
):
    model.train()

    total_loss = 0.0
    total_samples = 0

    data_wait_total = 0.0
    transfer_total = 0.0
    forward_loss_total = 0.0
    backward_step_total = 0.0
    epoch_start = time.perf_counter()
    previous_step_end = epoch_start

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    num_batches = len(loader)

    for batch_idx, (X, target) in enumerate(loader, start=1):
        batch_received = time.perf_counter()
        data_wait = batch_received - previous_step_end
        data_wait_total += data_wait

        transfer_start = time.perf_counter()
        X = X.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        cuda_sync(device)
        transfer_time = time.perf_counter() - transfer_start
        transfer_total += transfer_time

        optimizer.zero_grad(set_to_none=True)

        forward_start = time.perf_counter()

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            pred = model(X)
            loss = criterion(pred, target)

        cuda_sync(device)
        forward_loss_time = time.perf_counter() - forward_start
        forward_loss_total += forward_loss_time

        backward_start = time.perf_counter()
        scaler.scale(loss).backward()

        if grad_clip is not None:
            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )

        scaler.step(optimizer)
        scaler.update()
        cuda_sync(device)
        backward_step_time = time.perf_counter() - backward_start
        backward_step_total += backward_step_time

        batch_size = X.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size

        step_time = (
            data_wait
            + transfer_time
            + forward_loss_time
            + backward_step_time
        )
        compute_time = forward_loss_time + backward_step_time
        data_fraction = data_wait / max(step_time, 1e-12)

        should_log = (
            batch_idx == 1
            or batch_idx == num_batches
            or batch_idx % log_every == 0
            or data_wait > max(2.0, compute_time)
        )

        if should_log:
            epoch_label = f" epoch={epoch}" if epoch is not None else ""

            log_info(
                f"[TRAIN{epoch_label} batch={batch_idx}/{num_batches}] "
                f"wait={data_wait:.3f}s | "
                f"H2D={transfer_time:.3f}s | "
                f"fwd+loss={forward_loss_time:.3f}s | "
                f"bwd+step={backward_step_time:.3f}s | "
                f"total={step_time:.3f}s | "
                f"data_wait={100.0 * data_fraction:.1f}% | "
                f"loss={loss.detach().item():.6f}",
                flush=True,
            )

            print_cuda_memory(
                f"[TRAIN batch={batch_idx}]",
                device,
            )
            print_ram_status(
                f"[TRAIN epoch={epoch} batch={batch_idx}/{num_batches}]"
            )

        previous_step_end = time.perf_counter()

    if total_samples == 0:
        raise RuntimeError("Training loader produced no samples")

    epoch_seconds = time.perf_counter() - epoch_start
    measured_total = (
        data_wait_total
        + transfer_total
        + forward_loss_total
        + backward_step_total
    )

    log_info(
        f"[TRAIN SUMMARY epoch={epoch}] "
        f"batches={num_batches} | "
        f"samples={total_samples} | "
        f"wall={epoch_seconds:.2f}s | "
        f"wait={data_wait_total:.2f}s "
        f"({100.0 * data_wait_total / max(measured_total, 1e-12):.1f}%) | "
        f"H2D={transfer_total:.2f}s "
        f"({100.0 * transfer_total / max(measured_total, 1e-12):.1f}%) | "
        f"fwd+loss={forward_loss_total:.2f}s "
        f"({100.0 * forward_loss_total / max(measured_total, 1e-12):.1f}%) | "
        f"bwd+step={backward_step_total:.2f}s "
        f"({100.0 * backward_step_total / max(measured_total, 1e-12):.1f}%) | "
        f"throughput={total_samples / max(epoch_seconds, 1e-12):.1f} samples/s",
        flush=True,
    )

    print_ram_status(f"[TRAIN SUMMARY epoch={epoch}]")
    print_cuda_memory(f"[TRAIN SUMMARY epoch={epoch}]", device)

    return total_loss / total_samples


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    use_amp: bool,
    phase: str = "eval",
    log_every: int = 20,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    data_wait_total = 0.0
    transfer_total = 0.0
    compute_total = 0.0
    phase_start = time.perf_counter()
    previous_step_end = phase_start

    num_batches = len(loader)

    for batch_idx, (X, target) in enumerate(loader, start=1):
        batch_received = time.perf_counter()
        data_wait = batch_received - previous_step_end
        data_wait_total += data_wait

        transfer_start = time.perf_counter()
        X = X.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        cuda_sync(device)
        transfer_time = time.perf_counter() - transfer_start
        transfer_total += transfer_time

        compute_start = time.perf_counter()

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            pred = model(X)
            loss = criterion(pred, target)

        cuda_sync(device)
        compute_time = time.perf_counter() - compute_start
        compute_total += compute_time

        batch_size = X.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size

        if (
            batch_idx == 1
            or batch_idx == num_batches
            or batch_idx % log_every == 0
            or data_wait > max(2.0, compute_time)
        ):
            log_info(
                f"[{phase.upper()} batch={batch_idx}/{num_batches}] "
                f"wait={data_wait:.3f}s | "
                f"H2D={transfer_time:.3f}s | "
                f"forward+loss={compute_time:.3f}s | "
                f"loss={loss.detach().item():.6f}",
                flush=True,
            )
            print_ram_status(
                f"[{phase.upper()} batch={batch_idx}/{num_batches}]"
            )
            print_cuda_memory(
                f"[{phase.upper()} batch={batch_idx}/{num_batches}]",
                device,
            )

        previous_step_end = time.perf_counter()

    if total_samples == 0:
        raise RuntimeError("Evaluation loader produced no samples")

    wall_seconds = time.perf_counter() - phase_start
    measured_total = data_wait_total + transfer_total + compute_total

    log_info(
        f"[{phase.upper()} SUMMARY] "
        f"batches={num_batches} | "
        f"samples={total_samples} | "
        f"wall={wall_seconds:.2f}s | "
        f"wait={data_wait_total:.2f}s "
        f"({100.0 * data_wait_total / max(measured_total, 1e-12):.1f}%) | "
        f"H2D={transfer_total:.2f}s "
        f"({100.0 * transfer_total / max(measured_total, 1e-12):.1f}%) | "
        f"compute={compute_total:.2f}s "
        f"({100.0 * compute_total / max(measured_total, 1e-12):.1f}%) | "
        f"throughput={total_samples / max(wall_seconds, 1e-12):.1f} samples/s",
        flush=True,
    )

    print_ram_status(f"[{phase.upper()} SUMMARY]")
    print_cuda_memory(f"[{phase.upper()} SUMMARY]", device)

    return total_loss / total_samples


# ============================================================
# Main training function
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the SDLCRNN model.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--batch-size", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=2e-4)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--lr-min-delta", type=float, default=2e-4)
    args = parser.parse_args()

    seed = 42
    set_seed(seed)

    dataset_root = args.dataset_root
    distribution_path = os.path.join(dataset_root, "distribution.json")
    checkpoint_path = args.checkpoint_path

    distribution = load_distribution(distribution_path)
    normalization_statistics = distribution["normalized_statistics"]
    hoa_order = int(distribution["hoa_order"])
    n_mels = int(distribution["n_mels"])

    train_pt_folder, train_files = discover_split_shards(dataset_root, "train")
    val_pt_folder, val_files = discover_split_shards(dataset_root, "val")
    test_pt_folder, test_files = discover_split_shards(dataset_root, "test")

    log_info(f"Dataset root:       {os.path.abspath(dataset_root)}")
    log_info(f"Distribution:       {os.path.abspath(distribution_path)}")
    log_info(f"Train PT folder:    {train_pt_folder}")
    log_info(f"Validation folder:  {val_pt_folder}")
    log_info(f"Test PT folder:     {test_pt_folder}")
    log_info(f"Train files:        {len(train_files)}")
    log_info(f"Val files:          {len(val_files)}")
    log_info(f"Test files:         {len(test_files)}")
    log_info(
        "Z-score statistics (train scope): "
        f"hopiv mean={normalization_statistics['hopiv'][0]:.9g}, "
        f"stdev={normalization_statistics['hopiv'][1]:.9g}; "
        f"logmel mean={normalization_statistics['logmel'][0]:.9g}, "
        f"stdev={normalization_statistics['logmel'][1]:.9g}"
    )

    all_shard_paths = [
        *(os.path.join(train_pt_folder, name) for name in train_files),
        *(os.path.join(val_pt_folder, name) for name in val_files),
        *(os.path.join(test_pt_folder, name) for name in test_files),
    ]
    shard_sizes = [os.path.getsize(path) for path in all_shard_paths]
    log_info(
        "Shard sizes: "
        f"min={format_bytes(min(shard_sizes))}, "
        f"mean={format_bytes(sum(shard_sizes) / len(shard_sizes))}, "
        f"max={format_bytes(max(shard_sizes))}, "
        f"total={format_bytes(sum(shard_sizes))}"
    )

    batch_size = args.batch_size
    available_cpus = available_cpu_count()
    torch.set_num_threads(available_cpus)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    # Keep preloaded tensors in the main process. num_workers=0 avoids
    # creating worker-process copies or relying on fork semantics.
    # Two workers overlap CPU batch collation and pinned-memory copies with GPU work.
    # With Linux fork, preloaded tensors remain copy-on-write shared.
    requested_loader_workers = 2
    train_num_workers = min(requested_loader_workers, available_cpus)
    eval_num_workers = min(requested_loader_workers, available_cpus)

    log_info(f"Available CPUs: {available_cpus}")
    log_info(f"Training DataLoader workers: {train_num_workers}")
    log_info(f"Evaluation DataLoader workers: {eval_num_workers}")
    print_ram_status("[STARTUP]")

    max_epochs = 10000

    initial_lr = args.learning_rate
    weight_decay = args.weight_decay

    # Early stopping reacts only to meaningful validation improvements.
    # Training is allowed to run for at least min_epochs before it can stop.
    early_stopping_patience = args.patience
    early_stopping_min_delta = args.min_delta
    min_epochs = args.min_epochs
    gradient_clip = None

    input_channels = 46

    if n_mels != 128:
        raise ValueError(f"The current model expects n_mels=128, got {n_mels}")
    num_speakers = 3

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    log_info(f"Using device: {device}")
    log_info(f"Process PID: {os.getpid()}")
    log_info(f"CPU affinity count: {len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 'unknown'}")
    log_info(f"torch.get_num_threads(): {torch.get_num_threads()}")
    log_info(f"torch.get_num_interop_threads(): {torch.get_num_interop_threads()}")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        log_info(f"GPU: {props.name}")
        log_info(f"GPU memory: {format_bytes(props.total_memory)}")

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    log_info(f"Mixed precision: {use_amp}")

    train_dataset = HOARoomDataset(
        pt_folder=train_pt_folder,
        pt_files=train_files,
        normalization_statistics=normalization_statistics,
        hoa_order=hoa_order,
        preload_label="train",
        feature_layout="hopiv_then_logmel",
    )

    val_dataset = HOARoomDataset(
        pt_folder=val_pt_folder,
        pt_files=val_files,
        normalization_statistics=normalization_statistics,
        hoa_order=hoa_order,
        preload_label="validation",
        feature_layout="hopiv_then_logmel",
    )

    # Do not load the test split into RAM during training.
    # It is constructed only after early stopping / training completion.
    log_info(f"Train samples: {len(train_dataset)}")
    log_info(f"Val samples:   {len(val_dataset)}")
    log_info("Test samples:  deferred until final evaluation")

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader_kwargs = {
        "dataset": train_dataset,
        "batch_size": batch_size,
        "shuffle": True,
        "generator": train_generator,
        "num_workers": train_num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": train_num_workers > 0,
        "drop_last": True,
    }

    if train_num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 2
        train_loader_kwargs["multiprocessing_context"] = "fork"

    train_loader = DataLoader(**train_loader_kwargs)

    val_loader_kwargs = {
        "dataset": val_dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": eval_num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": eval_num_workers > 0,
        "drop_last": False,
    }

    if eval_num_workers > 0:
        val_loader_kwargs["prefetch_factor"] = 2
        val_loader_kwargs["multiprocessing_context"] = "fork"

    val_loader = DataLoader(**val_loader_kwargs)

    model = SDLCRNN(
        input_channels=input_channels,
        num_speakers=num_speakers,
        dropout_rate=0.1,
    ).to(device)

    criterion = FramewisePITLoss(
        nb_tracks=num_speakers,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=initial_lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=args.lr_min_delta,
        threshold_mode="abs",
        min_lr=1e-6,
    )

    # best_val_loss is the exact lowest validation loss and determines
    # which checkpoint is saved. early_stopping_reference_loss changes only
    # after a meaningful improvement of at least early_stopping_min_delta.
    best_val_loss = float("inf")
    early_stopping_reference_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=gradient_clip,
            epoch=epoch,
            log_every=20,
        )

        val_loss = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            phase=f"val epoch={epoch}",
            log_every=20,
        )

        previous_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        log_result(
            f"Epoch {epoch:03d}/{max_epochs} | "
            f"train={train_loss:.6f} | "
            f"val={val_loss:.6f} | "
            f"lr={current_lr:.7f}"
        )


        # Always preserve the numerically best checkpoint, even when the
        # improvement is smaller than the early-stopping min_delta.
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "early_stopping_reference_loss": early_stopping_reference_loss,
                    "epochs_without_improvement": epochs_without_improvement,
                    "input_channels": input_channels,
                    "num_speakers": num_speakers,
                    "train_files": train_files,
                    "val_files": val_files,
                    "test_files": test_files,
                    "dataset_root": os.path.abspath(dataset_root),
                    "distribution_path": os.path.abspath(distribution_path),
                    "normalization_statistics": normalization_statistics,
                    "hoa_order": hoa_order,
                    "n_mels": n_mels,
                    "feature_layout": "hopiv_then_logmel",
                },
                checkpoint_path,
            )

            log_result(f"Saved new best model to {checkpoint_path}")

        # Reset early stopping only for a meaningful improvement. Tiny changes
        # within min_delta are treated as validation noise.
        if val_loss < early_stopping_reference_loss - early_stopping_min_delta:
            early_stopping_reference_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epoch >= min_epochs
            and epochs_without_improvement >= early_stopping_patience
        ):
            log_result(
                "Early stopping: validation loss did not improve by at least "
                f"{early_stopping_min_delta:.1e} for "
                f"{early_stopping_patience} consecutive epochs "
                f"after minimum epoch {min_epochs}."
            )
            break

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"No best-model checkpoint was created: {checkpoint_path}"
        )

    # Training is over. Release objects that are no longer needed before
    # loading the test split into RAM.
    del train_loader
    del val_loader
    del train_dataset
    del val_dataset
    del optimizer
    del scheduler
    del scaler
    del train_generator

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print_ram_status("[BEFORE LAZY TEST LOAD]")
    print_cuda_memory("[BEFORE LAZY TEST LOAD]", device)

    test_dataset = HOARoomDataset(
        pt_folder=test_pt_folder,
        pt_files=test_files,
        normalization_statistics=normalization_statistics,
        hoa_order=hoa_order,
        preload_label="test",
        feature_layout="hopiv_then_logmel",
    )

    log_info(f"Test samples:  {len(test_dataset)}")

    test_loader_kwargs = {
        "dataset": test_dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": eval_num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": False,
        "drop_last": False,
    }

    if eval_num_workers > 0:
        test_loader_kwargs["prefetch_factor"] = 2
        test_loader_kwargs["multiprocessing_context"] = "fork"

    test_loader = DataLoader(**test_loader_kwargs)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    best_epoch = checkpoint["epoch"]
    checkpoint_best_val_loss = checkpoint["best_val_loss"]
    del checkpoint
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    log_result(
        f"Loaded best model from epoch {best_epoch} "
        f"with validation loss "
        f"{checkpoint_best_val_loss:.6f}"
    )

    test_loss = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        phase="test",
        log_every=20,
    )

    log_result(f"Final test PIT loss: {test_loss:.6f}")


if __name__ == "__main__":
    main()
