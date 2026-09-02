#!/usr/bin/env python3
"""
Framewise test-set evaluation for the SDLCRNN model.

This script:
1. Loads best_model.pth first.
2. Imports SDLCRNN and HOARoomDataset from train.py.
3. Reads dataset_root from the checkpoint and evaluates its test/pt split.
4. Evaluates every time frame independently.
5. Reports the paper's framewise SDL metrics only:
   - localization error (LE), without an angular threshold
   - localization recall (LR), without an angular threshold
   - location-sensitive F-score at 15 degrees (F15)
   - location-sensitive error rate at 15 degrees (ER15)
   - aggregate SDL score (ESDL)
6. Saves one uncompressed CSV containing every original metadata row and
   corresponding model prediction from all evaluated datasets.

Performance notes:
    Metric calculations are vectorized across each batch. Exact small-track
    assignments are solved in grouped activity-mask batches, avoiding a SciPy
    call and duplicate metric calculation for every frame. Use
    --skip-prediction-csv when only aggregate metrics are needed.

Example:
    python eval_5_fast.py \
        --checkpoint best_model.pth \
        --train-module train \
        --arni-dataset /path/to/arni-dataset \
        --motus-dataset /path/to/motus-dataset \
        --ivan-dataset /path/to/ivan-dataset
"""

from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


EPS = 1e-10
SPATIAL_THRESHOLD_DEG = 15.0


@dataclass
class MetricAccumulator:
    num_frames: int = 0
    num_reference_events: int = 0
    num_prediction_events: int = 0

    tp: int = 0
    fp: int = 0
    fn: int = 0

    tp_spatial: int = 0
    fp_spatial: int = 0

    localization_error_sum: float = 0.0
    localization_matches: int = 0

    exact_count_frames: int = 0
    count_abs_error_sum: int = 0


@dataclass
class MetadataSampleGroup:
    """All metadata rows belonging to one tensor sample."""

    csv_path: Path
    key: tuple[str, ...]
    rows: list[dict[str, str]]


@dataclass(frozen=True)
class MetadataLayout:
    """Resolved metadata columns used for grouping and alignment checks."""

    frame: str
    slot: str
    sample_key: tuple[str, ...]
    accdoa_x: str
    accdoa_y: str
    accdoa_z: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one checkpoint on the checkpoint dataset and up to three "
            "additional dataset test splits."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("best_model.pth"),
        help="Path to the saved best-model checkpoint.",
    )
    parser.add_argument(
        "--train-module",
        default="train_model",
        help=(
            "Python module containing SDLCRNN and HOARoomDataset. "
            "For train.py, use: train"
        ),
    )
    parser.add_argument(
        "--arni-dataset",
        "--eval-dataset-root-1",
        dest="arni_dataset",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Root of the arni-dataset evaluation set; evaluates <PATH>/test/pt. "
            "The legacy alias --eval-dataset-root-1 is also accepted."
        ),
    )
    parser.add_argument(
        "--motus-dataset",
        "--eval-dataset-root-2",
        dest="motus_dataset",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Root of the motus_dataset evaluation set; evaluates <PATH>/test/pt. "
            "The legacy alias --eval-dataset-root-2 is also accepted."
        ),
    )
    parser.add_argument(
        "--ivan-dataset",
        "--eval-dataset-root-3",
        dest="ivan_dataset",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Root of the ivan_dataset evaluation set; evaluates <PATH>/test/pt. "
            "The legacy alias --eval-dataset-root-3 is also accepted."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--activity-threshold",
        type=float,
        default=0.5,
        help="Predicted ACCDOA norm required for an active source.",
    )
    parser.add_argument(
        "--reference-activity-threshold",
        type=float,
        default=1e-6,
        help="Reference ACCDOA norm required for an active source.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results"),
        help=(
            "Directory for the combined JSON results file and the combined "
            "augmented predictions CSV (.csv)."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--skip-prediction-csv",
        action="store_true",
        help="Compute metrics without writing the large per-frame prediction CSV.",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")

    return torch.device(name)


def import_training_objects(module_name: str):
    """
    Import the model and dataset definitions from train.py.

    The supplied training script uses the class name SDLCRNN. If your local
    file instead uses SDL_CRNN, this function accepts that name as a fallback.
    """
    module = importlib.import_module(module_name)

    model_class = getattr(module, "SDLCRNN", None)
    if model_class is None:
        model_class = getattr(module, "SDL_CRNN", None)

    if model_class is None:
        raise ImportError(
            f"{module_name}.py must define SDLCRNN or SDL_CRNN."
        )

    dataset_class = getattr(module, "HOARoomDataset", None)
    if dataset_class is None:
        raise ImportError(
            f"{module_name}.py must define HOARoomDataset."
        )

    return model_class, dataset_class


def resolve_checkpoint_path(checkpoint_argument: Path) -> Path:
    """Accept either a checkpoint file or a directory containing best_model.pth."""
    checkpoint_argument = checkpoint_argument.expanduser().resolve()
    if checkpoint_argument.is_dir():
        return checkpoint_argument / "best_model.pth"
    return checkpoint_argument


def checkpoint_title(checkpoint_argument: Path, checkpoint_path: Path) -> str:
    """Create a filesystem-safe title for the combined JSON output."""
    if checkpoint_argument.expanduser().resolve().is_dir():
        raw_title = checkpoint_argument.expanduser().resolve().name
    elif checkpoint_path.name == "best_model.pth":
        raw_title = checkpoint_path.parent.name
    else:
        raw_title = checkpoint_path.stem

    safe_title = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in raw_title
    )
    return safe_title or "checkpoint"


def load_checkpoint_first(
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """
    Load best_model.pth before constructing the model or test dataset.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected best_model.pth to contain a checkpoint dictionary."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    if "dataset_root" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'dataset_root'. "
            "The training script must save the dataset root."
        )

    return checkpoint


def normalize_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Remove a DataParallel/DistributedDataParallel 'module.' prefix if present.
    """
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }




def load_training_distribution(checkpoint: dict[str, Any]) -> tuple[Path, dict[str, tuple[float, float]], int]:
    """Load normalization statistics only from the checkpoint dataset root."""
    checkpoint_dataset_root = Path(checkpoint["dataset_root"]).expanduser().resolve()
    distribution_path = checkpoint_dataset_root / "distribution.json"

    if not distribution_path.is_file():
        raise FileNotFoundError(
            "distribution.json from the checkpoint dataset root was not found: "
            f"{distribution_path}"
        )

    with distribution_path.open("r", encoding="utf-8") as handle:
        distribution = json.load(handle)

    statistics = distribution.get("statistics")
    if not isinstance(statistics, dict):
        raise ValueError(f"Missing 'statistics' in {distribution_path}")

    normalization_statistics: dict[str, tuple[float, float]] = {}
    for feature_name in ("hopiv", "logmel"):
        feature_stats = statistics.get(feature_name)
        if not isinstance(feature_stats, dict):
            raise ValueError(
                f"Missing statistics for {feature_name!r} in {distribution_path}"
            )
        normalization_statistics[feature_name] = (
            float(feature_stats["mean"]),
            float(feature_stats["stdev"]),
        )

    if distribution.get("scope") != "train":
        raise ValueError(
            f"Expected scope='train' in {distribution_path} to avoid leakage."
        )

    hoa_order = int(distribution["hoa_order"])
    return distribution_path, normalization_statistics, hoa_order


def discover_test_shards(dataset_root: Path) -> tuple[Path, list[str]]:
    """Discover all .pt shards in <dataset_root>/test/pt."""
    dataset_root = dataset_root.expanduser().resolve()
    pt_folder = dataset_root / "test" / "pt"

    if not pt_folder.is_dir():
        raise FileNotFoundError(f"Test PT folder not found: {pt_folder}")

    test_files = sorted(
        path.name for path in pt_folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".pt"
    )
    if not test_files:
        raise RuntimeError(f"No .pt test shards found in {pt_folder}")

    return pt_folder, test_files


PREDICTION_COLUMNS = (
    "prediction_dataset",
    "prediction_source_csv",
    "prediction_accdoa_x",
    "prediction_accdoa_y",
    "prediction_accdoa_z",
    "prediction_activity",
    "prediction_active",
    "prediction_az_deg",
    "prediction_el_deg",
)


def discover_metadata_csvs(
    pt_folder: Path,
    test_files: list[str],
) -> tuple[list[Path], list[str]]:
    """Match every PT shard to <split>/csv/<same-stem>.csv."""
    csv_folder = pt_folder.parent / "csv"
    if not csv_folder.is_dir():
        raise FileNotFoundError(
            f"Metadata CSV folder not found next to {pt_folder}: {csv_folder}"
        )

    csv_paths = [csv_folder / f"{Path(name).stem}.csv" for name in test_files]
    missing = [path for path in csv_paths if not path.is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(
            f"Missing metadata CSV for {len(missing)} PT shard(s): {preview}"
        )

    fieldnames: list[str] | None = None
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            current = csv.DictReader(handle).fieldnames
        if not current:
            raise ValueError(f"Metadata CSV has no header: {csv_path}")
        if fieldnames is None:
            fieldnames = list(current)
        elif list(current) != fieldnames:
            raise ValueError(
                "All metadata CSV files must use the same columns and order; "
                f"header differs in {csv_path}"
            )

    assert fieldnames is not None
    collisions = sorted(set(fieldnames).intersection(PREDICTION_COLUMNS))
    if collisions:
        raise ValueError(
            "Metadata already contains reserved prediction column(s): "
            + ", ".join(collisions)
        )
    return csv_paths, fieldnames


def union_fieldnames(fieldname_lists: list[list[str]]) -> list[str]:
    """Return an ordered union suitable for one heterogeneous output CSV."""
    union: list[str] = []
    seen: set[str] = set()
    for fieldnames in fieldname_lists:
        for fieldname in fieldnames:
            if fieldname not in seen:
                seen.add(fieldname)
                union.append(fieldname)
    return union


def metadata_index_column(fieldnames: list[str], wanted: str) -> str:
    """Find a column while tolerating escaped Markdown-style headers."""
    for fieldname in fieldnames:
        normalized = fieldname.replace("\\", "").strip().strip("*").lower()
        if normalized == wanted:
            return fieldname
    raise ValueError(
        f"Metadata CSV must contain a {wanted!r} column; found {fieldnames}"
    )


def resolve_metadata_layout(fieldnames: list[str]) -> MetadataLayout:
    """Resolve sample identity, frame/slot, and reference ACCDOA columns."""
    normalized_to_original = {
        name.replace("\\", "").strip().strip("*").lower(): name
        for name in fieldnames
    }

    # base_name identifies the exported segment in the supplied datasets.  Add
    # sample_id and segment_idx when present to make accidental collisions even
    # less likely.  For older exports, the pair is sufficient on its own.
    available_identity_columns = tuple(
        normalized_to_original[name]
        for name in ("sample_id", "segment_idx", "base_name")
        if name in normalized_to_original
    )
    normalized_identity_names = {
        name.replace("\\", "").strip().strip("*").lower()
        for name in available_identity_columns
    }
    if not (
        "base_name" in normalized_identity_names
        or {"sample_id", "segment_idx"}.issubset(normalized_identity_names)
    ):
        raise ValueError(
            "Metadata CSV must contain base_name or both sample_id and "
            f"segment_idx; found {fieldnames}"
        )

    return MetadataLayout(
        frame=metadata_index_column(fieldnames, "frame_idx"),
        slot=metadata_index_column(fieldnames, "slot"),
        sample_key=available_identity_columns,
        accdoa_x=metadata_index_column(fieldnames, "accdoa_x"),
        accdoa_y=metadata_index_column(fieldnames, "accdoa_y"),
        accdoa_z=metadata_index_column(fieldnames, "accdoa_z"),
    )


def iter_metadata_sample_groups(
    csv_paths: list[Path],
    layout: MetadataLayout,
):
    """
    Stream one sample group at a time in PT-shard/CSV-row order.

    HOARoomDataset concatenates samples shard by shard, preserving the supplied
    sorted PT-file order.  The matching CSV exports use that same order.  A
    second target-value check in write_sample_predictions guards against any
    unexpected ordering difference.
    """
    for csv_path in csv_paths:
        completed_keys: set[tuple[str, ...]] = set()
        current_key: tuple[str, ...] | None = None
        current_rows: list[dict[str, str]] = []

        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=2):
                key = tuple((row.get(column) or "").strip() for column in layout.sample_key)
                if any(not value for value in key):
                    raise ValueError(
                        f"Empty sample identity in {csv_path} row {row_number}: {key}"
                    )

                if current_key is None:
                    current_key = key
                elif key != current_key:
                    completed_keys.add(current_key)
                    if key in completed_keys:
                        raise ValueError(
                            f"Metadata sample {key} is not contiguous in {csv_path}"
                        )
                    yield MetadataSampleGroup(
                        csv_path=csv_path,
                        key=current_key,
                        rows=current_rows,
                    )
                    current_key = key
                    current_rows = []

                current_rows.append(row)

        if current_key is not None:
            yield MetadataSampleGroup(
                csv_path=csv_path,
                key=current_key,
                rows=current_rows,
            )


def write_sample_predictions(
    writer: csv.DictWriter,
    metadata_group: MetadataSampleGroup,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    activity_threshold: float,
    layout: MetadataLayout,
    prediction_dataset: str,
) -> None:
    """Append one sample's metadata rows with frame/slot-matched predictions."""
    num_frames, num_slots, coordinates = predictions.shape
    if coordinates != 3:
        raise ValueError(
            f"Expected prediction shape (T, tracks, 3), got {tuple(predictions.shape)}"
        )
    if tuple(targets.shape) != tuple(predictions.shape):
        raise ValueError(
            f"Target shape {tuple(targets.shape)} does not match prediction shape "
            f"{tuple(predictions.shape)}"
        )

    prediction_array = predictions.numpy()
    target_array = targets.numpy()
    activity_array = np.linalg.norm(prediction_array, axis=-1)
    azimuth_array = np.degrees(
        np.arctan2(prediction_array[..., 1], prediction_array[..., 0])
    )
    elevation_array = np.degrees(
        np.arctan2(
            prediction_array[..., 2],
            np.hypot(prediction_array[..., 0], prediction_array[..., 1]),
        )
    )

    seen: set[tuple[int, int]] = set()
    metadata_targets = np.empty_like(target_array)
    output_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(metadata_group.rows, start=1):
        try:
            frame_index = int(row[layout.frame])
            slot_index = int(row[layout.slot])
            reference_xyz = (
                float(row[layout.accdoa_x]),
                float(row[layout.accdoa_y]),
                float(row[layout.accdoa_z]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid frame/slot/ACCDOA value in {metadata_group.csv_path}, "
                f"sample {metadata_group.key}, group row {row_number}"
            ) from error

        index = (frame_index, slot_index)
        if index in seen:
            raise ValueError(
                f"Duplicate frame/slot {index} in {metadata_group.csv_path}, "
                f"sample {metadata_group.key}"
            )
        if not (0 <= frame_index < num_frames and 0 <= slot_index < num_slots):
            raise ValueError(
                f"Metadata index {index} in {metadata_group.csv_path}, sample "
                f"{metadata_group.key}, is outside prediction shape "
                f"({num_frames}, {num_slots})"
            )
        seen.add(index)
        metadata_targets[frame_index, slot_index] = reference_xyz

        x, y, z = prediction_array[frame_index, slot_index]
        activity = activity_array[frame_index, slot_index]
        row.update(
            {
                "prediction_dataset": prediction_dataset,
                "prediction_source_csv": metadata_group.csv_path.name,
                "prediction_accdoa_x": float(x),
                "prediction_accdoa_y": float(y),
                "prediction_accdoa_z": float(z),
                "prediction_activity": float(activity),
                "prediction_active": int(activity >= activity_threshold),
                "prediction_az_deg": float(azimuth_array[frame_index, slot_index]),
                "prediction_el_deg": float(elevation_array[frame_index, slot_index]),
            }
        )
        output_rows.append(row)

    expected = num_frames * num_slots
    if len(seen) != expected:
        raise ValueError(
            f"{metadata_group.csv_path}, sample {metadata_group.key}, contains "
            f"{len(seen)} unique frame/slot rows, but the prediction contains "
            f"{expected}. Refusing to save misaligned data."
        )

    if not np.allclose(metadata_targets, target_array, rtol=1e-5, atol=1e-6):
        maximum_error = float(np.max(np.abs(metadata_targets - target_array)))
        raise ValueError(
            f"Metadata sample {metadata_group.key} from {metadata_group.csv_path} "
            "does not match the corresponding PT target; maximum ACCDOA "
            f"difference={maximum_error:.6g}. Prediction alignment is unsafe."
        )

    writer.writerows(output_rows)


def make_test_loader(
    dataset_class,
    dataset_root: Path,
    normalization_statistics: dict[str, tuple[float, float]],
    hoa_order: int,
    feature_layout: str,
    batch_size: int,
    num_workers: int,
    dataset_label: str,
) -> tuple[DataLoader, Path, list[str]]:
    """Construct a test loader from <dataset_root>/test/pt."""
    pt_folder, test_files = discover_test_shards(dataset_root)

    dataset = dataset_class(
        pt_folder=str(pt_folder),
        pt_files=test_files,
        normalization_statistics=normalization_statistics,
        hoa_order=hoa_order,
        preload_label=f"test-eval-{dataset_label}",
        feature_layout=feature_layout,
    )

    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "drop_last": False,
    }

    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["multiprocessing_context"] = "fork"

    return DataLoader(**loader_kwargs), pt_folder, test_files


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def batch_frame_statistics(
    references: torch.Tensor,
    predictions: torch.Tensor,
    activity_threshold: float,
    reference_activity_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact per-frame assignment statistics in vectorized mask groups."""
    reference = references.reshape(-1, references.shape[-2], 3).numpy().astype(
        np.float64, copy=False
    )
    prediction = predictions.reshape(-1, predictions.shape[-2], 3).numpy().astype(
        np.float64, copy=False
    )
    num_frames, num_slots, _ = reference.shape

    reference_norm = np.linalg.norm(reference, axis=-1)
    prediction_norm = np.linalg.norm(prediction, axis=-1)
    reference_active = reference_norm > reference_activity_threshold
    prediction_active = prediction_norm >= activity_threshold
    reference_count = reference_active.sum(axis=1).astype(np.int64)
    prediction_count = prediction_active.sum(axis=1).astype(np.int64)

    matched_count = np.minimum(reference_count, prediction_count)
    spatial_tp = np.zeros(num_frames, dtype=np.int64)
    angle_sum = np.zeros(num_frames, dtype=np.float64)

    reference_unit = reference / np.maximum(reference_norm[..., None], EPS)
    prediction_unit = prediction / np.maximum(prediction_norm[..., None], EPS)
    cosine = np.einsum("nsi,nti->nst", reference_unit, prediction_unit)
    angular_cost = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    bit_weights = 1 << np.arange(num_slots, dtype=np.int64)
    reference_mask = reference_active @ bit_weights
    prediction_mask = prediction_active @ bit_weights
    mask_pairs = np.unique(np.stack((reference_mask, prediction_mask), axis=1), axis=0)

    for reference_bits, prediction_bits in mask_pairs:
        reference_slots = np.flatnonzero(reference_bits & bit_weights)
        prediction_slots = np.flatnonzero(prediction_bits & bit_weights)
        if reference_slots.size == 0 or prediction_slots.size == 0:
            continue

        frame_indices = np.flatnonzero(
            (reference_mask == reference_bits) & (prediction_mask == prediction_bits)
        )
        if reference_slots.size <= prediction_slots.size:
            assignments = list(
                itertools.permutations(prediction_slots.tolist(), reference_slots.size)
            )
            candidates = np.stack(
                [
                    angular_cost[
                        frame_indices[:, None],
                        reference_slots[None, :],
                        np.asarray(assignment, dtype=np.int64)[None, :],
                    ]
                    for assignment in assignments
                ],
                axis=1,
            )
        else:
            assignments = list(
                itertools.permutations(reference_slots.tolist(), prediction_slots.size)
            )
            candidates = np.stack(
                [
                    angular_cost[
                        frame_indices[:, None],
                        np.asarray(assignment, dtype=np.int64)[None, :],
                        prediction_slots[None, :],
                    ]
                    for assignment in assignments
                ],
                axis=1,
            )

        best_assignment = candidates.sum(axis=2).argmin(axis=1)
        selected_angles = candidates[np.arange(frame_indices.size), best_assignment]
        angle_sum[frame_indices] = selected_angles.sum(axis=1)
        spatial_tp[frame_indices] = (
            selected_angles <= SPATIAL_THRESHOLD_DEG
        ).sum(axis=1)

    return reference_count, prediction_count, matched_count, spatial_tp, angle_sum


def accumulate_frame_statistics(
    accumulator: MetricAccumulator,
    reference_count: np.ndarray,
    prediction_count: np.ndarray,
    matched_count: np.ndarray,
    spatial_tp: np.ndarray,
    angle_sum: np.ndarray,
    mask: np.ndarray | None = None,
) -> None:
    if mask is not None:
        reference_count = reference_count[mask]
        prediction_count = prediction_count[mask]
        matched_count = matched_count[mask]
        spatial_tp = spatial_tp[mask]
        angle_sum = angle_sum[mask]

    accumulator.num_frames += int(reference_count.size)
    accumulator.num_reference_events += int(reference_count.sum())
    accumulator.num_prediction_events += int(prediction_count.sum())
    accumulator.count_abs_error_sum += int(
        np.abs(reference_count - prediction_count).sum()
    )
    accumulator.exact_count_frames += int(
        (reference_count == prediction_count).sum()
    )
    accumulator.tp += int(matched_count.sum())
    accumulator.fn += int((reference_count - matched_count).sum())
    accumulator.fp += int((prediction_count - matched_count).sum())
    accumulator.tp_spatial += int(spatial_tp.sum())
    accumulator.fp_spatial += int((matched_count - spatial_tp).sum())
    accumulator.localization_error_sum += float(angle_sum.sum())
    accumulator.localization_matches += int(matched_count.sum())


def finalize_metrics(accumulator: MetricAccumulator) -> dict[str, float]:
    # F15 and ER15 use the paper's fixed 15-degree location threshold.
    precision_15 = safe_divide(
        accumulator.tp_spatial,
        accumulator.tp_spatial
        + accumulator.fp_spatial
        + accumulator.fp,
    )

    recall_15 = safe_divide(
        accumulator.tp_spatial,
        accumulator.tp_spatial + accumulator.fn,
    )

    f15 = safe_divide(
        2.0 * precision_15 * recall_15,
        precision_15 + recall_15,
    )

    er15 = safe_divide(
        accumulator.fp
        + accumulator.fp_spatial
        + accumulator.fn,
        accumulator.num_reference_events,
    )

    # LE and LR are threshold-free localization metrics. Every Hungarian
    # assignment contributes to LE, irrespective of its angular error.
    le = safe_divide(
        accumulator.localization_error_sum,
        accumulator.localization_matches,
    )

    lr = safe_divide(
        accumulator.tp,
        accumulator.tp + accumulator.fn,
    )

    esdl = 0.25 * (
        er15
        + (1.0 - f15)
        + le / 180.0
        + (1.0 - lr)
    )

    return {
        "LE": le,
        "LR": lr,
        "F15": f15,
        "ER15": er15,
        "ESDL": esdl,
    }


@torch.inference_mode()
def evaluate_every_frame(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    activity_threshold: float,
    reference_activity_threshold: float,
    csv_paths: list[Path],
    prediction_output_path: Path | None,
    metadata_fieldnames: list[str],
    output_metadata_fieldnames: list[str],
    prediction_dataset: str,
    append_prediction_output: bool,
) -> dict[str, Any]:
    """
    Evaluate all frames and also evaluate frames grouped by the number of
    active ground-truth sources.

    The grouped metrics are conditioned on the reference source count:
        by_gt_source_count["0"] -> frames with 0 active reference sources
        by_gt_source_count["1"] -> frames with 1 active reference source
        by_gt_source_count["2"] -> frames with 2 active reference sources
        by_gt_source_count["3"] -> frames with 3 active reference sources
    """
    model.eval()

    overall_accumulator = MetricAccumulator()
    count_accumulators = {
        count: MetricAccumulator()
        for count in range(4)
    }
    source_count_confusion = np.zeros((4, 4), dtype=np.int64)
    sample_offset = 0

    prediction_handle = None
    prediction_writer = None
    metadata_groups = None
    metadata_layout = None
    if prediction_output_path is not None:
        if not csv_paths:
            raise ValueError("Prediction CSV output requires metadata CSV shards.")
        metadata_layout = resolve_metadata_layout(metadata_fieldnames)
        metadata_groups = iter_metadata_sample_groups(csv_paths, metadata_layout)
        prediction_handle = prediction_output_path.open(
            "a" if append_prediction_output else "w",
            encoding="utf-8",
            newline="",
        )
        prediction_writer = csv.DictWriter(
            prediction_handle,
            fieldnames=output_metadata_fieldnames + list(PREDICTION_COLUMNS),
            restval="",
        )
        if not append_prediction_output:
            prediction_writer.writeheader()

    try:
        for batch_index, (features, targets) in enumerate(loader, start=1):
            features = features.to(device, non_blocking=True)

            predictions = model(features)

            if predictions.ndim == 3:
                if predictions.shape[-1] % 3 != 0:
                    raise ValueError(
                        "The model output's final dimension must be divisible by 3."
                    )

                predictions = predictions.reshape(
                    predictions.shape[0],
                    predictions.shape[1],
                    predictions.shape[-1] // 3,
                    3,
                )

            if targets.ndim != 4 or targets.shape[-1] != 3:
                raise ValueError(
                    "Expected targets with shape (B, T, tracks, 3), "
                    f"got {tuple(targets.shape)}."
                )

            if predictions.shape != targets.shape:
                raise ValueError(
                    f"Prediction shape {tuple(predictions.shape)} does not match "
                    f"target shape {tuple(targets.shape)}."
                )

            predictions = predictions.detach().cpu()
            targets = targets.detach().cpu()

            batch_size, num_frames, _, _ = targets.shape

            if prediction_writer is not None:
                assert metadata_groups is not None
                assert metadata_layout is not None
                for sample_index in range(batch_size):
                    try:
                        metadata_group = next(metadata_groups)
                    except StopIteration as error:
                        raise ValueError(
                            "Metadata CSV shards ended after "
                            f"{sample_offset + sample_index} sample groups, but "
                            f"the dataset exposes {len(loader.dataset)} samples."
                        ) from error
                    write_sample_predictions(
                        writer=prediction_writer,
                        metadata_group=metadata_group,
                        predictions=predictions[sample_index],
                        targets=targets[sample_index],
                        activity_threshold=activity_threshold,
                        layout=metadata_layout,
                        prediction_dataset=prediction_dataset,
                    )

            statistics = batch_frame_statistics(
                references=targets,
                predictions=predictions,
                activity_threshold=activity_threshold,
                reference_activity_threshold=reference_activity_threshold,
            )
            reference_count, prediction_count, matched_count, spatial_tp, angle_sum = statistics
            accumulate_frame_statistics(
                overall_accumulator,
                reference_count,
                prediction_count,
                matched_count,
                spatial_tp,
                angle_sum,
            )
            for source_count, accumulator in count_accumulators.items():
                accumulate_frame_statistics(
                    accumulator,
                    reference_count,
                    prediction_count,
                    matched_count,
                    spatial_tp,
                    angle_sum,
                    mask=reference_count == source_count,
                )

            max_count = source_count_confusion.shape[0] - 1
            np.add.at(
                source_count_confusion,
                (
                    np.minimum(reference_count, max_count),
                    np.minimum(prediction_count, max_count),
                ),
                1,
            )

            sample_offset += batch_size
            if batch_index % 100 == 0:
                print(
                    f"Processed {batch_index} batches / "
                    f"{overall_accumulator.num_frames} frames"
                )

        if metadata_groups is not None:
            extra_group = next(metadata_groups, None)
            if extra_group is not None:
                raise ValueError(
                    f"Dataset ended after {sample_offset} samples, but metadata "
                    "contains additional sample group "
                    f"{extra_group.key} in {extra_group.csv_path}."
                )
    finally:
        if metadata_groups is not None:
            metadata_groups.close()
        if prediction_handle is not None:
            prediction_handle.close()

    return {
        "overall": finalize_metrics(overall_accumulator),
        "by_gt_source_count": {
            str(count): finalize_metrics(accumulator)
            for count, accumulator in count_accumulators.items()
        },
        "source_count_confusion_matrix": source_count_confusion.tolist(),
    }



def print_metric_block(
    results: dict[str, float],
    title: str,
) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)
    print(f"LE                                    {results['LE']:.2f} deg")
    print(f"LR                                    {results['LR']:.4f}")
    print(f"F15                                   {results['F15']:.4f}")
    print(f"ER15                                  {results['ER15']:.4f}")
    print(f"ESDL                                  {results['ESDL']:.4f}")


def print_results(results: dict[str, Any], dataset_name: str) -> None:
    print_metric_block(
        results["overall"],
        f"{dataset_name}: framewise test-set SDL evaluation: all frames",
    )

    for source_count in range(4):
        print_metric_block(
            results["by_gt_source_count"][str(source_count)],
            (
                f"{dataset_name}: framewise SDL evaluation: "
                f"{source_count} active ground-truth source(s)"
            ),
        )


def safe_dataset_name(dataset_root: Path, index: int) -> str:
    name = dataset_root.resolve().name or f"dataset_{index}"
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in name)
    return f"{index}_{safe}"


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    checkpoint = load_checkpoint_first(checkpoint_path, device)
    model_class, dataset_class = import_training_objects(args.train_module)

    checkpoint_dataset_root = Path(checkpoint["dataset_root"]).expanduser().resolve()
    distribution_path, normalization_statistics, hoa_order = load_training_distribution(
        checkpoint
    )

    requested_dataset_specs: list[tuple[str, Path]] = [
        ("checkpoint_dataset", checkpoint_dataset_root),
    ]
    for dataset_name, dataset_root in (
        ("arni-dataset", args.arni_dataset),
        ("motus-dataset", args.motus_dataset),
        ("ivan-dataset", args.ivan_dataset),
    ):
        if dataset_root is not None:
            requested_dataset_specs.append((dataset_name, dataset_root))

    # Do not silently evaluate the same resolved dataset path more than once.
    dataset_specs: list[tuple[str, Path]] = []
    seen_dataset_roots: set[Path] = set()
    for dataset_label, dataset_root in requested_dataset_specs:
        resolved_root = dataset_root.expanduser().resolve()
        if resolved_root in seen_dataset_roots:
            print(f"Skipping duplicate dataset: {dataset_label} ({resolved_root})")
            continue
        seen_dataset_roots.add(resolved_root)
        dataset_specs.append((dataset_label, resolved_root))

    input_channels = int(checkpoint.get("input_channels", 46))
    num_speakers = int(checkpoint.get("num_speakers", 3))
    feature_layout = str(checkpoint.get("feature_layout", "hopiv_then_logmel"))

    print(f"Checkpoint:                    {checkpoint_path}")
    print(f"Device:                        {device}")
    print(f"Checkpoint dataset root:       {checkpoint_dataset_root}")
    print(f"Normalization distribution:    {distribution_path}")
    print(f"Number of evaluation datasets: {len(dataset_specs)}")

    model = model_class(
        input_channels=input_channels,
        num_speakers=num_speakers,
        dropout_rate=0.1,
    ).to(device)
    model.load_state_dict(normalize_state_dict(checkpoint["model_state_dict"]))

    if "epoch" in checkpoint:
        print(f"Loaded epoch:                  {checkpoint['epoch']}")
    if "best_val_loss" in checkpoint:
        print(f"Best val loss:                 {checkpoint['best_val_loss']:.6f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {}
    combined_prediction_path = None if args.skip_prediction_csv else (
        args.output_dir / f"predictions_{checkpoint_title(args.checkpoint, checkpoint_path)}.csv"
    )
    metadata_by_dataset_root: dict[Path, tuple[list[Path], list[str]]] = {}
    combined_metadata_fieldnames: list[str] = []
    if combined_prediction_path is not None:
        all_metadata_fieldnames: list[list[str]] = []
        for _, dataset_root in dataset_specs:
            preflight_test_folder, preflight_test_files = discover_test_shards(
                dataset_root
            )
            csv_paths, metadata_fieldnames = discover_metadata_csvs(
                preflight_test_folder,
                preflight_test_files,
            )
            metadata_by_dataset_root[dataset_root] = (
                csv_paths,
                metadata_fieldnames,
            )
            all_metadata_fieldnames.append(metadata_fieldnames)
        combined_metadata_fieldnames = union_fieldnames(all_metadata_fieldnames)

    prediction_file_started = False

    for dataset_index, (dataset_label, dataset_root) in enumerate(dataset_specs, start=1):
        dataset_root = dataset_root.expanduser().resolve()
        dataset_name = safe_dataset_name(Path(dataset_label), dataset_index)

        print("\n" + "#" * 76)
        print(f"Evaluating dataset {dataset_index}/{len(dataset_specs)}: {dataset_label} ({dataset_root})")
        print("#" * 76)

        test_loader, test_folder, test_files = make_test_loader(
            dataset_class=dataset_class,
            dataset_root=dataset_root,
            normalization_statistics=normalization_statistics,
            hoa_order=hoa_order,
            feature_layout=feature_layout,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            dataset_label=dataset_name,
        )

        print(f"Test folder: {test_folder}")
        print(f"Test shards: {len(test_files)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        print(f"Normalization source: {distribution_path}")

        if args.skip_prediction_csv:
            # Aggregate metrics depend only on PT tensors.  Do not require or
            # scan metadata CSVs when no augmented prediction file is wanted.
            csv_paths: list[Path] = []
            metadata_fieldnames: list[str] = []
            prediction_output_path = None
            append_prediction_output = False
        else:
            csv_paths, metadata_fieldnames = metadata_by_dataset_root[dataset_root]
            prediction_output_path = combined_prediction_path
            append_prediction_output = prediction_file_started

        results = evaluate_every_frame(
            model=model,
            loader=test_loader,
            device=device,
            activity_threshold=args.activity_threshold,
            reference_activity_threshold=args.reference_activity_threshold,
            csv_paths=csv_paths,
            prediction_output_path=prediction_output_path,
            metadata_fieldnames=metadata_fieldnames,
            output_metadata_fieldnames=combined_metadata_fieldnames,
            prediction_dataset=dataset_label,
            append_prediction_output=append_prediction_output,
        )
        if prediction_output_path is not None:
            prediction_file_started = True
            print(
                "Appended predictions + metadata to combined file: "
                f"{prediction_output_path}"
            )
        print_results(results, dataset_name=dataset_label)

        source_count_confusion = np.asarray(
            results["source_count_confusion_matrix"],
            dtype=np.int64,
        )
        row_totals = source_count_confusion.sum(axis=1, keepdims=True)
        source_count_confusion_percent = np.divide(
            source_count_confusion.astype(np.float64),
            row_totals,
            out=np.zeros_like(source_count_confusion, dtype=np.float64),
            where=row_totals != 0,
        ) * 100.0

        print("\nSource-count confusion matrix (%) (rows=true, columns=predicted):")
        print(np.round(source_count_confusion_percent, 2))
        result_payload = {
            "dataset_label": dataset_label,
            "dataset_root": str(dataset_root),
            "test_folder": str(test_folder),
            #"test_files": test_files,
            "normalization_distribution": str(distribution_path),
            "predictions_with_metadata": (
                str(prediction_output_path)
                if prediction_output_path is not None
                else None
            ),
            "results": results,
            "source_count_confusion_matrix_counts": source_count_confusion.tolist(),
            "source_count_confusion_matrix_percent": source_count_confusion_percent.tolist(),
        }
        all_results[dataset_name] = result_payload

        del test_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if combined_prediction_path is not None and prediction_file_started:
        print(f"\nSaved combined predictions: {combined_prediction_path}")

    output_path = args.output_dir / f"eval_{checkpoint_title(args.checkpoint, checkpoint_path)}.json"
    combined_payload = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "checkpoint_dataset_root": str(checkpoint_dataset_root),
        "normalization_distribution": str(distribution_path),
        "activity_threshold": args.activity_threshold,
        "reference_activity_threshold": args.reference_activity_threshold,
        "datasets": all_results,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(combined_payload, handle, indent=2)
    print(f"\nSaved combined results: {output_path}")


if __name__ == "__main__":
    main()
