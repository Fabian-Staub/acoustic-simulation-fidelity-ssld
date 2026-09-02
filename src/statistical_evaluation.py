#!/usr/bin/env python3
"""Jackknife significance analysis for raw prediction CSVs from eval_5_fast.py.

Each ``--csvs`` input is one model's augmented prediction CSV.  Metrics are
recomputed from the reference and predicted ACCDOA vectors.  A unique room and
receiver-position pair parsed from ``sample_id`` is one independent sample;
for example, ``R1800_P00`` becomes room ``R1800`` and position ``P00``.  Room
names are not restricted to ``R`` followed by a number.

Overall metrics, metrics conditioned on exactly 1, 2, or 3 active
ground-truth sources, and metrics conditioned on reverberation time or the
direct-to-reverberant ratio (DRR) are
reported for every dataset.  The input must contain ``rt60_s`` for every source
slot; ``drr_db`` is optional.  A frame's finite RT60/DRR is the mean of
the corresponding column over the active ground-truth slots,
where activity uses the same reference ACCDOA threshold as the existing
metrics.  Inactive slots and predicted activity never affect RT60.  The one
frame RT60 is used for every metric, including every LE match from that frame,
and is first assigned to a deterministic half-open seed bin such as
``[0.2, 0.4)``. By default, adjacent seed bins are then merged until each
resulting interval contains a configurable minimum number of independent
receiver positions. Real evaluation datasets share one set of adaptive
boundaries across all models; simulated datasets use model-specific boundaries
because their reference RT60 distributions can differ between models.
Zero-reference frames remain in the existing analyses but have no RT60 or DRR
bin. DRR uses the same binning and resampling construction, with widths in dB;
its negative-valued bins retain the same half-open ``[low, high)`` convention.
If ``drr_db`` is absent, or the active-slot mean is greater than +35 dB, the
frame is assigned to a separate infinite-DRR category.

The delete-one unit remains the unique room/receiver-position pair.  RT60 bins
contain only positions with at least one frame in that bin; RT60-conditioned
jackknife estimates and confidence intervals therefore use each model's own
contributing positions.  No pairwise significance tests are performed for
individual RT60 bins.  ``checkpoint_dataset`` is
reported as ``simulated-dataset``, is optional for each model, and is
metrics-only: it is never included in pairwise significance tests
or multiple-comparison correction. ``model_metrics.csv`` also contains
event-detection confusion counts and percentages plus a ground-truth versus
predicted source-count confusion matrix. Source-count percentages are
normalized separately for each ground-truth source-count row.

For every model metric and every paired model difference, this script uses
the delete-one jackknife construction described by Mesaros et al., "Sound
Event Detection in the DCASE 2017 Challenge": full-sample and leave-one-out
estimates, pseudo-values, the bias-corrected jackknife estimate, its standard
error, and a Student-t confidence interval with N-1 degrees of freedom. The
standard error uses the standard delete-one jackknife scaling; the printed
leave-one-out form of Equation (6) in that paper is missing a factor of
(N-1)^2 and would make standard errors N-1 times too small.

A paired difference is
Raw two-sided p-values are obtained from the jackknife t statistic. By default,
Holm's method corrects all model-pair tests within each dataset/metric family;
the correction method and family scope are configurable on the command line.

Examples:
    python compare_receiver_position_metrics.py \
        --csvs \
        evaluation_results/predictions_shoeBox_SH_Direct.csv \
        evaluation_results/predictions_shoeBox_SH_ISMOrder12.csv \
        --output-dir significance_results

    python compare_receiver_position_metrics.py \
        --csvs \
        "Direct=predictions_shoeBox_SH_Direct.csv" \
        "ISM12=predictions_shoeBox_SH_ISMOrder12.csv"

The input CSV filenames/paths are supplied through the required ``--csvs``
argument. Their command-line order is
preserved everywhere: model summaries and pairwise-comparison columns.

Outputs:
    model_metrics.csv
    model_metrics_by_rt60.csv
    model_metrics_by_drr.csv
    pairwise_significance.csv
    significance_matrix.csv
    significance_results.json

Requirements:
    pip install numpy scipy
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from itertools import combinations, repeat
from pathlib import Path
from typing import Iterable, Sequence


import numpy as np
from scipy.stats import t as student_t
from scipy.optimize import linear_sum_assignment


EPS = 1e-12
DRR_INFINITY_BIN = "inf"
DRR_INFINITY_THRESHOLD_DB = Decimal("35")
SPATIAL_THRESHOLD_DEG = 15.0
DEFAULT_METRICS = (
    "LE",
    "LR",
    "F",
    "F15",
    "ER15",
    "ESDL",
    "count_MAE",
    "count_accuracy",
)
LOWER_IS_BETTER = {"LE", "ER15", "ESDL", "count_MAE"}
METRICS_ONLY_DATASET = "simulated-dataset"
MODEL_FILENAME_PREFIXES = ("metrics_receiverPos_", "predictions_")


@dataclass
class ModelData:
    label: str
    path: Path
    # dataset -> room/receiver-position key -> additive sufficient statistics
    samples: dict[str, dict[str, "MetricAccumulator"]]
    # dataset -> active reference-source count -> position -> statistics
    samples_by_gt_source_count: dict[
        str, dict[int, dict[str, "MetricAccumulator"]]
    ]
    # dataset -> integer RT60-bin index -> position -> statistics.  Unlike the
    # source-count slices, absent positions are deliberately not materialized.
    samples_by_rt60: dict[str, dict[int, dict[str, "MetricAccumulator"]]]
    # dataset -> output-bin index -> half-open interval of seed-bin indices.
    rt60_bin_ranges: dict[str, dict[int, tuple[int, int]]]
    # DRR mirrors the RT60 construction, using drr_db from the prediction CSV.
    samples_by_drr: dict[
        str, dict[int | str, dict[str, "MetricAccumulator"]]
    ]
    drr_bin_ranges: dict[str, dict[int, tuple[int, int]]]
    # dataset -> (ground-truth source count, predicted source count) -> frames
    source_count_confusions: dict[str, dict[tuple[int, int], int]]


@dataclass
class MetricAccumulator:
    num_frames: int = 0
    num_reference_events: int = 0
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    tp_spatial: int = 0
    fp_spatial: int = 0
    localization_error_sum: float = 0.0
    localization_matches: int = 0
    exact_count_frames: int = 0
    count_abs_error_sum: int = 0

    def __add__(self, other: "MetricAccumulator") -> "MetricAccumulator":
        return MetricAccumulator(**{
            name: getattr(self, name) + getattr(other, name)
            for name in self.__dataclass_fields__
        })

    def __sub__(self, other: "MetricAccumulator") -> "MetricAccumulator":
        return MetricAccumulator(**{
            name: getattr(self, name) - getattr(other, name)
            for name in self.__dataclass_fields__
        })


@dataclass(frozen=True)
class JackknifeEstimate:
    full_sample: float
    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float


@dataclass
class ComparisonRow:
    dataset: str
    model_a: str
    model_b: str
    metric: str
    n_paired_positions: int
    estimate_a: float
    estimate_b: float
    difference_a_minus_b: float
    ci_low: float
    ci_high: float
    jackknife_standard_error: float
    p_value: float
    adjusted_p_value: float = math.nan
    correction_method: str = "none"
    correction_family: str = ""
    p_value_display: float = math.nan
    significant: bool = False
    better_model: str = ""


@dataclass
class ModelJackknifeData:
    full_metrics: dict[str, float]
    leave_one_out: dict[str, np.ndarray]
    estimates: dict[str, JackknifeEstimate]


@dataclass
class BinnedJackknifeData:
    """Prepared bin metrics; jackknife is absent when fewer than two units exist."""
    full_metrics: dict[str, float]
    number_of_receiver_positions: int
    number_of_frames: int
    jackknife: ModelJackknifeData | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csvs",
        nargs="+",
        required=True,
        metavar="[LABEL=]CSV",
        help=(
            "Full filenames/paths of two or more prediction CSVs written by "
            "the eval script, in display order. "
            "An optional LABEL= prefix overrides the label derived from a "
            "filename."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("significance_results"),
    )
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "prediction_dataset value to skip entirely; repeat to exclude more. "
            "The checkpoint_dataset is retained as simulated-dataset for metrics "
            "but is always omitted from significance comparisons."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=DEFAULT_METRICS,
        default=list(DEFAULT_METRICS),
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
        "--rt60-bin-width",
        type=float,
        default=0.2,
        metavar="SECONDS",
        help=(
            "Width in seconds of the initial half-open RT60 seed bins "
            "(default: %(default)s). Adaptive binning merges adjacent seed "
            "bins; fixed binning reports each seed bin unchanged."
        ),
    )
    parser.add_argument(
        "--rt60-binning",
        choices=("adaptive", "fixed"),
        default="adaptive",
        help=(
            "RT60 binning strategy (default: %(default)s). Adaptive bins are "
            "shared across models for real datasets and selected separately "
            "per model for simulated-dataset."
        ),
    )
    parser.add_argument(
        "--rt60-min-positions",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Minimum number of unique room/receiver positions targeted in "
            "each adaptive RT60 bin (default: %(default)s). The final bin is "
            "merged into its predecessor when necessary; a dataset with "
            "fewer than N contributing positions produces one flagged bin."
        ),
    )
    parser.add_argument(
        "--drr-bin-width",
        type=float,
        default=5.0,
        metavar="DB",
        help=(
            "Width in dB of the initial half-open DRR seed bins "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--drr-binning",
        choices=("adaptive", "fixed"),
        default="adaptive",
        help=(
            "DRR binning strategy (default: %(default)s). Adaptive bins are "
            "shared across models for real datasets and selected separately "
            "per model for simulated-dataset."
        ),
    )
    parser.add_argument(
        "--drr-min-positions",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Minimum number of unique room/receiver positions targeted in "
            "each adaptive DRR bin (default: %(default)s)."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--multiple-comparison-correction",
        choices=("none", "bonferroni", "holm", "fdr_bh"),
        default="holm",
        help=(
            "Correction applied to pairwise p-values (default: %(default)s). "
            "Holm and Bonferroni control the family-wise error rate; fdr_bh "
            "controls the false discovery rate."
        ),
    )
    parser.add_argument(
        "--correction-scope",
        choices=("dataset_metric", "dataset", "metric", "global"),
        default="dataset_metric",
        help=(
            "Tests grouped into one correction family (default: %(default)s). "
            "dataset_metric corrects all model pairs separately for each "
            "dataset and metric."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Processes shared by CSV parsing, jackknife preparation, and large "
            "pairwise-comparison batches. "
            "0 chooses up to the available CPU count automatically; use 1 "
            "for fully sequential execution."
        ),
    )
    return parser.parse_args()


def split_model_argument(argument: str) -> tuple[str | None, Path]:
    label: str | None = None
    path_text = argument
    if "=" in argument:
        possible_label, possible_path = argument.split("=", 1)
        # Treat the prefix as a label only when the complete argument is not
        # itself an existing path. This keeps filenames containing '=' valid.
        if possible_label.strip() and not Path(argument).expanduser().exists():
            label = possible_label.strip()
            path_text = possible_path
    return label, Path(path_text).expanduser().resolve()


def label_from_path(path: Path) -> str:
    stem = path.stem
    for prefix in MODEL_FILENAME_PREFIXES:
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    if not stem:
        raise ValueError(f"Could not derive a model label from {path.name!r}")
    return stem


def collect_model_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    # argparse preserves the order of values passed to --csvs. Do not sort this
    # list: it is the authoritative display order for all downstream outputs.
    raw_specs = [split_model_argument(value) for value in args.csvs]

    specs: list[tuple[str, Path]] = []
    seen_paths: set[Path] = set()
    seen_labels: set[str] = set()
    for explicit_label, path in raw_specs:
        if path in seen_paths:
            raise ValueError(f"CSV path was specified more than once: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Metric CSV not found: {path}")
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file, got {path.name!r}")
        label = explicit_label or label_from_path(path)
        if label in seen_labels:
            raise ValueError(
                f"Duplicate model label {label!r}; use LABEL=PATH to disambiguate"
            )
        seen_paths.add(path)
        seen_labels.add(label)
        specs.append((label, path))

    if len(specs) < 2:
        raise ValueError(
            "--csvs requires at least two metric CSV filenames/paths."
        )
    return specs


def normalized_column(fieldnames: Sequence[str], wanted: str) -> str | None:
    for fieldname in fieldnames:
        normalized = fieldname.replace("\\", "").strip().strip("*").lower()
        if normalized == wanted.lower():
            return fieldname
    return None


def require_column(fieldnames: Sequence[str], wanted: str) -> str:
    column = normalized_column(fieldnames, wanted)
    if column is None:
        raise ValueError(f"Missing column {wanted!r}; found {list(fieldnames)}")
    return column


def finite_float(value: str, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value in {context}: {value!r}") from error
    return parsed


def finite_decimal(value: str, context: str) -> Decimal:
    """Parse an exact finite decimal for stable RT60 averaging and binning."""
    try:
        parsed = Decimal(value.strip())
    except (AttributeError, InvalidOperation) as error:
        raise ValueError(f"Invalid numeric value in {context}: {value!r}") from error
    if not parsed.is_finite():
        raise ValueError(f"Non-finite numeric value in {context}: {value!r}")
    return parsed


def rt60_bin_index(frame_rt60: Decimal, bin_width: Decimal) -> int:
    """Return the integer index of the half-open bin containing frame_rt60."""
    return int((frame_rt60 / bin_width).to_integral_value(rounding=ROUND_FLOOR))


def drr_bin_index(frame_drr: Decimal, bin_width: Decimal) -> int:
    """Return the integer index of the half-open dB bin containing frame DRR."""
    return int((frame_drr / bin_width).to_integral_value(rounding=ROUND_FLOOR))


def frame_drr_bin_index(
    active_values: Sequence[str] | None,
    bin_width: Decimal,
    context: str,
) -> int | str:
    """Return a finite DRR bin, or the dedicated infinity category.

    A missing ``drr_db`` column makes DRR infinite for every frame. Explicit
    positive infinity and an active-source mean above +35 dB are treated the
    same way. Other non-finite or invalid active values remain input errors.
    """
    if active_values is None:
        return DRR_INFINITY_BIN
    parsed: list[Decimal] = []
    for value in active_values:
        try:
            drr = Decimal(value.strip())
        except (AttributeError, InvalidOperation) as error:
            raise ValueError(f"Invalid numeric value in {context}: {value!r}") from error
        if drr == Decimal("Infinity"):
            return DRR_INFINITY_BIN
        if not drr.is_finite():
            raise ValueError(f"Non-finite numeric value in {context}: {value!r}")
        parsed.append(drr)
    frame_drr = sum(parsed, Decimal(0)) / Decimal(len(parsed))
    if frame_drr > DRR_INFINITY_THRESHOLD_DB:
        return DRR_INFINITY_BIN
    return drr_bin_index(frame_drr, bin_width)


def rt60_bin_bounds(
    low_index: int, high_index: int, bin_width: float
) -> tuple[float, float, float]:
    """Return exact-decimal-derived labels for a half-open seed-index range."""
    width = Decimal(str(bin_width))
    low = Decimal(low_index) * width
    high = Decimal(high_index) * width
    center = (low + high) / Decimal(2)
    return float(low), float(high), float(center)


def drr_bin_bounds(
    low_index: int, high_index: int, bin_width: float
) -> tuple[float, float, float]:
    """Return exact-decimal-derived labels for a half-open DRR-bin range."""
    width = Decimal(str(bin_width))
    low = Decimal(low_index) * width
    high = Decimal(high_index) * width
    center = (low + high) / Decimal(2)
    return float(low), float(high), float(center)


def receiver_position_key(sample_id: str, context: str) -> str:
    """Extract the final ``<room>_P<digits>`` prefix from a sample id."""
    matches = list(re.finditer(r"_P(\d+)(?=$|_)", sample_id))
    if not matches:
        raise ValueError(
            f"Cannot extract room/receiver position from sample_id {sample_id!r} "
            f"in {context}; expected '<room>_P<digits>'."
        )
    match = matches[-1]
    room = sample_id[: match.start()]
    if not room:
        raise ValueError(f"Empty room name in sample_id {sample_id!r} ({context})")
    position = f"P{match.group(1)}"
    return json.dumps((room, position), ensure_ascii=False, separators=(",", ":"))


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def finalize_metrics(accumulator: MetricAccumulator) -> dict[str, float]:
    # Location-independent event-detection F-score.  Unlike F15, this uses
    # the ordinary TP/FP/FN counts and therefore does not depend on whether a
    # matched prediction falls within the spatial threshold.
    f_score = safe_divide(
        2 * accumulator.tp,
        2 * accumulator.tp + accumulator.fp + accumulator.fn,
    )
    precision_15 = safe_divide(
        accumulator.tp_spatial,
        accumulator.tp_spatial + accumulator.fp_spatial + accumulator.fp,
    )
    recall_15 = safe_divide(
        accumulator.tp_spatial, accumulator.tp_spatial + accumulator.fn
    )
    f15 = safe_divide(2.0 * precision_15 * recall_15, precision_15 + recall_15)
    er15 = safe_divide(
        accumulator.fp + accumulator.fp_spatial + accumulator.fn,
        accumulator.num_reference_events,
    )
    le = safe_divide(
        accumulator.localization_error_sum, accumulator.localization_matches
    )
    lr = safe_divide(accumulator.tp, accumulator.tp + accumulator.fn)
    esdl = 0.25 * (er15 + (1.0 - f15) + le / 180.0 + (1.0 - lr))
    return {
        "LE": le,
        "LR": lr,
        "F": f_score,
        "F15": f15,
        "ER15": er15,
        "ESDL": esdl,
        "count_MAE": safe_divide(
            accumulator.count_abs_error_sum, accumulator.num_frames
        ),
        "count_accuracy": safe_divide(
            accumulator.exact_count_frames, accumulator.num_frames
        ),
    }


def accumulate_frame(
    accumulator: MetricAccumulator,
    references: np.ndarray,
    predictions: np.ndarray,
    activity_threshold: float,
    reference_activity_threshold: float,
) -> int:
    reference_norm = np.linalg.norm(references, axis=1)
    prediction_norm = np.linalg.norm(predictions, axis=1)
    reference_slots = np.flatnonzero(reference_norm > reference_activity_threshold)
    prediction_slots = np.flatnonzero(prediction_norm >= activity_threshold)
    reference_count = len(reference_slots)
    prediction_count = len(prediction_slots)
    matched_count = min(reference_count, prediction_count)
    selected_angles = np.empty(0, dtype=np.float64)

    if matched_count:
        reference_unit = references[reference_slots] / reference_norm[reference_slots, None]
        prediction_unit = predictions[prediction_slots] / prediction_norm[prediction_slots, None]
        angular_cost = np.degrees(
            np.arccos(np.clip(reference_unit @ prediction_unit.T, -1.0, 1.0))
        )
        # Exact rectangular assignment in O(n^3), without constructing every
        # subset/permutation and all of its temporary arrays.
        row_indices, column_indices = linear_sum_assignment(angular_cost)
        selected_angles = angular_cost[row_indices, column_indices]

    spatial_tp = int(np.count_nonzero(selected_angles <= SPATIAL_THRESHOLD_DEG))
    accumulator.num_frames += 1
    accumulator.num_reference_events += reference_count
    accumulator.count_abs_error_sum += abs(reference_count - prediction_count)
    accumulator.exact_count_frames += int(reference_count == prediction_count)
    accumulator.tp += matched_count
    accumulator.tn += len(references) - max(reference_count, prediction_count)
    accumulator.fn += reference_count - matched_count
    accumulator.fp += prediction_count - matched_count
    accumulator.tp_spatial += spatial_tp
    accumulator.fp_spatial += matched_count - spatial_tp
    accumulator.localization_error_sum += float(np.sum(selected_angles))
    accumulator.localization_matches += matched_count
    return reference_count


def merge_accumulator(
    target: MetricAccumulator, source: MetricAccumulator
) -> None:
    """Add already-computed frame statistics without repeating source matching."""
    target.num_frames += source.num_frames
    target.num_reference_events += source.num_reference_events
    target.tp += source.tp
    target.tn += source.tn
    target.fp += source.fp
    target.fn += source.fn
    target.tp_spatial += source.tp_spatial
    target.fp_spatial += source.fp_spatial
    target.localization_error_sum += source.localization_error_sum
    target.localization_matches += source.localization_matches
    target.exact_count_frames += source.exact_count_frames
    target.count_abs_error_sum += source.count_abs_error_sum


def canonical_dataset_name(dataset: str) -> str:
    """Map checkpoint-dataset aliases to the metrics-only simulated dataset."""
    normalized = dataset.strip().lower().replace("-", "_")

    if normalized in {"checkpoint_dataset", "chekpoint_dataset"}:
        return METRICS_ONLY_DATASET

    return dataset


def load_model(
    label: str,
    path: Path,
    excluded_datasets: set[str],
    activity_threshold: float,
    reference_activity_threshold: float,
    rt60_bin_width: float,
    drr_bin_width: float,
) -> ModelData:
    samples: dict[str, dict[str, MetricAccumulator]] = {}
    samples_by_gt_source_count: dict[
        str, dict[int, dict[str, MetricAccumulator]]
    ] = {}
    samples_by_rt60: dict[
        str, dict[int, dict[str, MetricAccumulator]]
    ] = {}
    samples_by_drr: dict[
        str, dict[int | str, dict[str, MetricAccumulator]]
    ] = {}
    source_count_confusions: dict[str, dict[tuple[int, int], int]] = {}
    position_cache: dict[str, str] = {}
    bin_width_decimal = Decimal(str(rt60_bin_width))
    drr_bin_width_decimal = Decimal(str(drr_bin_width))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            fieldnames = next(reader)
        except StopIteration:
            raise ValueError(f"CSV has no header: {path}")
        dataset_column = require_column(fieldnames, "prediction_dataset")
        required_column_names = (
            "sample_id", "segment_idx", "frame_idx", "slot", "accdoa_x",
            "accdoa_y", "accdoa_z", "prediction_accdoa_x",
            "prediction_accdoa_y", "prediction_accdoa_z", "rt60_s",
        )
        named_columns = {
            name: require_column(fieldnames, name)
            for name in required_column_names
        }
        drr_column = normalized_column(fieldnames, "drr_db")
        if drr_column is not None:
            named_columns["drr_db"] = drr_column
        index_by_name = {name: index for index, name in enumerate(fieldnames)}
        dataset_index = index_by_name[dataset_column]
        columns = {name: index_by_name[column] for name, column in named_columns.items()}
        current_frame: tuple[str, str, str, str, str] | None = None
        frame_rows: list[
            tuple[int, float, float, float, float, float, float, str, str]
        ] = []

        def flush_frame() -> None:
            nonlocal frame_rows
            if current_frame is None:
                return
            dataset, position, _sample_id, _segment, _frame = current_frame
            ordered = sorted(frame_rows, key=lambda item: item[0])
            slots = [item[0] for item in ordered]
            if slots != list(range(len(slots))):
                raise ValueError(
                    f"Frame slots must be contiguous from zero in {path}: {slots}"
                )
            accumulator = samples.setdefault(dataset, {}).setdefault(
                position, MetricAccumulator()
            )
            values = np.asarray([item[:7] for item in ordered], dtype=np.float64)
            references = values[:, 1:4]
            frame_accumulator = MetricAccumulator()
            reference_count = accumulate_frame(
                frame_accumulator,
                references,
                values[:, 4:7],
                activity_threshold,
                reference_activity_threshold,
            )
            merge_accumulator(accumulator, frame_accumulator)
            prediction_count = frame_accumulator.tp + frame_accumulator.fp
            confusion = source_count_confusions.setdefault(dataset, {})
            confusion_key = (reference_count, prediction_count)
            confusion[confusion_key] = confusion.get(confusion_key, 0) + 1
            # Create every requested slice for every receiver position so that
            # delete-one jackknife samples remain aligned, including positions
            # with no frames of a particular source count.
            grouped = samples_by_gt_source_count.setdefault(
                dataset, {count: {} for count in (1, 2, 3)}
            )
            for count in (1, 2, 3):
                grouped[count].setdefault(position, MetricAccumulator())
            if reference_count in grouped:
                merge_accumulator(
                    grouped[reference_count][position], frame_accumulator
                )
            # RT60 activity is reference-only and uses exactly the existing
            # strict reference ACCDOA threshold.  Merge the already-computed
            # frame statistics so matching is never repeated per bin.
            reference_norm = np.linalg.norm(references, axis=1)
            active_reference_slots = np.flatnonzero(
                reference_norm > reference_activity_threshold
            )
            if len(active_reference_slots):
                active_rt60_values = [
                    finite_decimal(
                        ordered[int(slot)][7],
                        f"rt60_s for active slot {int(slot)} in {path}, "
                        f"frame {current_frame}",
                    )
                    for slot in active_reference_slots
                ]
                # Values for inactive slots are intentionally never parsed or
                # validated and therefore cannot affect the frame condition.
                frame_rt60 = sum(active_rt60_values, Decimal(0)) / Decimal(
                    len(active_rt60_values)
                )
                bin_index = rt60_bin_index(frame_rt60, bin_width_decimal)
                rt60_accumulator = (
                    samples_by_rt60.setdefault(dataset, {})
                    .setdefault(bin_index, {})
                    .setdefault(position, MetricAccumulator())
                )
                merge_accumulator(rt60_accumulator, frame_accumulator)
                active_drr_values = (
                    [ordered[int(slot)][8] for slot in active_reference_slots]
                    if drr_column is not None
                    else None
                )
                # As for RT60, inactive-slot values are deliberately ignored.
                # If the column is absent, every non-empty frame is assigned
                # to the dedicated infinite-DRR category.
                drr_index = frame_drr_bin_index(
                    active_drr_values,
                    drr_bin_width_decimal,
                    f"drr_db for active slots in {path}, frame {current_frame}",
                )
                drr_accumulator = (
                    samples_by_drr.setdefault(dataset, {})
                    .setdefault(drr_index, {})
                    .setdefault(position, MetricAccumulator())
                )
                merge_accumulator(drr_accumulator, frame_accumulator)
            frame_rows = []

        for row_number, row in enumerate(reader, start=2):
            try:
                dataset = canonical_dataset_name(row[dataset_index].strip())
            except IndexError as error:
                raise ValueError(f"Short CSV row in {path} row {row_number}") from error
            if not dataset:
                raise ValueError(f"Empty prediction_dataset in {path} row {row_number}")
            if dataset in excluded_datasets:
                continue
            sample_id = row[columns["sample_id"]].strip()
            position = position_cache.get(sample_id)
            if position is None:
                position = receiver_position_key(sample_id, f"{path} row {row_number}")
                position_cache[sample_id] = position
            segment = row[columns["segment_idx"]].strip()
            frame = row[columns["frame_idx"]].strip()
            frame_key = (dataset, position, sample_id, segment, frame)
            if current_frame is not None and frame_key != current_frame:
                flush_frame()
            current_frame = frame_key
            try:
                frame_rows.append((
                    int(row[columns["slot"]]),
                    float(row[columns["accdoa_x"]]),
                    float(row[columns["accdoa_y"]]),
                    float(row[columns["accdoa_z"]]),
                    float(row[columns["prediction_accdoa_x"]]),
                    float(row[columns["prediction_accdoa_y"]]),
                    float(row[columns["prediction_accdoa_z"]]),
                    row[columns["rt60_s"]],
                    (
                        row[columns["drr_db"]]
                        if drr_column is not None
                        else "Infinity"
                    ),
                ))
            except (IndexError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid value in {path} row {row_number}") from error
        flush_frame()

    if not samples:
        raise ValueError(f"No included prediction rows found in {path}")
    return ModelData(
        label=label,
        path=path,
        samples=samples,
        samples_by_gt_source_count=samples_by_gt_source_count,
        samples_by_rt60=samples_by_rt60,
        rt60_bin_ranges={
            dataset: {index: (index, index + 1) for index in bins}
            for dataset, bins in samples_by_rt60.items()
        },
        samples_by_drr=samples_by_drr,
        drr_bin_ranges={
            dataset: {
                index: (index, index + 1)
                for index in bins
                if index != DRR_INFINITY_BIN
            }
            for dataset, bins in samples_by_drr.items()
        },
        source_count_confusions=source_count_confusions,
    )


def adaptive_rt60_ranges(
    bin_maps: Sequence[dict[int, dict[str, MetricAccumulator]]],
    minimum_positions: int,
) -> list[tuple[int, int]]:
    """Merge seed bins until every participating model has enough positions.

    A receiver position can contribute to multiple seed bins. Position sets
    are therefore unioned instead of adding per-bin counts, which would
    incorrectly inflate the number of independent jackknife units.
    """
    occupied_indices = sorted({index for bins in bin_maps for index in bins})
    if not occupied_indices:
        return []

    ranges: list[tuple[int, int]] = []
    start_index = occupied_indices[0]
    contributing_positions = [set() for _ in bin_maps]
    for index in occupied_indices:
        for model_positions, bins in zip(contributing_positions, bin_maps):
            model_positions.update(bins.get(index, {}))
        if all(
            len(model_positions) >= minimum_positions
            for model_positions in contributing_positions
        ):
            ranges.append((start_index, index + 1))
            start_index = index + 1
            contributing_positions = [set() for _ in bin_maps]

    if any(contributing_positions):
        tail_stop = occupied_indices[-1] + 1
        if ranges:
            previous_start, _previous_stop = ranges[-1]
            ranges[-1] = (previous_start, tail_stop)
        else:
            ranges.append((start_index, tail_stop))
    return ranges


def merge_rt60_seed_bins(
    bins: dict[int, dict[str, MetricAccumulator]],
    ranges: Sequence[tuple[int, int]],
) -> tuple[
    dict[int, dict[str, MetricAccumulator]],
    dict[int, tuple[int, int]],
]:
    """Reaggregate sufficient statistics without counting a position twice."""
    merged_bins: dict[int, dict[str, MetricAccumulator]] = {}
    merged_ranges: dict[int, tuple[int, int]] = {}
    for low_index, high_index in ranges:
        positions: dict[str, MetricAccumulator] = {}
        for seed_index in range(low_index, high_index):
            for position, accumulator in bins.get(seed_index, {}).items():
                merge_accumulator(
                    positions.setdefault(position, MetricAccumulator()),
                    accumulator,
                )
        if positions:
            # Preserve the original lower seed-bin index as a stable output id.
            merged_bins[low_index] = positions
            merged_ranges[low_index] = (low_index, high_index)
    return merged_bins, merged_ranges


def configure_rt60_bins(
    models: Sequence[ModelData],
    datasets: Sequence[str],
    strategy: str,
    minimum_positions: int,
) -> None:
    """Use shared real-dataset intervals and model-specific simulated intervals."""
    if strategy == "fixed":
        return
    for dataset in datasets:
        dataset_models = [model for model in models if dataset in model.samples]
        model_groups = (
            [[model] for model in dataset_models]
            if dataset == METRICS_ONLY_DATASET
            else [dataset_models]
        )
        for group in model_groups:
            ranges = adaptive_rt60_ranges(
                [model.samples_by_rt60.get(dataset, {}) for model in group],
                minimum_positions,
            )
            for model in group:
                merged_bins, merged_ranges = merge_rt60_seed_bins(
                    model.samples_by_rt60.get(dataset, {}), ranges
                )
                model.samples_by_rt60[dataset] = merged_bins
                model.rt60_bin_ranges[dataset] = merged_ranges


def configure_drr_bins(
    models: Sequence[ModelData],
    datasets: Sequence[str],
    strategy: str,
    minimum_positions: int,
) -> None:
    """Configure DRR bins with the same adaptive policy used for RT60."""
    if strategy == "fixed":
        return
    for dataset in datasets:
        dataset_models = [model for model in models if dataset in model.samples]
        model_groups = (
            [[model] for model in dataset_models]
            if dataset == METRICS_ONLY_DATASET
            else [dataset_models]
        )
        for group in model_groups:
            finite_bin_maps = [
                {
                    index: positions
                    for index, positions in model.samples_by_drr.get(
                        dataset, {}
                    ).items()
                    if index != DRR_INFINITY_BIN
                }
                for model in group
            ]
            ranges = adaptive_rt60_ranges(
                finite_bin_maps,
                minimum_positions,
            )
            for model, finite_bins in zip(group, finite_bin_maps):
                merged_bins, merged_ranges = merge_rt60_seed_bins(
                    finite_bins, ranges
                )
                infinity_positions = model.samples_by_drr.get(dataset, {}).get(
                    DRR_INFINITY_BIN
                )
                if infinity_positions:
                    # Infinite DRR is categorical and must never enlarge or be
                    # merged into the final finite adaptive interval.
                    merged_bins[DRR_INFINITY_BIN] = infinity_positions
                model.samples_by_drr[dataset] = merged_bins
                model.drr_bin_ranges[dataset] = merged_ranges


def validate_dataset_sets(models: Sequence[ModelData]) -> list[str]:
    reference = set(models[0].samples) - {METRICS_ONLY_DATASET}
    for model in models[1:]:
        current = set(model.samples) - {METRICS_ONLY_DATASET}
        if current != reference:
            raise ValueError(
                f"Dataset mismatch for {model.label}: "
                f"missing={sorted(reference-current)}, extra={sorted(current-reference)}"
            )
    all_datasets = set(reference)
    if any(METRICS_ONLY_DATASET in model.samples for model in models):
        all_datasets.add(METRICS_ONLY_DATASET)
    return sorted(all_datasets)


def validate_positions(models: Sequence[ModelData], dataset: str) -> list[str]:
    reference = set(models[0].samples[dataset])
    for model in models[1:]:
        current = set(model.samples[dataset])
        if current != reference:
            raise ValueError(
                f"Receiver-position mismatch for {model.label} in {dataset}: "
                f"missing={sorted(reference-current)[:5]}, "
                f"extra={sorted(current-reference)[:5]}"
            )
    if not reference:
        raise ValueError(f"Dataset {dataset!r} has no receiver positions")
    return sorted(reference)


def sum_samples(samples: Iterable[MetricAccumulator]) -> MetricAccumulator:
    total = MetricAccumulator()
    for sample in samples:
        total = total + sample
    return total


def confusion_matrix_fieldnames(maximum_source_count: int) -> tuple[str, ...]:
    """Return stable CSV columns for detection and source-count confusion."""
    detection_fields = tuple(
        f"{outcome}_{unit}"
        for outcome in (
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
        )
        for unit in ("count", "percent")
    )
    source_count_fields = tuple(
        f"confusion_gt_{ground_truth}_pred_{predicted}_{unit}"
        for ground_truth in range(maximum_source_count + 1)
        for predicted in range(maximum_source_count + 1)
        for unit in ("count", "percent")
    )
    return detection_fields + source_count_fields


def confusion_matrix_summary(
    accumulator: MetricAccumulator,
    source_count_confusion: dict[tuple[int, int], int],
    maximum_source_count: int,
    ground_truth_source_count: int | None = None,
) -> tuple[dict[str, int | float], dict[str, object]]:
    """Build flat CSV columns and structured, explicitly normalized matrices."""
    detection_counts = {
        "true_positive": accumulator.tp,
        "false_positive": accumulator.fp,
        "false_negative": accumulator.fn,
        "true_negative": accumulator.tn,
    }
    total_detection_outcomes = sum(detection_counts.values())
    csv_values: dict[str, int | float] = {}
    detection_percentages: dict[str, float] = {}
    for outcome, count in detection_counts.items():
        percentage = 100.0 * safe_divide(count, total_detection_outcomes)
        csv_values[f"{outcome}_count"] = count
        csv_values[f"{outcome}_percent"] = percentage
        detection_percentages[outcome] = percentage

    counts: list[list[int]] = []
    percentages: list[list[float]] = []
    source_counts = list(range(maximum_source_count + 1))
    for ground_truth in source_counts:
        count_row = [
            (
                source_count_confusion.get((ground_truth, predicted), 0)
                if ground_truth_source_count is None
                or ground_truth == ground_truth_source_count
                else 0
            )
            for predicted in source_counts
        ]
        row_total = sum(count_row)
        percentage_row = [
            100.0 * safe_divide(count, row_total) for count in count_row
        ]
        counts.append(count_row)
        percentages.append(percentage_row)
        for predicted, count, percentage in zip(
            source_counts, count_row, percentage_row
        ):
            prefix = f"confusion_gt_{ground_truth}_pred_{predicted}"
            csv_values[f"{prefix}_count"] = count
            csv_values[f"{prefix}_percent"] = percentage

    payload: dict[str, object] = {
        "event_detection_confusion_matrix": {
            "counts": detection_counts,
            "percentages": detection_percentages,
            "percentage_denominator": "all available source slots",
        },
        "source_count_confusion_matrix": {
            "ground_truth_source_counts": source_counts,
            "predicted_source_counts": source_counts,
            "counts": counts,
            "percentages": percentages,
            "percentage_normalization": "ground_truth_source_count_row",
        },
    }
    return csv_values, payload


def jackknife_from_full_and_loo(
    full_sample: float,
    leave_one_out: np.ndarray,
    alpha: float,
) -> JackknifeEstimate:
    count = len(leave_one_out)
    if count < 2 or not np.all(np.isfinite(leave_one_out)) or not math.isfinite(full_sample):
        raise ValueError("Jackknife requires at least two finite leave-one-out estimates")
    pseudo_values = count * full_sample - (count - 1) * leave_one_out
    estimate = float(np.mean(pseudo_values))
    mean_loo = float(np.mean(leave_one_out))
    # Standard delete-one jackknife SE:
    #
    #   sqrt((N - 1) / N * sum_i(theta_(i) - mean(theta_(.)))**2)
    #
    # This is also std(pseudo_values, ddof=1) / sqrt(N).  Do not divide the
    # leave-one-out sum of squares by N(N-1): that expression omits the
    # (N-1)^2 scaling induced by the pseudo-values and makes the SE exactly
    # N-1 times too small.
    standard_error = math.sqrt(
        ((count - 1) / count)
        * float(np.sum((leave_one_out - mean_loo) ** 2))
    )
    critical = float(student_t.ppf(1.0 - alpha / 2.0, df=count - 1))
    return JackknifeEstimate(
        full_sample=full_sample,
        estimate=estimate,
        standard_error=standard_error,
        ci_low=estimate - critical * standard_error,
        ci_high=estimate + critical * standard_error,
    )


def model_jackknife(
    model: ModelData,
    dataset: str,
    positions: Sequence[str],
    metric: str,
    alpha: float,
) -> JackknifeEstimate:
    total = sum_samples(model.samples[dataset][position] for position in positions)
    full = finalize_metrics(total)[metric]
    leave_one_out = np.asarray([
        finalize_metrics(total - model.samples[dataset][position])[metric]
        for position in positions
    ], dtype=np.float64)
    return jackknife_from_full_and_loo(full, leave_one_out, alpha)


def prepare_model_jackknife(
    model: ModelData,
    dataset: str,
    positions: Sequence[str],
    metrics: Sequence[str],
    alpha: float,
) -> ModelJackknifeData:
    """Compute every reusable model/position metric exactly once."""
    return prepare_accumulator_jackknife(
        model.samples[dataset], positions, metrics, alpha
    )


def prepare_accumulator_jackknife(
    samples_by_position: dict[str, MetricAccumulator],
    positions: Sequence[str],
    metrics: Sequence[str],
    alpha: float,
) -> ModelJackknifeData:
    """Compute jackknife metrics for one overall or source-count slice."""
    total = sum_samples(samples_by_position[position] for position in positions)
    full_metrics = finalize_metrics(total)
    loo_metric_rows = [
        finalize_metrics(total - samples_by_position[position])
        for position in positions
    ]
    leave_one_out = {
        metric: np.fromiter(
            (row[metric] for row in loo_metric_rows),
            dtype=np.float64,
            count=len(loo_metric_rows),
        )
        for metric in metrics
    }
    estimates = {
        metric: jackknife_from_full_and_loo(
            full_metrics[metric], leave_one_out[metric], alpha
        )
        for metric in metrics
    }
    return ModelJackknifeData(full_metrics, leave_one_out, estimates)


def prepare_all_model_jackknives(
    model: ModelData,
    positions_by_dataset: dict[str, list[str]],
    metrics: Sequence[str],
    alpha: float,
) -> tuple[
    str,
    dict[str, ModelJackknifeData],
    dict[str, dict[int, ModelJackknifeData]],
    dict[str, dict[int, BinnedJackknifeData]],
    dict[str, dict[int, BinnedJackknifeData]],
]:
    """Prepare every dataset and source-count slice in one per-model job."""
    prepared: dict[str, ModelJackknifeData] = {}
    prepared_by_source_count: dict[str, dict[int, ModelJackknifeData]] = {}
    prepared_by_rt60: dict[str, dict[int, BinnedJackknifeData]] = {}
    prepared_by_drr: dict[str, dict[int, BinnedJackknifeData]] = {}
    for dataset in sorted(model.samples):
        positions = positions_by_dataset.get(dataset)
        if positions is None:
            if dataset != METRICS_ONLY_DATASET:
                raise ValueError(f"Missing validated positions for dataset {dataset!r}")
            positions = sorted(model.samples[dataset])
        prepared[dataset] = prepare_model_jackknife(
            model, dataset, positions, metrics, alpha
        )
        prepared_by_source_count[dataset] = {
            count: prepare_accumulator_jackknife(
                model.samples_by_gt_source_count[dataset][count],
                positions,
                metrics,
                alpha,
            )
            for count in (1, 2, 3)
        }
        prepared_by_rt60[dataset] = {}
        for bin_index, bin_samples in sorted(
            model.samples_by_rt60.get(dataset, {}).items()
        ):
            bin_positions = sorted(bin_samples)
            total = sum_samples(bin_samples[position] for position in bin_positions)
            jackknife = None
            if len(bin_positions) >= 2:
                jackknife = prepare_accumulator_jackknife(
                    bin_samples, bin_positions, metrics, alpha
                )
            prepared_by_rt60[dataset][bin_index] = BinnedJackknifeData(
                full_metrics=finalize_metrics(total),
                number_of_receiver_positions=len(bin_positions),
                number_of_frames=total.num_frames,
                jackknife=jackknife,
            )
        prepared_by_drr[dataset] = {}
        for bin_index, bin_samples in sorted(
            model.samples_by_drr.get(dataset, {}).items(),
            key=lambda item: (
                item[0] == DRR_INFINITY_BIN,
                0 if item[0] == DRR_INFINITY_BIN else item[0],
            ),
        ):
            bin_positions = sorted(bin_samples)
            total = sum_samples(bin_samples[position] for position in bin_positions)
            jackknife = None
            if len(bin_positions) >= 2:
                jackknife = prepare_accumulator_jackknife(
                    bin_samples, bin_positions, metrics, alpha
                )
            prepared_by_drr[dataset][bin_index] = BinnedJackknifeData(
                full_metrics=finalize_metrics(total),
                number_of_receiver_positions=len(bin_positions),
                number_of_frames=total.num_frames,
                jackknife=jackknife,
            )
    return (
        model.label,
        prepared,
        prepared_by_source_count,
        prepared_by_rt60,
        prepared_by_drr,
    )


def compare_pair(
    model_a: ModelData,
    model_b: ModelData,
    dataset: str,
    positions: Sequence[str],
    metrics: Sequence[str],
    alpha: float,
    prepared_a: ModelJackknifeData | None = None,
    prepared_b: ModelJackknifeData | None = None,
) -> list[ComparisonRow]:
    prepared_a = prepared_a or prepare_model_jackknife(
        model_a, dataset, positions, metrics, alpha
    )
    prepared_b = prepared_b or prepare_model_jackknife(
        model_b, dataset, positions, metrics, alpha
    )
    return compare_prepared_pair(
        model_a.label,
        model_b.label,
        dataset,
        len(positions),
        metrics,
        alpha,
        prepared_a,
        prepared_b,
    )


def compare_prepared_pair(
    label_a: str,
    label_b: str,
    dataset: str,
    number_of_positions: int,
    metrics: Sequence[str],
    alpha: float,
    prepared_a: ModelJackknifeData,
    prepared_b: ModelJackknifeData,
) -> list[ComparisonRow]:
    """Compare prepared values without sending complete model data to workers."""
    rows: list[ComparisonRow] = []
    for metric in metrics:
        full_difference = (
            prepared_a.full_metrics[metric] - prepared_b.full_metrics[metric]
        )
        leave_one_out = (
            prepared_a.leave_one_out[metric] - prepared_b.leave_one_out[metric]
        )
        jackknife = jackknife_from_full_and_loo(full_difference, leave_one_out, alpha)
        if jackknife.standard_error == 0.0:
            p_value = 1.0 if jackknife.estimate == 0.0 else 0.0
        else:
            statistic = abs(jackknife.estimate / jackknife.standard_error)
            p_value = float(
                2.0 * student_t.sf(statistic, df=number_of_positions - 1)
            )
        estimate_a = prepared_a.estimates[metric]
        estimate_b = prepared_b.estimates[metric]
        significant = jackknife.ci_low > 0.0 or jackknife.ci_high < 0.0
        better_model = ""
        if significant and jackknife.estimate != 0.0:
            if metric in LOWER_IS_BETTER:
                better_model = label_a if jackknife.estimate < 0 else label_b
            else:
                better_model = label_a if jackknife.estimate > 0 else label_b
        rows.append(
            ComparisonRow(
                dataset=dataset,
                model_a=label_a,
                model_b=label_b,
                metric=metric,
                n_paired_positions=number_of_positions,
                estimate_a=estimate_a.estimate,
                estimate_b=estimate_b.estimate,
                difference_a_minus_b=jackknife.estimate,
                ci_low=jackknife.ci_low,
                ci_high=jackknife.ci_high,
                jackknife_standard_error=jackknife.standard_error,
                p_value=p_value,
                p_value_display=p_value,
                significant=significant,
                better_model=better_model,
            )
        )
    return rows


def compare_pair_batch(
    dataset: str,
    number_of_positions: int,
    metrics: Sequence[str],
    alpha: float,
    pairs: Sequence[
        tuple[str, str, ModelJackknifeData, ModelJackknifeData]
    ],
) -> list[ComparisonRow]:
    """Batch neighboring model pairs to keep multiprocessing overhead small."""
    rows: list[ComparisonRow] = []
    for label_a, label_b, prepared_a, prepared_b in pairs:
        rows.extend(
            compare_prepared_pair(
                label_a,
                label_b,
                dataset,
                number_of_positions,
                metrics,
                alpha,
                prepared_a,
                prepared_b,
            )
        )
    return rows


def adjust_p_values(p_values: Sequence[float], method: str) -> np.ndarray:
    """Return multiplicity-adjusted p-values in the original input order."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("p-values must be one-dimensional")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be finite and between zero and one")
    count = len(values)
    if count == 0 or method == "none":
        return values.copy()
    if method == "bonferroni":
        return np.minimum(values * count, 1.0)

    order = np.argsort(values, kind="stable")
    ordered = values[order]
    if method == "holm":
        adjusted_ordered = np.maximum.accumulate(
            (count - np.arange(count)) * ordered
        )
    elif method == "fdr_bh":
        ranks = np.arange(1, count + 1)
        adjusted_ordered = np.minimum.accumulate(
            (count * ordered / ranks)[::-1]
        )[::-1]
    else:
        raise ValueError(f"Unknown multiple-comparison correction: {method!r}")

    adjusted = np.empty(count, dtype=np.float64)
    adjusted[order] = np.minimum(adjusted_ordered, 1.0)
    return adjusted


def correction_family_key(row: ComparisonRow, scope: str) -> tuple[str, ...]:
    if scope == "dataset_metric":
        return row.dataset, row.metric
    if scope == "dataset":
        return (row.dataset,)
    if scope == "metric":
        return (row.metric,)
    if scope == "global":
        return ("all",)
    raise ValueError(f"Unknown correction scope: {scope!r}")


def apply_multiple_comparison_correction(
    rows: Sequence[ComparisonRow],
    method: str,
    scope: str,
    alpha: float,
) -> None:
    """Adjust p-values by family and update every significance-dependent field."""
    families: dict[tuple[str, ...], list[ComparisonRow]] = {}
    for row in rows:
        families.setdefault(correction_family_key(row, scope), []).append(row)

    for key, family in families.items():
        adjusted = adjust_p_values([row.p_value for row in family], method)
        family_label = f"{scope}:" + "/".join(key)
        for row, adjusted_p_value in zip(family, adjusted):
            row.adjusted_p_value = float(adjusted_p_value)
            row.correction_method = method
            row.correction_family = family_label
            row.p_value_display = row.adjusted_p_value
            row.significant = row.adjusted_p_value <= alpha
            row.better_model = ""
            if row.significant and row.difference_a_minus_b != 0.0:
                if row.metric in LOWER_IS_BETTER:
                    row.better_model = (
                        row.model_a if row.difference_a_minus_b < 0.0 else row.model_b
                    )
                else:
                    row.better_model = (
                        row.model_a if row.difference_a_minus_b > 0.0 else row.model_b
                    )


def json_number(value: float) -> float | None:
    return None if not math.isfinite(value) else float(value)


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pair_label(model_a: str, model_b: str) -> str:
    return f"{model_a} vs {model_b}"


def write_significance_matrix(
    path: Path,
    comparison_rows: Sequence[ComparisonRow],
    datasets: Sequence[str],
    metrics: Sequence[str],
    model_pairs: Sequence[tuple[str, str]],
) -> None:
    pair_columns = [pair_label(*pair) for pair in model_pairs]
    lookup = {
        (row.dataset, row.metric, row.model_a, row.model_b): row
        for row in comparison_rows
    }
    matrix_rows: list[dict[str, object]] = []
    for dataset in datasets:
        for metric in metrics:
            output: dict[str, object] = {"dataset": dataset, "metric": metric}
            for pair, column in zip(model_pairs, pair_columns):
                row = lookup[(dataset, metric, *pair)]
                output[column] = json_number(row.p_value_display)
            matrix_rows.append(output)
    write_csv(path, matrix_rows, ("dataset", "metric", *pair_columns))


def main() -> None:
    args = parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be between 0 and 1")
    if args.workers < 0:
        raise ValueError("--workers must be zero or positive")
    if not math.isfinite(args.rt60_bin_width) or args.rt60_bin_width <= 0.0:
        raise ValueError("--rt60-bin-width must be a positive finite number")
    if args.rt60_min_positions < 2:
        raise ValueError("--rt60-min-positions must be at least 2")
    if not math.isfinite(args.drr_bin_width) or args.drr_bin_width <= 0.0:
        raise ValueError("--drr-bin-width must be a positive finite number")
    if args.drr_min_positions < 2:
        raise ValueError("--drr-min-positions must be at least 2")

    specs = collect_model_specs(args)
    excluded = {
        canonical_dataset_name(value.strip())
        for value in args.exclude_dataset
        if value.strip()
    }


    maximum_useful_workers = len(specs)
    worker_count = min(
        args.workers or os.cpu_count() or 1,
        maximum_useful_workers,
    )
    if worker_count == 1:
        run_analysis(args, specs, excluded, worker_count, None)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            run_analysis(args, specs, excluded, worker_count, executor)


def run_analysis(
    args: argparse.Namespace,
    specs: Sequence[tuple[str, Path]],
    excluded: set[str],
    worker_count: int,
    executor: ProcessPoolExecutor | None,
) -> None:
    """Reuse one worker pool throughout independent CPU-intensive stages."""
    load_arguments = [
        (label, path, excluded, args.activity_threshold,
         args.reference_activity_threshold, args.rt60_bin_width,
         args.drr_bin_width)
        for label, path in specs
    ]
    if executor is None:
        models = [load_model(*values) for values in load_arguments]
    else:
        parsing_workers = min(worker_count, len(specs))
        print(f"Parsing {len(specs)} CSVs with {parsing_workers} worker processes...")
        futures = [executor.submit(load_model, *values) for values in load_arguments]
        # Resolve in input order so all output ordering remains deterministic.
        models = [future.result() for future in futures]
    datasets = validate_dataset_sets(models)
    positions_by_dataset = {
        dataset: validate_positions(models, dataset) for dataset in datasets
        if dataset != METRICS_ONLY_DATASET
    }
    configure_rt60_bins(
        models, datasets, args.rt60_binning, args.rt60_min_positions
    )
    configure_drr_bins(
        models, datasets, args.drr_binning, args.drr_min_positions
    )
    if executor is None:
        prepared_results = [
            prepare_all_model_jackknives(
                model, positions_by_dataset, args.metrics, args.alpha
            )
            for model in models
        ]
    else:
        print(
            f"Preparing jackknife estimates for {len(models)} models with "
            f"{min(worker_count, len(models))} worker processes..."
        )
        prepared_results = list(
            executor.map(
                prepare_all_model_jackknives,
                models,
                repeat(positions_by_dataset),
                repeat(args.metrics),
                repeat(args.alpha),
            )
        )
    prepared_by_model = {
        label: prepared for label, prepared, _, _, _ in prepared_results
    }
    source_count_prepared_by_model = {
        label: prepared_by_count
        for label, _, prepared_by_count, _, _ in prepared_results
    }
    rt60_prepared_by_model = {
        label: prepared_by_rt60
        for label, _, _, prepared_by_rt60, _ in prepared_results
    }
    drr_prepared_by_model = {
        label: prepared_by_drr
        for label, _, _, _, prepared_by_drr in prepared_results
    }
    significance_datasets = [
        dataset for dataset in datasets if dataset != METRICS_ONLY_DATASET
    ]
    maximum_source_count = max(
        3,
        *(
            source_count
            for model in models
            for confusion in model.source_count_confusions.values()
            for counts in confusion
            for source_count in counts
        ),
    )
    confusion_fields = confusion_matrix_fieldnames(maximum_source_count)
    labels = [model.label for model in models]
    model_pairs = list(combinations(labels, 2))
    comparisons: list[ComparisonRow] = []
    metric_rows: list[dict[str, object]] = []
    rt60_metric_rows: list[dict[str, object]] = []
    drr_metric_rows: list[dict[str, object]] = []
    dataset_payload: dict[str, object] = {}

    for dataset in datasets:
        dataset_models = [model for model in models if dataset in model.samples]
        positions = positions_by_dataset.get(dataset)
        comparison_enabled = dataset in significance_datasets
        if comparison_enabled:
            if positions is None:
                raise ValueError(f"Missing paired positions for dataset {dataset!r}")
            print(f"Dataset {dataset}: {len(positions)} paired receiver positions.")
        else:
            print(
                f"Dataset {dataset}: metrics only for "
                f"{len(dataset_models)} of {len(models)} models."
            )
        prepared = {
            model.label: prepared_by_model[model.label][dataset]
            for model in dataset_models
        }
        prepared_by_source_count = {
            model.label: source_count_prepared_by_model[model.label][dataset]
            for model in dataset_models
        }
        prepared_by_rt60 = {
            model.label: rt60_prepared_by_model[model.label][dataset]
            for model in dataset_models
        }
        prepared_by_drr = {
            model.label: drr_prepared_by_model[model.label][dataset]
            for model in dataset_models
        }
        dataset_comparisons: list[ComparisonRow] = []
        if comparison_enabled:
            pair_arguments = [
                (
                    model_a.label,
                    model_b.label,
                    prepared[model_a.label],
                    prepared[model_b.label],
                )
                for model_a, model_b in combinations(dataset_models, 2)
            ]
            for label_a, label_b, _, _ in pair_arguments:
                print(f"  Comparing {label_a} vs {label_b}...")
            if executor is not None and len(pair_arguments) >= 2 * worker_count:
                batch_size = math.ceil(len(pair_arguments) / worker_count)
                futures = [
                    executor.submit(
                        compare_pair_batch,
                        dataset,
                        len(positions),
                        args.metrics,
                        args.alpha,
                        pair_arguments[start : start + batch_size],
                    )
                    for start in range(0, len(pair_arguments), batch_size)
                ]
                for future in futures:
                    dataset_comparisons.extend(future.result())
            else:
                dataset_comparisons.extend(
                    compare_pair_batch(
                        dataset,
                        len(positions),
                        args.metrics,
                        args.alpha,
                        pair_arguments,
                    )
                )
        comparisons.extend(dataset_comparisons)

        models_payload: dict[str, object] = {}
        for model in dataset_models:
            model_positions = (
                positions if positions is not None else sorted(model.samples[dataset])
            )
            source_count_confusion = model.source_count_confusions[dataset]
            total_accumulator = sum_samples(
                model.samples[dataset][position] for position in model_positions
            )
            overall_confusion_values, overall_confusion_payload = (
                confusion_matrix_summary(
                    total_accumulator,
                    source_count_confusion,
                    maximum_source_count,
                )
            )
            metrics_payload: dict[str, object] = {}
            for metric in args.metrics:
                estimate = prepared[model.label].estimates[metric]
                metrics_payload[metric] = {
                    key: json_number(value)
                    for key, value in asdict(estimate).items()
                }
                metric_rows.append(
                    {
                        "dataset": dataset,
                        "model": model.label,
                        "gt_source_count": "all",
                        "metric": metric,
                        "number_of_receiver_positions": len(model_positions),
                        **asdict(estimate),
                        **overall_confusion_values,
                    }
                )
            by_source_count_payload: dict[str, object] = {}
            confusion_by_source_count_payload: dict[str, object] = {}
            for count in (1, 2, 3):
                source_count_accumulator = sum_samples(
                    model.samples_by_gt_source_count[dataset][count][position]
                    for position in model_positions
                )
                count_confusion_values, count_confusion_payload = (
                    confusion_matrix_summary(
                        source_count_accumulator,
                        source_count_confusion,
                        maximum_source_count,
                        count,
                    )
                )
                count_metrics_payload: dict[str, object] = {}
                for metric in args.metrics:
                    estimate = prepared_by_source_count[model.label][count].estimates[
                        metric
                    ]
                    count_metrics_payload[metric] = {
                        key: json_number(value)
                        for key, value in asdict(estimate).items()
                    }
                    metric_rows.append(
                        {
                            "dataset": dataset,
                            "model": model.label,
                            "gt_source_count": count,
                            "metric": metric,
                            "number_of_receiver_positions": len(model_positions),
                            **asdict(estimate),
                            **count_confusion_values,
                        }
                    )
                by_source_count_payload[str(count)] = count_metrics_payload
                confusion_by_source_count_payload[str(count)] = (
                    count_confusion_payload
                )
            by_rt60_payload: dict[str, object] = {}
            for bin_index, bin_prepared in prepared_by_rt60[model.label].items():
                low_index, high_index = model.rt60_bin_ranges[dataset][bin_index]
                low, high, center = rt60_bin_bounds(
                    low_index, high_index, args.rt60_bin_width
                )
                bin_width = float(
                    Decimal(str(args.rt60_bin_width))
                    * Decimal(high_index - low_index)
                )
                minimum_positions_met = (
                    bin_prepared.number_of_receiver_positions
                    >= args.rt60_min_positions
                )
                binning_scope = (
                    "fixed"
                    if args.rt60_binning == "fixed"
                    else (
                        "model_specific"
                        if dataset == METRICS_ONLY_DATASET
                        else "shared_dataset"
                    )
                )
                bin_metrics_payload: dict[str, object] = {}
                for metric in args.metrics:
                    estimate = (
                        bin_prepared.jackknife.estimates[metric]
                        if bin_prepared.jackknife is not None
                        else None
                    )
                    metric_payload = {
                        "full_sample": json_number(
                            bin_prepared.full_metrics[metric]
                        ),
                        "estimate": (
                            json_number(estimate.estimate) if estimate else None
                        ),
                        "standard_error": (
                            json_number(estimate.standard_error) if estimate else None
                        ),
                        "ci_low": json_number(estimate.ci_low) if estimate else None,
                        "ci_high": json_number(estimate.ci_high) if estimate else None,
                        "jackknife_valid": estimate is not None,
                    }
                    bin_metrics_payload[metric] = metric_payload
                    rt60_metric_rows.append(
                        {
                            "dataset": dataset,
                            "model": model.label,
                            "metric": metric,
                            "rt60_bin_index": bin_index,
                            "rt60_bin_low": low,
                            "rt60_bin_high": high,
                            "rt60_bin_center": center,
                            "rt60_bin_width": bin_width,
                            "rt60_binning_scope": binning_scope,
                            "rt60_min_positions_met": minimum_positions_met,
                            "number_of_receiver_positions": (
                                bin_prepared.number_of_receiver_positions
                            ),
                            "number_of_frames": bin_prepared.number_of_frames,
                            **metric_payload,
                        }
                    )
                by_rt60_payload[str(bin_index)] = {
                    "rt60_bin_index": bin_index,
                    "rt60_bin_low": low,
                    "rt60_bin_high": high,
                    "rt60_bin_center": center,
                    "rt60_bin_width": bin_width,
                    "rt60_binning_scope": binning_scope,
                    "rt60_min_positions_met": minimum_positions_met,
                    "number_of_receiver_positions": (
                        bin_prepared.number_of_receiver_positions
                    ),
                    "number_of_frames": bin_prepared.number_of_frames,
                    "metrics": bin_metrics_payload,
                }
            by_drr_payload: dict[str, object] = {}
            for bin_index, bin_prepared in prepared_by_drr[model.label].items():
                if bin_index == DRR_INFINITY_BIN:
                    low = high = center = bin_width = DRR_INFINITY_BIN
                else:
                    low_index, high_index = model.drr_bin_ranges[dataset][bin_index]
                    low, high, center = drr_bin_bounds(
                        low_index, high_index, args.drr_bin_width
                    )
                    bin_width = float(
                        Decimal(str(args.drr_bin_width))
                        * Decimal(high_index - low_index)
                    )
                minimum_positions_met = (
                    bin_prepared.number_of_receiver_positions
                    >= args.drr_min_positions
                )
                binning_scope = (
                    "fixed"
                    if args.drr_binning == "fixed"
                    else (
                        "model_specific"
                        if dataset == METRICS_ONLY_DATASET
                        else "shared_dataset"
                    )
                )
                bin_metrics_payload: dict[str, object] = {}
                for metric in args.metrics:
                    estimate = (
                        bin_prepared.jackknife.estimates[metric]
                        if bin_prepared.jackknife is not None
                        else None
                    )
                    metric_payload = {
                        "full_sample": json_number(
                            bin_prepared.full_metrics[metric]
                        ),
                        "estimate": (
                            json_number(estimate.estimate) if estimate else None
                        ),
                        "standard_error": (
                            json_number(estimate.standard_error) if estimate else None
                        ),
                        "ci_low": json_number(estimate.ci_low) if estimate else None,
                        "ci_high": json_number(estimate.ci_high) if estimate else None,
                        "jackknife_valid": estimate is not None,
                    }
                    bin_metrics_payload[metric] = metric_payload
                    drr_metric_rows.append(
                        {
                            "dataset": dataset,
                            "model": model.label,
                            "metric": metric,
                            "drr_bin_index": bin_index,
                            "drr_bin_low_db": low,
                            "drr_bin_high_db": high,
                            "drr_bin_center_db": center,
                            "drr_bin_width_db": bin_width,
                            "drr_binning_scope": binning_scope,
                            "drr_min_positions_met": minimum_positions_met,
                            "number_of_receiver_positions": (
                                bin_prepared.number_of_receiver_positions
                            ),
                            "number_of_frames": bin_prepared.number_of_frames,
                            **metric_payload,
                        }
                    )
                by_drr_payload[str(bin_index)] = {
                    "drr_bin_index": bin_index,
                    "drr_bin_low_db": low,
                    "drr_bin_high_db": high,
                    "drr_bin_center_db": center,
                    "drr_bin_width_db": bin_width,
                    "drr_binning_scope": binning_scope,
                    "drr_min_positions_met": minimum_positions_met,
                    "number_of_receiver_positions": (
                        bin_prepared.number_of_receiver_positions
                    ),
                    "number_of_frames": bin_prepared.number_of_frames,
                    "metrics": bin_metrics_payload,
                }
            models_payload[model.label] = {
                "prediction_file": str(model.path),
                "number_of_receiver_positions": len(model_positions),
                "metrics": metrics_payload,
                "metrics_by_gt_source_count": by_source_count_payload,
                "metrics_by_rt60": by_rt60_payload,
                "metrics_by_drr": by_drr_payload,
                **overall_confusion_payload,
                "confusion_matrices_by_gt_source_count": (
                    confusion_by_source_count_payload
                ),
            }
        dataset_payload[dataset] = {
            "number_of_paired_receiver_positions": (
                len(positions) if positions is not None else None
            ),
            "number_of_models_with_data": len(dataset_models),
            "included_in_significance_comparison": comparison_enabled,
            "rt60_binning_scope": (
                "fixed"
                if args.rt60_binning == "fixed"
                else (
                    "model_specific"
                    if dataset == METRICS_ONLY_DATASET
                    else "shared_dataset"
                )
            ),
            "drr_binning_scope": (
                "fixed"
                if args.drr_binning == "fixed"
                else (
                    "model_specific"
                    if dataset == METRICS_ONLY_DATASET
                    else "shared_dataset"
                )
            ),
            "models": models_payload,
        }

    apply_multiple_comparison_correction(
        comparisons,
        args.multiple_comparison_correction,
        args.correction_scope,
        args.alpha,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_dicts = [asdict(row) for row in comparisons]
    write_csv(
        args.output_dir / "model_metrics.csv",
        metric_rows,
        (
            "dataset",
            "model",
            "gt_source_count",
            "metric",
            "number_of_receiver_positions",
            "full_sample",
            "estimate",
            "standard_error",
            "ci_low",
            "ci_high",
            *confusion_fields,
        ),
    )
    write_csv(
        args.output_dir / "model_metrics_by_rt60.csv",
        rt60_metric_rows,
        (
            "dataset",
            "model",
            "metric",
            "rt60_bin_index",
            "rt60_bin_low",
            "rt60_bin_high",
            "rt60_bin_center",
            "rt60_bin_width",
            "rt60_binning_scope",
            "rt60_min_positions_met",
            "number_of_receiver_positions",
            "number_of_frames",
            "full_sample",
            "estimate",
            "standard_error",
            "ci_low",
            "ci_high",
            "jackknife_valid",
        ),
    )
    write_csv(
        args.output_dir / "model_metrics_by_drr.csv",
        drr_metric_rows,
        (
            "dataset",
            "model",
            "metric",
            "drr_bin_index",
            "drr_bin_low_db",
            "drr_bin_high_db",
            "drr_bin_center_db",
            "drr_bin_width_db",
            "drr_binning_scope",
            "drr_min_positions_met",
            "number_of_receiver_positions",
            "number_of_frames",
            "full_sample",
            "estimate",
            "standard_error",
            "ci_low",
            "ci_high",
            "jackknife_valid",
        ),
    )
    write_csv(
        args.output_dir / "pairwise_significance.csv",
        comparison_dicts,
        tuple(ComparisonRow.__dataclass_fields__),
    )
    write_significance_matrix(
        args.output_dir / "significance_matrix.csv",
        comparisons,
        significance_datasets,
        args.metrics,
        model_pairs,
    )
    payload = {
        "method": {
            "input": "raw augmented prediction CSVs from the evaluation script",
            "resampling_unit": "unique room and receiver position parsed from sample_id",
            "rt60": {
                "input_column": "rt60_s",
                "frame_value": (
                    "mean rt60_s of active ground-truth sources using the "
                    "reference ACCDOA activity threshold"
                ),
                "inactive_slots_used": False,
                "predicted_activity_used": False,
                "zero_reference_frames_binned": False,
                "all_metrics_use_frame_rt60_including_LE": True,
                "binning_strategy": args.rt60_binning,
                "bin_width_seconds": args.rt60_bin_width,
                "seed_bin_width_seconds": args.rt60_bin_width,
                "minimum_positions_per_adaptive_bin": args.rt60_min_positions,
                "bins": (
                    "half-open intervals [low, high), indexed by their first "
                    "seed-bin index"
                ),
                "adaptive_bin_construction": (
                    "merge adjacent occupied seed-bin ranges until each "
                    "participating model reaches the minimum number of unique "
                    "room/receiver positions; merge an undersized final range "
                    "into the previous range"
                    if args.rt60_binning == "adaptive"
                    else "report each fixed-width seed bin unchanged"
                ),
                "real_dataset_boundaries_shared_across_models": True,
                "simulated_dataset_boundaries_model_specific": (
                    args.rt60_binning == "adaptive"
                ),
                "position_counts_are_unique_within_merged_bins": True,
                "resampling_unit": (
                    "unique room and receiver position with frames in that bin"
                ),
                "minimum_positions_for_jackknife": 2,
                "paired_significance_comparisons_performed": False,
                "confidence_intervals_computed_independently_per_model": True,
            },
            "drr": {
                "input_column": "drr_db",
                "input_column_optional": True,
                "missing_column_frame_value": "inf",
                "values_above_db_treated_as_infinity": 35.0,
                "infinity_bin_is_separate_from_finite_adaptive_bins": True,
                "frame_value": (
                    "mean drr_db of active ground-truth sources using the "
                    "reference ACCDOA activity threshold"
                ),
                "inactive_slots_used": False,
                "predicted_activity_used": False,
                "zero_reference_frames_binned": False,
                "all_metrics_use_frame_drr_including_LE": True,
                "binning_strategy": args.drr_binning,
                "bin_width_db": args.drr_bin_width,
                "seed_bin_width_db": args.drr_bin_width,
                "minimum_positions_per_adaptive_bin": args.drr_min_positions,
                "bins": (
                    "half-open intervals [low, high), indexed by their first "
                    "seed-bin index"
                ),
                "adaptive_bin_construction": (
                    "merge adjacent occupied seed-bin ranges until each "
                    "participating model reaches the minimum number of unique "
                    "room/receiver positions; merge an undersized final range "
                    "into the previous range"
                    if args.drr_binning == "adaptive"
                    else "report each fixed-width seed bin unchanged"
                ),
                "real_dataset_boundaries_shared_across_models": True,
                "simulated_dataset_boundaries_model_specific": (
                    args.drr_binning == "adaptive"
                ),
                "position_counts_are_unique_within_merged_bins": True,
                "resampling_unit": (
                    "unique room and receiver position with frames in that bin"
                ),
                "minimum_positions_for_jackknife": 2,
                "paired_significance_comparisons_performed": False,
                "confidence_intervals_computed_independently_per_model": True,
            },
            "point_estimate": "bias-corrected jackknife pseudo-value mean",
            "confidence_interval": (
                "Student-t jackknife interval using N-1 degrees of freedom, "
                "following Equations (1)-(7) of Mesaros et al."
            ),
            "paired_significance": (
                "adjusted two-sided jackknife t p-value is less than or equal "
                "to alpha; reported confidence intervals remain unadjusted"
            ),
            "p_value": (
                "raw two-sided Student-t p-value from jackknife estimate / "
                "standard error"
            ),
            "adjusted_p_value": (
                f"{args.multiple_comparison_correction} correction with "
                f"{args.correction_scope} families"
            ),
            "multiple_testing": {
                "method": args.multiple_comparison_correction,
                "scope": args.correction_scope,
            },
            "excluded_datasets": sorted(excluded),
            "metrics_only_datasets": (
                [METRICS_ONLY_DATASET] if METRICS_ONLY_DATASET in datasets else []
            ),
            "confusion_matrices": {
                "event_detection_percentage_normalization": (
                    "all available source slots"
                ),
                "source_count_percentage_normalization": (
                    "each ground-truth source-count row"
                ),
                "maximum_source_count": maximum_source_count,
            },
            "alpha": args.alpha,
            "activity_threshold": args.activity_threshold,
            "reference_activity_threshold": args.reference_activity_threshold,
        },
        "number_of_models": len(models),
        "number_of_datasets": len(datasets),
        "number_of_significance_datasets": len(significance_datasets),
        "datasets": dataset_payload,
        "pairwise_comparisons": [
            {
                key: json_number(value) if isinstance(value, float) else value
                for key, value in row.items()
            }
            for row in comparison_dicts
        ],
    }
    with (args.output_dir / "significance_results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)

    print(f"Saved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
