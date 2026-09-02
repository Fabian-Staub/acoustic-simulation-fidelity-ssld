import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from math import gcd
import os
import logging
import sys
import resource
import time
import json
import pickle
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfar as pf
import pyroomacoustics as pra
from pyroomacoustics.directivities import MeasuredDirectivityFile, Rotation3D
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample_poly, fftconvolve, find_peaks
from scipy.special import spherical_jn, spherical_yn, sph_harm
from tqdm import tqdm

from SphericalHarmonic import HOA_array, SphericalHarmonicDirectivity


# The stochastic late-reverberation level is estimated from this internal
# high-order ISM simulation.  The exported early ISM still uses ``ism_order``.
LATE_REFERENCE_ISM_ORDER = 12

# Configure logging.
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
log_file_path = os.path.join(
    logs_dir,
    f"rir_generation_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}.log",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file_path)],
)

# --- Global Constants ---
# Lookup table for materials based on pyroomacoustics examples/defaults
LOOKUP_TABLE = [
    "hard_surface",
    "brickwork",
    "rough_concrete",
    "unpainted_concrete",
    "rough_lime_wash",
    "smooth_brickwork_flush_pointing",
    "smooth_brickwork_10mm_pointing",
    "brick_wall_rough",
    "ceramic_tiles",
    "limestone_wall",
    "reverb_chamber",
    "plasterboard",
    "wooden_lining",
    "glass_3mm",
    "glass_window",
    "double_glazing_30mm",
    "double_glazing_10mm",
    "wood_1.6cm",
    "curtains_cotton_0.5",
    "curtains_0.2",
    "curtains_velvet",
    "curtains_glass_mat",
    "carpet_cotton",
    "carpet_6mm_closed_cell_foam",
    "carpet_6mm_open_cell_foam",
    "carpet_tufted_9m",
    "felt_5mm",
    "carpet_hairy",
    "concrete_floor",
    "marble_floor",
    "orchestra_1.5_m2",
    "panel_fabric_covered_6pcf",
    "panel_fabric_covered_8pcf",
    "ceiling_fibre_abosrber",
]

# Categorized material choices for different surfaces
WALLS = np.concatenate((LOOKUP_TABLE[:22], LOOKUP_TABLE[30:]))
FLOOR = np.concatenate((LOOKUP_TABLE[7:8], LOOKUP_TABLE[22:31]))
CEIL = np.concatenate((LOOKUP_TABLE[3:5], LOOKUP_TABLE[30:]))

# --- Utility Functions ---

def get_random_dimensions(
    x_range=(2.5, 20.1, 0.5),
    y_range=(2.5, 20.1, 0.5),
    z_range=(2.5, 5.1, 0.5),
) -> tuple:
    """
    Generate random room dimensions with independent ranges per axis.

    Args:
        x_range: (min, max, step) for x-dimension
        y_range: (min, max, step) for y-dimension
        z_range: (min, max, step) for z-dimension

    Returns:
        tuple: (Lx, Ly, Lz)
    """

    x_vals = np.arange(*x_range)
    y_vals = np.arange(*y_range)
    z_vals = np.arange(*z_range)

    return tuple(
        np.random.choice(values, size=1)
        for values in (x_vals, y_vals, z_vals)
    )
    #return (np.array([13.5]), np.array([19.5]), np.array([6.5]))

'''
def get_random_dimensions(
    x_range=(1.5, 20),
    y_range=(1.5, 20),
    z_range=(2.5, 7.5),
    step=0.5,
    ab=(1.2, 2.5)   # Beta shape (a, b)
):
    a, b = ab

    def sample_beta(low, high):
        u = np.random.beta(a, b)
        val = low + u * (high - low)
        val = np.round(val / step) * step
        return np.clip(val, low, high)

    def sample_exp(low, high, k=1.2):
        # inverse transform sampling for truncated exponential
        u = np.random.rand()

        # CDF inversion for shifted exponential:
        val = low - (1.0 / k) * np.log(1 - u * (1 - np.exp(-k * (high - low))))

        val = np.round(val / step) * step
        return np.clip(val, low, high)

    Lx = sample_beta(*x_range)
    Ly = sample_beta(*y_range)

    Lz = sample_exp(*z_range, k=0.8)  # adjust k for steepness

    return np.array([Lx]), np.array([Ly]), np.array([Lz])


def generate_frequency_absorption(centerFreqs,
                                  a=2.5,
                                  b_min=1.1,
                                  b_max=2.5,
                                  low=0.01,
                                  high=0.99,
                                  seed=None):
    """
    Generate absorption values across frequency bands where
    higher frequencies have higher Beta 'b' parameter -> lower absorption.

    Parameters
    ----------
    centerFreqs : array-like
        Frequency band centers (only length is used).
    a : float
        Fixed Beta 'a' parameter.
    b_min : float
        Beta parameter at lowest frequency.
    b_max : float
        Beta parameter at highest frequency.
    low, high : float
        Clipping bounds.
    seed : int or None
        Random seed.

    Returns
    -------
    np.ndarray
        Absorption values per frequency band.
    """
    rng = np.random.default_rng(seed)
    n = centerFreqs.shape[0]

    # normalized frequency axis: 0 -> low freq, 1 -> high freq
    t = np.linspace(0, 1, n)

    # interpolate b from b_min (low freq) to b_max (high freq)
    b = b_min + t * (b_max - b_min)

    absorption = rng.beta(a, b, size=n)

    return np.clip(absorption, low, high)
'''

def generate_frequency_absorption(centerFreqs,
                                  a_values,
                                  b_values,
                                  low=0.01,
                                  high=0.99,
                                  seed=None):
    """
    Generate absorption values using manually specified Beta
    distribution parameters for each frequency band.

    Parameters
    ----------
    centerFreqs : array-like
        Frequency band centers.
    a_values : array-like
        Beta 'a' parameter for each frequency band.
    b_values : array-like
        Beta 'b' parameter for each frequency band.
    low, high : float
        Clipping bounds.
    seed : int or None
        Random seed.

    Returns
    -------
    np.ndarray
        Absorption values per frequency band.
    """
    rng = np.random.default_rng(seed)

    centerFreqs = np.asarray(centerFreqs)
    a_values = np.asarray(a_values)
    b_values = np.asarray(b_values)

    n = len(centerFreqs)

    if len(a_values) != n:
        raise ValueError(
            f"a_values must have length {n}, got {len(a_values)}"
        )

    if len(b_values) != n:
        raise ValueError(
            f"b_values must have length {n}, got {len(b_values)}"
        )

    absorption = rng.beta(a_values, b_values)

    return np.clip(absorption, low, high)


def _normalize_floor_polygon(floor_corners: np.ndarray) -> np.ndarray:
    """Return floor corners as an ``(n_corners, 2)`` float array."""
    polygon = np.asarray(floor_corners, dtype=float)

    if polygon.ndim != 2:
        raise ValueError("floor_corners must be a two-dimensional array.")
    if polygon.shape[0] == 2:
        polygon = polygon.T
    elif polygon.shape[1] != 2:
        raise ValueError("floor_corners must have shape (2, n) or (n, 2).")

    if len(polygon) < 3 or not np.all(np.isfinite(polygon)):
        raise ValueError("floor_corners must contain at least three finite points.")

    return polygon


def _point_is_inside_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Test whether a 2-D point is inside a polygon using ray casting."""
    x, y = point
    inside = False
    previous = polygon[-1]

    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            x_intersection = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_intersection:
                inside = not inside
        previous = current

    return inside


def _distance_to_polygon_walls(point: np.ndarray, polygon: np.ndarray) -> float:
    """Return the shortest distance from a point to any polygon edge."""
    edge_starts = polygon
    edge_ends = np.roll(polygon, -1, axis=0)
    edges = edge_ends - edge_starts
    edge_lengths_squared = np.sum(edges * edges, axis=1)

    # Repeated corners do not define a wall, but handling them here keeps the
    # distance calculation well-defined for imperfect input polygons.
    projection = np.zeros(len(edges), dtype=float)
    nonzero_edges = edge_lengths_squared > 0.0
    projection[nonzero_edges] = np.sum(
        (point - edge_starts[nonzero_edges]) * edges[nonzero_edges], axis=1
    ) / edge_lengths_squared[nonzero_edges]
    projection = np.clip(projection, 0.0, 1.0)

    closest_points = edge_starts + projection[:, None] * edges
    return float(np.min(np.linalg.norm(point - closest_points, axis=1)))


def get_random_positions(
    Lx: float,
    Ly: float,
    Lz: float,
    min_dist_from_wall: float = 0.5,
    step: float = 0.05,
    floor_corners: np.ndarray = None,
) -> tuple:
    """
    Generate random positions for the source and microphone in the room.

    Args:
        Lx: Room length along x-axis.
        Ly: Room length along y-axis.
        Lz: Room length along z-axis.
        min_dist_from_wall: Minimum distance of source/mic from any side wall.
        step: Step size for position choices.
        floor_corners: Optional polygonal floor corners with shape ``(2, n)``
            or ``(n, 2)``. If omitted, the room is treated as a shoebox.

    Returns:
        tuple: Random source and receiver positions (Sx, Sy, Sz, Rx, Ry, Rz)
               as numpy arrays (size 1).
    """
    if min_dist_from_wall < 0.0:
        raise ValueError("min_dist_from_wall must be non-negative.")
    if step <= 0.0:
        raise ValueError("step must be positive.")

    polygon = None
    if floor_corners is None:
        x_pos_range = np.arange(min_dist_from_wall, Lx - min_dist_from_wall, step)
        y_pos_range = np.arange(min_dist_from_wall, Ly - min_dist_from_wall, step)
    else:
        polygon = _normalize_floor_polygon(floor_corners)
        # Sample over the polygon's actual bounding box. Candidate points are
        # subsequently checked against all (including slanted) side walls.
        x_pos_range = np.arange(
            np.min(polygon[:, 0]), np.max(polygon[:, 0]) + step * 0.5, step
        )
        y_pos_range = np.arange(
            np.min(polygon[:, 1]), np.max(polygon[:, 1]) + step * 0.5, step
        )
    z_pos_range = np.arange(1, Lz - 1, step)

    # Check if valid positions are possible
    if len(x_pos_range) == 0 or len(y_pos_range) == 0 or len(z_pos_range) == 0:
        raise ValueError(f"Room dimensions ({Lx}, {Ly}, {Lz}) too small for minimum distance {min_dist_from_wall}.")

    def sample_xy():
        if polygon is None:
            return (
                float(np.random.choice(x_pos_range)),
                float(np.random.choice(y_pos_range)),
            )

        for _ in range(10000):
            point = np.array(
                [np.random.choice(x_pos_range), np.random.choice(y_pos_range)]
            )
            if (
                _point_is_inside_polygon(point, polygon)
                and _distance_to_polygon_walls(point, polygon)
                >= min_dist_from_wall
            ):
                return float(point[0]), float(point[1])

        raise ValueError(
            "Could not find a position inside the polygonal room with minimum "
            f"wall distance {min_dist_from_wall}."
        )

    Sx_value, Sy_value = sample_xy()
    Rx_value, Ry_value = sample_xy()
    Sx, Sy = np.array([Sx_value]), np.array([Sy_value])
    Rx, Ry = np.array([Rx_value]), np.array([Ry_value])
    Sz = np.random.choice(z_pos_range, size=1)
    Rz = np.random.choice(z_pos_range, size=1)

    # Ensure source and receiver are not at the exact same position
    while np.allclose([Rx, Ry, Rz], [Sx, Sy, Sz]):
        Rx_value, Ry_value = sample_xy()
        Rx, Ry = np.array([Rx_value]), np.array([Ry_value])
        Rz = np.random.choice(z_pos_range, size=1)

    return Sx, Sy, Sz, Rx, Ry, Rz
    #return (np.array([5.0]), np.array([5.0]), np.array([5.0]), np.array([3.0]), np.array([5.0]), np.array([5.0]))

def doa_unit_vector(src, mic):
    """Return the world-frame unit vector pointing from microphone to source."""
    src = np.asarray(src, dtype=float).reshape(3)
    mic = np.asarray(mic, dtype=float).reshape(3)
    v = src - mic
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Source and microphone positions must be different.")
    return v / norm


def doa_in_mic_frame(src, mic, mic_orientation_az_rad):
    """Compute ACCDOA, azimuth, and elevation in the microphone-local frame.

    The microphone orientation is a yaw rotation about +z. Therefore, the
    world-frame DOA is rotated by the inverse yaw (-orientation) so that a
    source directly in front of the rotated microphone has azimuth 0 deg.
    """
    doa_world = doa_unit_vector(src, mic)

    c = np.cos(mic_orientation_az_rad)
    s = np.sin(mic_orientation_az_rad)
    world_to_mic = np.array([
        [ c,  s, 0.0],
        [-s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    accdoa = world_to_mic @ doa_world

    # Guard against small numerical drift before inverse trigonometric use.
    accdoa = accdoa / np.linalg.norm(accdoa)
    az_rad = np.arctan2(accdoa[1], accdoa[0])
    el_rad = np.arctan2(accdoa[2], np.hypot(accdoa[0], accdoa[1]))

    return accdoa, np.degrees(az_rad), np.degrees(el_rad)

def is_too_close(new_dir, dirs, min_angle_deg):
    if dirs is None or len(dirs) == 0:
        return False

    dirs = np.asarray(dirs)

    # ensure shape is (N, 3)
    if dirs.ndim == 1:
        dirs = dirs.reshape(1, -1)

    cos_thr = np.cos(np.deg2rad(min_angle_deg))
    dots = dirs @ new_dir

    return np.any(dots >= cos_thr)


def get_random_materials() -> dict:
    """
    Randomly choose materials for the room surfaces from predefined lists.

    Returns:
        dict: Material dictionary with random choices for walls, ceiling, and floor.
    """
    return {
        "east": np.random.choice(WALLS),
        "west": np.random.choice(WALLS),
        "north": np.random.choice(WALLS),
        "south": np.random.choice(WALLS),
        "ceiling": np.random.choice(CEIL),
        "floor": np.random.choice(FLOOR),
    }

def get_room_range(num_rooms: int):
    """
    Determine which rooms this SLURM array task should process.
    If not running in SLURM array mode, process all rooms.
    """

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    rooms_per_task = num_rooms // task_count
    remainder = num_rooms % task_count

    start = task_id * rooms_per_task + min(task_id, remainder)
    end = start + rooms_per_task + (1 if task_id < remainder else 0)

    return start, end


def inverse_eyring(rt60, room, c=343.0):
    """Estimate mean absorption for any 3D pyroomacoustics room."""

    rt60 = np.maximum(np.asarray(rt60, dtype=float), 1e-6)

    V = room.get_volume()
    S = sum(room.wall_area(wall) for wall in room.walls)

    return 1.0 - np.exp(
        -(24.0 * np.log(10.0) * V) / (c * S * rt60)
    )


def get_room_geometry_metadata(room_dims, floor_corners, shoebox):
    """Return serializable room geometry metadata in metres."""
    length, width, height = map(float, room_dims)

    if shoebox:
        floor = np.array(
            [[0.0, 0.0], [length, 0.0], [length, width], [0.0, width]]
        )
    else:
        floor = np.asarray(floor_corners, dtype=float).T

    floor_3d = np.column_stack((floor, np.zeros(len(floor))))
    ceiling_3d = floor_3d.copy()
    ceiling_3d[:, 2] = height

    return {
        "is_shoebox": bool(shoebox),
        "room_corners": np.vstack((floor_3d, ceiling_3d)).tolist(),
    }


def metadata_row(result):
    """Remove the RIR and serialize nested values for CSV output."""
    row = {key: value for key, value in result.items() if key != "rir"}
    if "room_corners" in row:
        row["room_corners"] = json.dumps(row["room_corners"], separators=(",", ":"))
    return row


def create_ambisonic_array(order_of_ambisonics: int, sample_rate: int, azimuth_rotation: float = 0.0):
    """
    Creates an Ambisonic microphone array definition.

    Args:
        order_of_ambisonics: The desired order of the Ambisonic array.
        sample_rate: The sample rate for the microphone array definition.

    Returns:
        tuple: A tuple containing:
               - mic_positions.T (numpy array): Transposed microphone positions.
               - sample_rate (int): The sample rate.
               - microphone_directivities (list): List of SphericalHarmonicDirectivity objects.
    """
    mic_radius = 0.000001 # Effectively a point mic array at the center
    order = order_of_ambisonics
    # Number of microphones required for a given Ambisonic order
    samples = (order + 1) ** 2

    # HOA_array expects radius=1 for generating positions on a unit sphere
    mic_positions, orientations, degrees = HOA_array(
        samples=samples,
        radius=1,
        n_order=order,
        azimuth_rotation=azimuth_rotation,
    )

    # Scale positions by the actual mic radius
    mic_positions = mic_positions * mic_radius

    microphone_directivities = []
    for i in range(samples):
        orientation = orientations[i]
        # Spherical Harmonic Directivity requires n and m parameters
        directivity = SphericalHarmonicDirectivity(
            orientation,
            n=degrees[i][0],
            m=degrees[i][1],
            azimuth_rotation=azimuth_rotation,
        )
        microphone_directivities.append(directivity)

    return mic_positions.T, sample_rate, microphone_directivities


def _azi_elev_to_cartesian(azi_elev, radius):
    """Convert [azimuth, elevation] directions to 3-D Cartesian positions."""
    azi_elev = np.asarray(azi_elev, dtype=float)
    az = azi_elev[:, 0]
    el = azi_elev[:, 1]
    cos_el = np.cos(el)
    return radius * np.column_stack((
        cos_el * np.cos(az),
        cos_el * np.sin(az),
        np.sin(el),
    ))


def _rotate_positions_about_z(positions, yaw_rad):
    """Rotate an (N, 3) position matrix about +z."""
    c_yaw = np.cos(yaw_rad)
    s_yaw = np.sin(yaw_rad)
    rotation = np.array([
        [c_yaw, -s_yaw, 0.0],
        [s_yaw,  c_yaw, 0.0],
        [0.0,    0.0,   1.0],
    ])
    return np.asarray(positions) @ rotation.T


# Per-process caches. Each ProcessPool worker loads the measured archive once and
# reuses EM32 encoding filters for repeated FFT lengths.
_EM32_DATA_CACHE = {}
_EM32_SHT_FILTER_CACHE = {}


def _get_worker_array(
    array_type,
    ambi_order,
    sample_rate,
    array_azimuth_rad,
    em32_directivity_path,
    em32_radius,
):
    """Build a rotated microphone array inside a worker process."""
    if array_type == "em32":
        array = create_em32_array(
            em32_directivity_path=em32_directivity_path,
            sample_rate=sample_rate,
            azimuth_rotation=array_azimuth_rad,
            radius=em32_radius,
        )
    else:
        array = create_ambisonic_array(
            order_of_ambisonics=ambi_order,
            sample_rate=sample_rate,
            azimuth_rotation=array_azimuth_rad,
        )

    return array


def _load_em32_npz(npz_path, requested_fs=None, radius=0.042):
    """Load, validate, resample, and cache the Eigenmike EM32 archive."""
    abs_path = os.path.abspath(os.fspath(npz_path))
    cache_key = (abs_path, None if requested_fs is None else int(requested_fs), float(radius))
    cached = _EM32_DATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    data = np.load(abs_path, allow_pickle=False)
    required = {
        "fs", "h_mics", "measurement_area_weights",
        "measurement_dirs_aziElev", "eigen_dirs_aziElev",
    }
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"Missing EM32 NPZ arrays: {sorted(missing)}")

    fs_file = int(np.asarray(data["fs"]).item())
    h_mics = np.asarray(data["h_mics"], dtype=float)
    measurement_dirs = np.asarray(data["measurement_dirs_aziElev"], dtype=float)
    eigen_dirs = np.asarray(data["eigen_dirs_aziElev"], dtype=float)
    weights = np.asarray(data["measurement_area_weights"], dtype=float).reshape(-1)

    # Stored archive convention supplied by the user: taps x 32 microphones x directions.
    if h_mics.ndim != 3:
        raise ValueError(f"h_mics must be 3-D, got {h_mics.shape}")
    if h_mics.shape[1] == 32 and h_mics.shape[2] == measurement_dirs.shape[0]:
        pass
    elif h_mics.shape[0] == measurement_dirs.shape[0] and h_mics.shape[1] == 32:
        h_mics = np.transpose(h_mics, (2, 1, 0))
    else:
        raise ValueError(
            "Could not identify h_mics axes. Expected (taps, 32, directions), "
            f"got {h_mics.shape} with {measurement_dirs.shape[0]} directions."
        )

    if eigen_dirs.shape != (32, 2):
        raise ValueError(f"eigen_dirs_aziElev must have shape (32, 2), got {eigen_dirs.shape}")
    if weights.size != measurement_dirs.shape[0]:
        raise ValueError("measurement_area_weights and measurement directions differ in length")

    fs_out = fs_file if requested_fs is None else int(requested_fs)
    if fs_out != fs_file:
        g = gcd(fs_out, fs_file)
        h_mics = resample_poly(h_mics, fs_out // g, fs_file // g, axis=0)

    # Normalize quadrature weights to integrate over 4*pi steradians.
    weight_sum = np.sum(weights)
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("Invalid measurement_area_weights")
    weights = weights * (4.0 * np.pi / weight_sum)

    result = {
        "fs": fs_out,
        "h_mics": np.ascontiguousarray(h_mics),
        "measurement_area_weights": np.ascontiguousarray(weights),
        "measurement_dirs_aziElev": np.ascontiguousarray(measurement_dirs),
        "eigen_dirs_aziElev": np.ascontiguousarray(eigen_dirs),
        "radius": float(radius),
    }
    _EM32_DATA_CACHE[cache_key] = result
    return result


def _em32_npz_reader(path, fs=None, radius=0.042):
    """pyroomacoustics reader callback for Eigenmike_em32_IRs.npz."""
    em32 = _load_em32_npz(path, requested_fs=fs, radius=radius)

    # MeasuredDirectivityFile expects: directions x microphones x taps.
    impulse_responses = np.transpose(em32["h_mics"], (2, 1, 0))

    directions = em32["measurement_dirs_aziElev"]
    source_locs = np.vstack((
        directions[:, 0],
        np.pi / 2.0 - directions[:, 1],
        np.ones(directions.shape[0]),
    ))
    mic_locs = _azi_elev_to_cartesian(
        em32["eigen_dirs_aziElev"], em32["radius"]
    ).T

    source_labels = [str(i) for i in range(source_locs.shape[1])]
    mic_labels = [f"EM32_{i + 1:02d}" for i in range(mic_locs.shape[1])]
    return impulse_responses, em32["fs"], source_locs, mic_locs, source_labels, mic_labels


def create_em32_array(
    em32_directivity_path,
    sample_rate,
    azimuth_rotation=0.0,
    radius=0.042,
    interp_order=None,
    placement_radius=1e-6,
):
    """
    Create an EM32 array from a SOFA or NPZ directivity file.

    The measured directivity impulse responses already contain the relative
    delays between the EM32 capsules. Therefore, the microphone coordinates
    returned to pyroomacoustics are placed effectively at the same point,
    preventing pyroomacoustics from adding those spatial delays again.

    Parameters
    ----------
    radius : float
        Physical EM32 radius used when reading/interpreting the measured data.
        This should normally remain 0.042 metres.
    placement_radius : float
        Effective array radius used by the room simulation. A very small
        nonzero value avoids exactly coincident microphone coordinates.
    """
    em32_path = os.fspath(em32_directivity_path)
    extension = os.path.splitext(em32_path)[1].lower()

    if placement_radius < 0:
        raise ValueError("placement_radius must be non-negative.")

    if extension == ".npz":
        em32 = _load_em32_npz(
            em32_path,
            requested_fs=sample_rate,
            radius=radius,
        )

        # Capsule directions are retained, but their simulated radius is tiny.
        position_directions = _azi_elev_to_cartesian(
            em32["eigen_dirs_aziElev"],
            radius=1.0,
        )

        reader = partial(_em32_npz_reader, radius=radius)
        measured_file = MeasuredDirectivityFile(
            path=em32_path,
            fs=sample_rate,
            interp_order=interp_order,
            file_reader_callback=reader,
        )

    elif extension == ".sofa":
        # The SOFA reader retains the physical metadata and the measured
        # impulse responses, including their relative delays.
        measured_file = MeasuredDirectivityFile(
            path=em32_path,
            fs=sample_rate,
            interp_order=interp_order,
        )

        physical_positions = np.stack(
            [
                np.asarray(
                    measured_file.get_mic_position(i),
                    dtype=float,
                ).reshape(3)
                for i in range(32)
            ],
            axis=0,
        )

        if physical_positions.shape != (32, 3):
            raise ValueError(
                "Expected 32 three-dimensional EM32 positions, "
                f"but received shape {physical_positions.shape}."
            )

        # Convert the physical coordinates into unit capsule directions.
        norms = np.linalg.norm(physical_positions, axis=1, keepdims=True)
        if np.any(norms <= np.finfo(float).eps):
            raise ValueError(
                "SOFA file contains a microphone position whose direction "
                "cannot be determined."
            )

        position_directions = physical_positions / norms

    else:
        raise ValueError(
            f"Unsupported EM32 directivity file format {extension!r}. "
            "Expected a '.npz' or '.sofa' file."
        )

    if position_directions.shape != (32, 3):
        raise ValueError(
            "Expected 32 three-dimensional EM32 position directions, "
            f"but received shape {position_directions.shape}."
        )

    # Coordinates used only by the room propagation model.
    positions = position_directions * placement_radius
    positions = _rotate_positions_about_z(positions, azimuth_rotation)

    # Rotate the measured directivities by the same array yaw.
    orientation = Rotation3D(
        [azimuth_rotation, 0.0, 0.0],
        rot_order="zyx",
        degrees=False,
    )
    directivities = [
        measured_file.get_mic_directivity(i, orientation)
        for i in range(32)
    ]

    return positions.T, sample_rate, directivities


# Eigenmike capsule directions: [azimuth, elevation] in degrees.
# These are used only for the theoretical rigid-sphere SHT encoder. The measured
# NPZ remains responsible for the microphone directivities used by the room model.
MIC_DIRS_DEG = np.array(
    [
        [0, 21], [32, 0], [0, -21], [328, 0],
        [0, 58], [45, 35], [69, 0], [45, -35],
        [0, -58], [315, -35], [291, 0], [315, 35],
        [91, 69], [90, 32], [90, -31], [89, -69],
        [180, 21], [212, 0], [180, -21], [148, 0],
        [180, 58], [225, 35], [249, 0], [225, -35],
        [180, -58], [135, -35], [111, 0], [135, 35],
        [269, 69], [270, 32], [270, -32], [271, -69],
    ],
    dtype=np.float64,
)


def acn_index(n: int, m: int) -> int:
    """Return the ACN channel index for degree n and order m."""
    return n * n + n + m


def complex_spherical_harmonic(n, m, azimuth, inclination):
    """Complex orthonormal spherical harmonic using SciPy's angle convention."""
    return sph_harm(m, n, azimuth, inclination)


def real_spherical_harmonics(max_order, azimuth, inclination):
    """Evaluate real ACN/N3D spherical harmonics."""
    azimuth = np.asarray(azimuth, dtype=np.float64)
    inclination = np.asarray(inclination, dtype=np.float64)
    if azimuth.shape != inclination.shape:
        raise ValueError("azimuth and inclination must have identical shapes")

    y = np.zeros((azimuth.size, (max_order + 1) ** 2), dtype=np.float64)
    for n in range(max_order + 1):
        for m in range(-n, n + 1):
            channel = acn_index(n, m)
            if m < 0:
                yc = complex_spherical_harmonic(n, -m, azimuth, inclination)
                y[:, channel] = np.sqrt(2.0) * ((-1.0) ** m) * np.imag(yc)
            elif m == 0:
                yc = complex_spherical_harmonic(n, 0, azimuth, inclination)
                y[:, channel] = np.real(yc)
            else:
                yc = complex_spherical_harmonic(n, m, azimuth, inclination)
                y[:, channel] = np.sqrt(2.0) * ((-1.0) ** m) * np.real(yc)

    return np.sqrt(4.0 * np.pi) * y


def rigid_sphere_modal_coefficients(max_order, kr):
    """Rigid-sphere modal coefficients used by the standalone EM32 converter."""
    kr = np.asarray(kr, dtype=np.float64)
    coefficients = np.zeros((kr.size, max_order + 1), dtype=np.complex128)
    nonzero = kr > 1e-12
    z = kr[nonzero]
    coefficients[0, 0] = 1.0

    for n in range(max_order + 1):
        if z.size == 0:
            continue
        jn = spherical_jn(n, z)
        yn = spherical_yn(n, z)
        jn_derivative = spherical_jn(n, z, derivative=True)
        yn_derivative = spherical_yn(n, z, derivative=True)
        hn = jn - 1j * yn
        hn_derivative = jn_derivative - 1j * yn_derivative
        rigid_response = jn - (jn_derivative / hn_derivative) * hn
        coefficients[nonzero, n] = (1j ** n) * rigid_response

    return coefficients


def repeat_per_order(values):
    """Repeat each order-dependent value for its 2*n+1 ACN channels."""
    values = np.asarray(values)
    return np.concatenate([
        np.repeat(values[n], 2 * n + 1) for n in range(values.shape[-1])
    ])


def generate_sht_filters_regls(
    radius,
    mic_dirs_az_el_rad,
    sht_order,
    filter_length,
    sample_rate,
    max_gain_db,
    speed_of_sound=343.0,
):
    """Generate regularized rigid-sphere EM32-to-SH filters.

    This is the same theoretical regLS conversion used in the standalone
    converter. It does not read h_mics from em32_directivity_path.
    """
    mic_dirs_az_el_rad = np.asarray(mic_dirs_az_el_rad, dtype=np.float64)
    if mic_dirs_az_el_rad.ndim != 2 or mic_dirs_az_el_rad.shape[1] != 2:
        raise ValueError("Microphone directions must have shape (microphones, 2)")
    if filter_length <= 0 or filter_length % 2 != 0:
        raise ValueError("filter_length must be a positive even integer")

    n_mics = mic_dirs_az_el_rad.shape[0]
    maximum_supported_order = int(np.floor(np.sqrt(n_mics) - 1))
    if sht_order > maximum_supported_order:
        logging.warning(
            "Requested SH order %d is too high for %d microphones; using %d.",
            sht_order, n_mics, maximum_supported_order,
        )
        sht_order = maximum_supported_order

    frequencies = np.arange(filter_length // 2 + 1) * sample_rate / filter_length
    kr = 2.0 * np.pi * frequencies * radius / speed_of_sound
    array_order = min(30, int(np.floor(2.0 * kr[-1])))

    azimuth = mic_dirs_az_el_rad[:, 0]
    elevation = mic_dirs_az_el_rad[:, 1]
    inclination = np.pi / 2.0 - elevation
    y_array = real_spherical_harmonics(array_order, azimuth, inclination)
    modal_coefficients = rigid_sphere_modal_coefficients(array_order, kr)

    n_output_sh = (sht_order + 1) ** 2
    h_frequency = np.zeros(
        (n_output_sh, n_mics, frequencies.size), dtype=np.complex128
    )
    alpha = 10.0 ** (max_gain_db / 20.0)
    beta = 1.0 / (2.0 * alpha)
    regularization = (beta ** 2) * np.eye(n_mics)

    for frequency_index in range(frequencies.size):
        repeated_modal = repeat_per_order(modal_coefficients[frequency_index])
        h_array = y_array * repeated_modal[np.newaxis, :]
        h_array_truncated = h_array[:, :n_output_sh]
        system_matrix = h_array @ h_array.conj().T + regularization
        solution = np.linalg.solve(system_matrix, h_array_truncated)
        h_frequency[:, :, frequency_index] = solution.conj().T

    full_spectrum = np.concatenate(
        [h_frequency, np.conj(h_frequency[:, :, -2:0:-1])], axis=2
    )
    full_spectrum[:, :, filter_length // 2] = np.abs(
        full_spectrum[:, :, filter_length // 2]
    )
    h_time = np.real(np.fft.ifft(full_spectrum, axis=2))
    h_time = np.fft.fftshift(h_time, axes=2)
    return h_frequency, h_time


def get_em32_sht_filters(
    order_sht,
    filter_length,
    sample_rate,
    amp_threshold=15.0,
    radius=0.042,
):
    """Create and cache the standalone-script-compatible EM32 SHT FIR bank."""
    key = (
        int(order_sht), int(filter_length), int(sample_rate),
        float(amp_threshold), float(radius),
    )
    cached = _EM32_SHT_FILTER_CACHE.get(key)
    if cached is not None:
        return cached

    _, filters_time = generate_sht_filters_regls(
        radius=radius,
        mic_dirs_az_el_rad=np.deg2rad(MIC_DIRS_DEG),
        sht_order=order_sht,
        filter_length=filter_length,
        sample_rate=sample_rate,
        max_gain_db=amp_threshold,
    )
    filters_time = np.ascontiguousarray(filters_time)
    _EM32_SHT_FILTER_CACHE[key] = filters_time
    return filters_time


def encode_em32_to_sh(microphone_signals, filters_time, preserve_length=True):
    """Apply the centered FIR bank exactly as in the standalone converter.

    Parameters use the standalone layout: microphone_signals is samples x mics;
    the returned array is samples x SH channels.
    """
    microphone_signals = np.asarray(microphone_signals, dtype=np.float64)
    filters_time = np.asarray(filters_time, dtype=np.float64)
    if microphone_signals.ndim != 2:
        raise ValueError("microphone_signals must have shape (samples, microphones)")
    if filters_time.ndim != 3:
        raise ValueError("filters_time must have shape (SH, microphones, taps)")

    n_samples, n_input_mics = microphone_signals.shape
    n_sh, n_filter_mics, filter_length = filters_time.shape
    if n_input_mics != n_filter_mics:
        raise ValueError(
            f"Input has {n_input_mics} microphone channels, but filters expect "
            f"{n_filter_mics}."
        )

    output_length = n_samples if preserve_length else n_samples + filter_length - 1
    sh_signals = np.zeros((output_length, n_sh), dtype=np.float64)
    for sh_index in range(n_sh):
        accumulated = np.zeros(n_samples + filter_length - 1, dtype=np.float64)
        for mic_index in range(n_input_mics):
            accumulated += fftconvolve(
                microphone_signals[:, mic_index],
                filters_time[sh_index, mic_index, :],
                mode="full",
            )
        sh_signals[:, sh_index] = accumulated[:output_length]
    return sh_signals


def convert_em32_to_sh(
    microphone_signals,
    order_sht,
    sample_rate,
    amp_threshold=15.0,
    radius=0.042,
    filter_length=1024,
):
    """Encode EM32 channels to real ACN/N3D SH using theoretical regLS filters.

    em32_directivity_path is intentionally not used here: the NPZ contains measured
    microphone directional responses, not precomputed microphone-to-SH filters.
    """
    signals = np.asarray(microphone_signals)
    if signals.ndim != 2 or signals.shape[0] != 32:
        raise ValueError(f"Expected EM32 signals with shape (32, samples), got {signals.shape}")

    filters_time = get_em32_sht_filters(
        order_sht=order_sht,
        filter_length=filter_length,
        sample_rate=sample_rate,
        amp_threshold=amp_threshold,
        radius=radius,
    )
    sh = encode_em32_to_sh(
        microphone_signals=signals.T,
        filters_time=filters_time,
        preserve_length=False,
    )
    return sh.T

def getMixingTimeEstimateFromVolume(volume):
    # Cremer, L.; Mueller, H. A. (1978): Die wissenschaftlichen Grundlagen der Raumakustik
    #return 2 * np.sqrt(volume) / 1000.0
    return (0.0117 * volume + 50.1) / 1000



def addStochasticLateReverbPyfar(
    srir,
    rt,
    fs,
    volume,
    late_reference_srir,
    centerFreqs=None,
    num_fractions=1,
    filter_order=14,
    peak_threshold=0.1,
    peak_prominence=0.01,
    ism_order=None,
    filter_margin_s=0.05,
    random_seed=None,
):
    """Add a frequency-dependent stochastic late tail to an RIR.

    ``srir`` is the requested-order ISM that supplies the exported early RIR.
    ``late_reference_srir`` is the mandatory high-order ISM used only to
    estimate late energy. Both inputs and all returned RIRs have shape
    ``(channels, samples)``.

    One shared gain is derived by matching the stochastic omni-channel energy
    to the late energy of channel 0 in ``late_reference_srir``. That gain is
    applied to all independently generated Ambisonics channels. The reference
    RIR itself is never mixed into the returned RIR.

    The stochastic-noise envelope starts decaying at the detected direct-sound
    peak, independently of the later mixing point used for the crossfade.

    ``volume`` is retained for call-site compatibility. The diagnostics dict
    contains only values consumed by ``save_rir_visualization``.
    """
    srir = np.asarray(srir, dtype=float)
    if srir.ndim == 1:
        srir = srir[np.newaxis, :]
    if srir.ndim != 2 or srir.shape[1] == 0:
        raise ValueError(
            f"srir must have shape (channels, samples), got {srir.shape}"
        )
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be finite and positive, got {fs}")

    rt = np.asarray(rt, dtype=float).reshape(-1)
    if rt.size == 0 or np.any(~np.isfinite(rt)) or np.any(rt <= 0):
        raise ValueError("rt must contain finite values greater than zero.")

    if ism_order is not None:
        order_value = float(ism_order)
        if (
            not np.isfinite(order_value)
            or order_value < 0
            or not order_value.is_integer()
        ):
            raise ValueError("ism_order must be a non-negative integer or None.")
        ism_order = int(order_value)

    filter_margin_s = float(filter_margin_s)
    if not np.isfinite(filter_margin_s) or filter_margin_s < 0:
        raise ValueError("filter_margin_s must be finite and non-negative.")

    # Work internally with samples x channels.
    srir = srir.T
    _, num_channels = srir.shape

    late_reference_srir = np.asarray(late_reference_srir, dtype=float)
    if late_reference_srir.ndim == 1:
        late_reference_srir = late_reference_srir[np.newaxis, :]
    if (
        late_reference_srir.ndim != 2
        or late_reference_srir.shape[0] != num_channels
        or late_reference_srir.shape[1] == 0
    ):
        raise ValueError(
            "late_reference_srir must have the same channel count as srir "
            f"and shape (channels, samples), got {late_reference_srir.shape}"
        )
    late_reference_srir = late_reference_srir.T

    rng = np.random.default_rng(random_seed)
    _ = volume  # Kept only for compatibility with the existing caller.

    if centerFreqs is not None:
        centerFreqs = np.asarray(centerFreqs, dtype=float).reshape(-1)
        if centerFreqs.size != rt.size:
            raise ValueError("centerFreqs and rt must have the same length.")
        valid = (
            np.isfinite(centerFreqs)
            & (centerFreqs > 0)
            & (centerFreqs <= float(fs) / 2.0)
        )
        centerFreqs = centerFreqs[valid]
        rt = rt[valid]
        if centerFreqs.size == 0:
            raise ValueError("No valid center frequencies remain below Nyquist.")
        frequency_range = (float(centerFreqs[0]), float(centerFreqs[-1]))
    else:
        frequency_range = (20.0, float(fs) / 2.0)

    peak_signal = np.abs(srir[:, 0])
    peak_max = float(np.max(peak_signal))
    if not np.isfinite(peak_max) or peak_max <= 0:
        first_peak_sample = 0
    else:
        peaks, _ = find_peaks(
            peak_signal,
            height=peak_threshold * peak_max,
            prominence=peak_prominence * peak_max,
        )
        first_peak_sample = int(peaks[0] if peaks.size else np.argmax(peak_signal))

    #if ism_order is None:
    #    transition_s = 0.080
    #else:
    #    transition_s = float(
    #        np.interp(ism_order, [0.0, 2.0, 12.0], [0.0035, 0.080, 0.080])
    #    )
    requested_fade_samples = max(1, int(round(0.010 * float(fs))))
    
    if ism_order == 0:
        transition_s = 0.0035
    else:
        transition_s = getMixingTimeEstimateFromVolume(volume) - requested_fade_samples / (2* float(fs))

    transition_samples = max(0, int(round(transition_s * float(fs))))
    mixing_sample = first_peak_sample + transition_samples
    decay_length = int(np.ceil(float(np.max(rt)) * float(fs)))
    filter_margin_samples = int(np.ceil(filter_margin_s * float(fs)))
    required_length = (
        mixing_sample
        + decay_length
        + requested_fade_samples
        + filter_margin_samples
    )
    if srir.shape[0] < required_length:
        srir = np.pad(srir, ((0, required_length - srir.shape[0]), (0, 0)))

    if late_reference_srir.shape[0] < required_length:
        late_reference_srir = np.pad(
            late_reference_srir,
            ((0, required_length - late_reference_srir.shape[0]), (0, 0)),
        )

    num_samples = srir.shape[0]
    fade_samples = min(requested_fade_samples, num_samples - mixing_sample)

    early_window = np.ones(num_samples, dtype=float)
    late_window = np.zeros(num_samples, dtype=float)
    if fade_samples == 1:
        early_window[mixing_sample] = 0.0
        late_window[mixing_sample] = 1.0
    elif fade_samples > 1:
        phase = np.linspace(0.0, np.pi / 2.0, fade_samples)
        fade_end = mixing_sample + fade_samples
        early_window[mixing_sample:fade_end] = np.cos(phase)
        late_window[mixing_sample:fade_end] = np.sin(phase)
    fade_end = mixing_sample + fade_samples
    early_window[fade_end:] = 0.0
    late_window[fade_end:] = 1.0

    noise_signal = pf.Signal(
        rng.normal(0.0, 1.0, size=(num_channels, num_samples)), fs
    )
    bands = np.asarray(
        pf.dsp.filter.fractional_octave_bands(
            noise_signal,
            num_fractions=num_fractions,
            frequency_range=frequency_range,
            order=filter_order,
        ).time
    )
    if bands.ndim != 3 or bands.shape[-1] != num_samples:
        raise ValueError(f"Unexpected pyfar filter-bank output shape: {bands.shape}.")

    expected_bands = rt.size
    if bands.shape[:2] == (expected_bands, num_channels):
        noise_by_band = np.transpose(bands, (2, 1, 0))
    elif bands.shape[:2] == (num_channels, expected_bands):
        noise_by_band = np.transpose(bands, (2, 0, 1))
    else:
        raise ValueError(
            f"Expected {expected_bands} bands and {num_channels} channels, "
            f"but pyfar returned {bands.shape}."
        )

    sample_index = np.arange(num_samples, dtype=float)
    decay_samples = np.maximum(
        sample_index - first_peak_sample,
        0.0,
    )
    tau_samples = rt * float(fs) / 6.9078
    for band_index, tau in enumerate(tau_samples):
        noise_by_band[:, :, band_index] *= np.exp(
            -decay_samples / tau
        )[:, np.newaxis]
    noise = np.sum(noise_by_band, axis=2)

    input_sample_energy = np.sum(srir**2, axis=1)
    input_schroeder = np.cumsum(input_sample_energy[::-1])[::-1]
    if input_schroeder[0] > 0:
        input_schroeder_db = 10.0 * np.log10(
            np.maximum(input_schroeder / input_schroeder[0], 1e-12)
        )
    else:
        input_schroeder_db = np.zeros(num_samples, dtype=float)

    windowed_noise = noise * late_window[:, np.newaxis]
    reference_end = min(num_samples, late_reference_srir.shape[0])
    omni_late_energy = float(
        np.sum(late_reference_srir[mixing_sample:reference_end, 0] ** 2)
    )
    omni_noise_energy = float(
        np.sum(windowed_noise[mixing_sample:, 0] ** 2)
    )
    shared_scale = (
        np.sqrt(omni_late_energy / omni_noise_energy)
        if omni_noise_energy > 0
        else 0.0
    )
    scale = np.full(num_channels, shared_scale, dtype=float)

    rir_early = srir * early_window[:, np.newaxis]
    late_reverb = noise * scale[np.newaxis, :] * late_window[:, np.newaxis]
    rir = rir_early + late_reverb

    diagnostics = {
        "rt_profile": rt.copy(),
        "centerFreqs": None if centerFreqs is None else centerFreqs.copy(),
        "mixingTime_sample": mixing_sample,
        "actual_fade_samples": fade_samples,
        "ism_energy_per_sample": input_sample_energy,
        "rir_early_energy_per_sample": np.sum(rir_early**2, axis=1),
        "late_reverb_energy_per_sample": np.sum(late_reverb**2, axis=1),
        "schroeder_db": input_schroeder_db,
    }

    return rir.T, rir_early.T, late_reverb.T, diagnostics



def random_rt_profile_with_long_band(
        rt_profiles,
        min_rt=1.0,
        rng=None):

    """
    Randomly select an RT60 profile where at least one
    frequency band has RT60 >= min_rt.

    Parameters
    ----------
    rt_profiles : ndarray
        Shape: (num_profiles, num_bands)

    min_rt : float
        Minimum RT60 threshold.

    Returns
    -------
    rt_profile : ndarray
        Selected RT60 vector.
    """

    if rng is None:
        rng = np.random.default_rng()

    rt_profiles = np.asarray(rt_profiles)

    # Find profiles containing at least one long RT band
    valid = np.any(
        rt_profiles >= min_rt,
        axis=1
    )

    candidates = rt_profiles[valid]

    if len(candidates) == 0:
        raise ValueError(
            f"No RT profile contains a band >= {min_rt}s"
        )

    idx = rng.integers(
        len(candidates)
    )

    return candidates[idx]



def apply_tikhonov_difference_filters(
    signal,
    fs,
    R=0.042,
    Nmic=32,
    amp_threshold=15,
    c=343.0
):
    
    """
    Apply order-dependent Moreau/Daniel difference filters to Ambisonic signals.

    Parameters
    ----------
    signal : ndarray
        Ambisonic signal with shape (channels, samples).
    fs : float
        Sampling rate in Hz.
    R : float
        Microphone array radius in meters.
    Nmic : int
        Number of microphones.
    amp_threshold : float
        Regularization threshold in dB.
    c : float
        Speed of sound in m/s.

    Returns
    -------
    filtered : ndarray
        Filtered Ambisonic signal with shape (channels, samples).
    """

    signal = signal.T
    signal = np.asarray(signal)
    dtype_in = signal.dtype
    signal = signal.astype(np.float64)

    if signal.ndim != 2:
        raise ValueError("signal must have shape (channels, samples).")

    n_samples, n_channels = signal.shape

    order_sht = int(np.sqrt(n_channels) - 1)

    if (order_sht + 1) ** 2 != n_channels:
        raise ValueError("Number of channels must be a square Ambisonics channel count.")

    # -----------------------------
    # Frequency axis for FFT bins
    # -----------------------------
    f = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    kR = 2 * np.pi * f * R / c

    kR_safe = kR.copy()
    kR_safe[0] = 1e-12

    # -----------------------------
    # Rigid-sphere modal coefficients
    # -----------------------------
    bN = rigid_sphere_modal_coeffs(order_sht, kR_safe) / (4 * np.pi)

    # DC handling
    bN[0, 1:] = 0.0

    # -----------------------------
    # Moreau/Daniel regularization
    # -----------------------------
    alpha = np.sqrt(Nmic) * 10 ** (amp_threshold / 20)

    if alpha <= 1:
        raise ValueError("amp_threshold is too low: alpha must be greater than 1.")

    beta = np.sqrt(
        (1 - np.sqrt(1 - 1 / alpha**2)) /
        (1 + np.sqrt(1 - 1 / alpha**2))
    )

    # -----------------------------
    # Difference filters
    #
    # H_reg = conj(bN) / (abs(bN)^2 + beta^2)
    # H_opt = 1 / bN
    #
    # Difference filter:
    # H_diff = H_reg / H_opt
    #        = H_reg * bN
    #        = abs(bN)^2 / (abs(bN)^2 + beta^2)
    # -----------------------------
    H_diff = np.abs(bN) ** 2 / (np.abs(bN) ** 2 + beta**2)

    # Avoid filtering order 0 at DC incorrectly
    H_diff[0, 0] = 1.0

    # -----------------------------
    # Apply filters in frequency domain
    # -----------------------------
    X = np.fft.rfft(signal, axis=0)

    for n in range(order_sht + 1):
        ch_start = n * n
        ch_end = (n + 1) * (n + 1)

        X[:, ch_start:ch_end] *= H_diff[:, n, None]

    filtered = np.fft.irfft(X, n=n_samples, axis=0)

    return filtered.T.astype(dtype_in)

def rigid_sphere_modal_coeffs(max_order, x):
    b = np.zeros((len(x), max_order + 1), dtype=complex)

    for n in range(max_order + 1):
        jn = spherical_jn(n, x)
        jn_der = spherical_jn(n, x, derivative=True)

        yn = spherical_yn(n, x)
        yn_der = spherical_yn(n, x, derivative=True)

        hn2 = jn - 1j * yn
        hn2_der = jn_der - 1j * yn_der

        b[:, n] = 4 * np.pi * (1j ** n) * (
            jn - (jn_der / hn2_der) * hn2
        )

    return b


def remove_image_sources_before_direct(room, atol: float = 1e-12) -> list[int]:
    """Remove randomized image sources that precede the direct sound.

    An image is removed if its distance to at least one microphone is shorter
    than the corresponding physical source-to-microphone distance.  All
    per-image arrays used by ``pyroomacoustics`` are filtered with the same
    mask so that they remain aligned for ``room.compute_rir()``.

    Returns:
        Number of removed images for each source in ``room.sources``.
    """
    microphone_positions = np.asarray(room.mic_array.R, dtype=float)
    removed_per_source = []

    for source_index, source in enumerate(room.sources):
        images = np.asarray(source.images)
        orders = np.asarray(source.orders)
        n_images = images.shape[1]

        if orders.ndim != 1 or orders.size != n_images:
            raise RuntimeError(
                f"Invalid image/order arrays for source {source_index}."
            )

        direct_indices = np.flatnonzero(orders == 0)
        if direct_indices.size == 0:
            raise RuntimeError(
                f"No order-0 image found for source {source_index}."
            )

        # Pyroomacoustics commonly stores image positions as float32.  Cast the
        # physical position to that dtype before assigning and verifying it;
        # comparing the stored float32 value with the original float64 value at
        # a 1e-12 tolerance can otherwise report a false restoration failure.
        physical_source = np.asarray(source.position, dtype=float).reshape(
            images.shape[0]
        )
        stored_physical_source = physical_source.astype(
            source.images.dtype, copy=False
        )
        source.images[:, direct_indices] = stored_physical_source[:, None]

        if not np.array_equal(
            source.images[:, direct_indices],
            np.broadcast_to(
                stored_physical_source[:, None],
                source.images[:, direct_indices].shape,
            ),
        ):
            raise RuntimeError(
                f"Could not restore the direct image for source {source_index}."
            )

        direct_distances = np.linalg.norm(
            microphone_positions - physical_source[:, None], axis=0
        )
        image_distances = np.linalg.norm(
            source.images[:, None, :] - microphone_positions[:, :, None],
            axis=0,
        )

        arrives_before_direct = np.any(
            (orders[None, :] != 0)
            & (image_distances < direct_distances[:, None] - atol),
            axis=0,
        )
        keep_indices = np.flatnonzero(~arrives_before_direct)

        # These attributes all contain one entry per image source, but their
        # image axis differs.  Filtering them together prevents index drift in
        # pyroomacoustics when the RIR is computed.
        per_image_attributes = {
            "images": 1,
            "orders": 0,
            "orders_xyz": 1,
            "directions": 2,
            "walls": 0,
            "damping": 1,
            "generators": 0,
        }
        for attribute, image_axis in per_image_attributes.items():
            value = getattr(source, attribute, None)
            if value is None:
                continue

            value = np.asarray(value)
            if value.shape[image_axis] != n_images:
                raise RuntimeError(
                    f"Unexpected {attribute} shape for source {source_index}: "
                    f"{value.shape}."
                )
            setattr(
                source,
                attribute,
                np.take(value, keep_indices, axis=image_axis),
            )

        visibility = np.asarray(room.visibility[source_index])
        if visibility.shape[1] != n_images:
            raise RuntimeError(
                f"Unexpected visibility shape for source {source_index}: "
                f"{visibility.shape}."
            )
        room.visibility[source_index] = visibility[:, keep_indices]

        removed_per_source.append(int(np.count_nonzero(arrives_before_direct)))

    return removed_per_source


def save_rir_visualization(
    rir,
    fs,
    output_path,
    name,
    diagnostics=None,
    selected_rt_profile=None,
    selected_center_freqs=None,
    energy_legend_linewidths=3.0,
    schroeder_legend_linewidths=3.0,
):
    """
    Save energy-decomposition, Schroeder-decay, RT-profile, and
    individual RIR-channel plots.

    Parameters
    ----------
    energy_legend_linewidths, schroeder_legend_linewidths : float or dict, optional
        Legend-only line widths. A scalar applies the same width to every
        compatible legend handle. A dict can set widths per legend label, e.g.
        {"Input ISM": 3.0, "Final RIR": 5.0}. These settings do not change
        the linewidths of the curves in the plots.
    """
    import os
    import unicodedata

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    # Larger, consistent font sizes for the complete diagnostic figure.
    font_sizes = {
        "suptitle": 20,
        "title": 16,
        "label": 14,
        "tick": 12,
        "legend": 11,
        "waveform_title": 13,
        "waveform_label": 12,
        "waveform_tick": 11,
        "annotation": 13,
    }

    def _set_legend_linewidths(legend, linewidths):
        """Adjust legend handles without changing the plotted artists."""
        if legend is None or linewidths is None:
            return

        # Matplotlib renamed this public attribute; support both spellings.
        handles = getattr(legend, "legend_handles", None)
        if handles is None:
            handles = getattr(legend, "legendHandles", [])

        labels = [text.get_text() for text in legend.get_texts()]

        if np.isscalar(linewidths):
            width_by_label = {
                label: float(linewidths)
                for label in labels
            }
        else:
            width_by_label = dict(linewidths)

        for label, handle in zip(labels, handles):
            if label not in width_by_label:
                continue

            width = float(width_by_label[label])
            if not np.isfinite(width) or width < 0:
                raise ValueError(
                    f"Legend linewidth for {label!r} must be finite and "
                    f"non-negative, got {width}"
                )

            if hasattr(handle, "set_linewidth"):
                handle.set_linewidth(width)

    rir = np.asarray(rir, dtype=float)

    if rir.ndim == 1:
        rir = rir[np.newaxis, :]

    if rir.ndim != 2 or rir.shape[0] == 0 or rir.shape[1] == 0:
        raise ValueError(
            f"rir must have shape (channels, samples), got {rir.shape}"
        )

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be finite and positive, got {fs}")

    diagnostics = {} if diagnostics is None else diagnostics

    num_channels, num_samples = rir.shape
    time_s = np.arange(num_samples, dtype=float) / float(fs)

    def _clean_text(value):
        """Remove control characters from externally supplied text."""
        return "".join(
            character
            for character in str(value)
            if not unicodedata.category(character).startswith("C")
        )

    def _match_length(values):
        """Convert diagnostics to a finite vector matching the RIR length."""
        if values is None:
            return None

        values = np.asarray(values, dtype=float).reshape(-1)
        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if values.size < num_samples:
            values = np.pad(
                values,
                (0, num_samples - values.size),
            )

        return values[:num_samples]

    # ------------------------------------------------------------------
    # Energy data
    # ------------------------------------------------------------------
    final_energy = np.sum(rir**2, axis=0)

    energy_data = {
        "Input ISM": _match_length(
            diagnostics.get("ism_energy_per_sample")
        ),
        "Windowed early": _match_length(
            diagnostics.get("rir_early_energy_per_sample")
        ),
        "Stochastic late": _match_length(
            diagnostics.get("late_reverb_energy_per_sample")
        ),
        "Final RIR": final_energy,
    }

    available_energy = [
        values
        for values in energy_data.values()
        if values is not None
    ]

    energy_reference = max(
        (
            float(np.max(np.maximum(values, 0.0)))
            for values in available_energy
        ),
        default=0.0,
    )
    energy_reference = max(
        energy_reference,
        np.finfo(float).tiny,
    )

    energy_floor_db = -120.0
    energy_floor_linear = (
        energy_reference * 10.0 ** (energy_floor_db / 10.0)
    )

    def _energy_to_db(values, mask_inactive=False):
        """
        Smooth energy in the linear domain and then convert it to dB.

        Smoothing directly in dB can distort sharp transitions and spread
        the lower plotting floor into adjacent samples.
        """
        values = np.maximum(
            np.asarray(values, dtype=float),
            0.0,
        )

        smoothed = gaussian_filter1d(
            values,
            sigma=2.0,
            mode="nearest",
        )

        values_db = 10.0 * np.log10(
            np.maximum(
                smoothed / energy_reference,
                1e-12,
            )
        )

        if mask_inactive:
            values_db = values_db.copy()
            values_db[smoothed <= energy_floor_linear] = np.nan

        return values_db

    def _truncate_input_ism_tail(
        values_db,
        minimum_gap_seconds=0.05,
    ):
        """
        Remove disconnected Input-ISM activity after a sustained gap.

        This is a plotting-only operation. It removes isolated activity
        near the end of the buffer after the Input ISM has already remained
        below the visible energy floor for the specified duration.
        """
        values_db = np.asarray(values_db, dtype=float).copy()

        if (
            values_db.size == 0
            or not np.any(np.isfinite(values_db))
        ):
            return values_db

        peak_sample = int(np.nanargmax(values_db))
        minimum_gap_samples = max(
            1,
            int(round(minimum_gap_seconds * float(fs))),
        )

        below_floor = (
            ~np.isfinite(values_db)
            | (values_db <= energy_floor_db + 1.0)
        )

        gap_start = None

        for sample in range(
            peak_sample + 1,
            values_db.size,
        ):
            if below_floor[sample]:
                if gap_start is None:
                    gap_start = sample

                gap_length = sample - gap_start + 1

                if gap_length >= minimum_gap_samples:
                    values_db[gap_start:] = np.nan
                    break
            else:
                gap_start = None

        return values_db

    # ------------------------------------------------------------------
    # Schroeder energy decay
    # ------------------------------------------------------------------
    final_schroeder = np.cumsum(
        final_energy[::-1]
    )[::-1]

    if final_schroeder[0] > 0:
        final_schroeder_db = 10.0 * np.log10(
            np.maximum(
                final_schroeder / final_schroeder[0],
                1e-12,
            )
        )
    else:
        final_schroeder_db = np.zeros(
            num_samples,
            dtype=float,
        )

    # Keep -inf/NaN in the diagnostic Schroeder curve.  In particular, do
    # not pass it through _match_length(), because that helper intentionally
    # turns non-finite values into 0.0 for energy arrays.  A -inf value in a
    # Schroeder decay means that the cumulative energy has reached zero and
    # the curve must end there rather than jump back up to 0 dB.
    raw_input_schroeder_db = diagnostics.get("schroeder_db")
    input_schroeder_db = None

    if raw_input_schroeder_db is not None:
        input_schroeder_db = np.asarray(
            raw_input_schroeder_db,
            dtype=float,
        ).reshape(-1)

        if input_schroeder_db.size < num_samples:
            input_schroeder_db = np.pad(
                input_schroeder_db,
                (0, num_samples - input_schroeder_db.size),
                constant_values=np.nan,
            )
        else:
            input_schroeder_db = input_schroeder_db[:num_samples]

        # Once the decay reaches -inf, stop plotting the curve from that
        # sample onward.  Also mask any +inf values, which are not meaningful
        # for a normalized Schroeder decay.
        neginf_samples = np.flatnonzero(
            np.isneginf(input_schroeder_db)
        )
        if neginf_samples.size:
            input_schroeder_db[neginf_samples[0]:] = np.nan

        input_schroeder_db[np.isposinf(input_schroeder_db)] = np.nan

    # ------------------------------------------------------------------
    # Mixing-time diagnostics
    # ------------------------------------------------------------------
    mixing_sample = int(
        np.clip(
            diagnostics.get("mixingTime_sample", 0),
            0,
            num_samples - 1,
        )
    )

    fade_samples = max(
        0,
        int(diagnostics.get("actual_fade_samples", 0)),
    )

    mixing_time_s = mixing_sample / float(fs)
    fade_end_s = (
        min(
            num_samples,
            mixing_sample + fade_samples,
        )
        / float(fs)
    )

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    waveform_columns = 4
    waveform_rows = int(
        np.ceil(num_channels / waveform_columns)
    )

    fig = plt.figure(
        figsize=(
            18,
            5.0 + 2.4 * waveform_rows,
        )
    )

    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[
            1.35,
            max(1.0, 0.75 * waveform_rows),
        ],
    )

    summary_grid = outer[0].subgridspec(
        1,
        3,
        wspace=0.28,
    )

    waveform_grid = outer[1].subgridspec(
        waveform_rows,
        waveform_columns,
        hspace=0.45,
        wspace=0.30,
    )

    energy_ax = fig.add_subplot(
        summary_grid[0, 0]
    )
    schroeder_ax = fig.add_subplot(
        summary_grid[0, 1]
    )
    rt_ax = fig.add_subplot(
        summary_grid[0, 2]
    )

    fig.suptitle(
        f"RIR diagnostics - {_clean_text(name)}",
        fontsize=font_sizes["suptitle"],
    )

    # ------------------------------------------------------------------
    # Energy decomposition
    # ------------------------------------------------------------------
    #if fade_end_s > mixing_time_s:
    #    energy_ax.axvspan(
    #        mixing_time_s,
    #        fade_end_s,
    #        color="0.75",
    #        alpha=0.5,
    #        label="Crossfade",
    #       zorder=0,
    #    )

    energy_styles = {
        "Input ISM": {
            "color": "#0072B2",
            "linewidth": 0.5,
            "linestyle": "-",
            "alpha": 1,
            "zorder": 8,
        },
        # The wide blue curve forms a visible base beneath the narrower
        # orange and green component curves.
        "Final RIR": {
            "color": "#B20000",
            "linewidth": 4.0,
            "linestyle": "-",
            "alpha": 1.0,
            "zorder": 4,
        },
        "Windowed early": {
            "color": "#E69F00",
            "linewidth": 2,
            "linestyle": "-",
            "alpha": 1,
            "zorder": 6,
        },
        # The dotted pattern exposes the blue curve between green segments.
        "Stochastic late": {
            "color": "#009E73",
            "linewidth": 1,
            "linestyle": "-",
            "alpha": 1,
            "zorder": 7,
        },
    }

    plot_order = [
        "Input ISM",
        "Final RIR",
        "Windowed early",
        "Stochastic late",
    ]

    component_labels = {
        "Windowed early",
        "Stochastic late",
    }

    for label in plot_order:
        values = energy_data[label]

        if values is None:
            continue

        values_db = _energy_to_db(
            values,
            mask_inactive=label in component_labels,
        )

        if label == "Input ISM":
            values_db = _truncate_input_ism_tail(
                values_db,
                minimum_gap_seconds=0.05,
            )

        line, = energy_ax.plot(
            time_s,
            values_db,
            label=label,
            **energy_styles[label],
        )

        if label == "Stochastic late":
            line.set_dash_capstyle("round")

    energy_ax.axvline(
        mixing_time_s,
        color="black",
        linewidth=1.5,
        linestyle="--",
        label="Mixing time",
        zorder=8,
    )

    energy_ax.set_xlim(
        time_s[0],
        (
            time_s[-1]
            if num_samples > 1
            else 1.0 / float(fs)
        ),
    )
    energy_ax.set_ylim(
        energy_floor_db,
        5,
    )
    energy_ax.set_title(
        "Energy decomposition over time",
        fontsize=font_sizes["title"],
    )
    energy_ax.set_xlabel(
        "Time [s]",
        fontsize=font_sizes["label"],
    )
    energy_ax.set_ylabel(
        "Energy [dB, shared reference]",
        fontsize=font_sizes["label"],
    )
    energy_ax.tick_params(
        axis="both",
        labelsize=font_sizes["tick"],
    )
    energy_ax.grid(
        True,
        which="both",
        alpha=0.25,
        zorder=-1,
    )

    handles, labels = (
        energy_ax.get_legend_handles_labels()
    )
    handle_by_label = dict(
        zip(labels, handles)
    )

    legend_order = [
        "Input ISM",
        "Windowed early",
        "Stochastic late",
        "Final RIR",
        "Mixing time",
        "Crossfade",
    ]

    legend_labels = [
        label
        for label in legend_order
        if label in handle_by_label
    ]

    energy_legend = energy_ax.legend(
        [
            handle_by_label[label]
            for label in legend_labels
        ],
        legend_labels,
        fontsize=font_sizes["legend"],
        ncol=2,
        framealpha=0.9,
    )
    _set_legend_linewidths(
        energy_legend,
        energy_legend_linewidths,
    )

    # ------------------------------------------------------------------
    # Schroeder energy-decay plot
    # ------------------------------------------------------------------
    if input_schroeder_db is not None:
        # Smooth only the finite part of the decay.  Leaving the remainder as
        # NaN prevents Matplotlib from drawing a vertical return to 0 dB.
        input_schroeder_smoothed = np.full(
            num_samples,
            np.nan,
            dtype=float,
        )
        finite_samples = np.flatnonzero(
            np.isfinite(input_schroeder_db)
        )

        if finite_samples.size:
            first_sample = int(finite_samples[0])
            invalid_after_start = np.flatnonzero(
                ~np.isfinite(
                    input_schroeder_db[first_sample:]
                )
            )
            stop_sample = (
                first_sample + int(invalid_after_start[0])
                if invalid_after_start.size
                else num_samples
            )

            if stop_sample > first_sample:
                input_schroeder_smoothed[
                    first_sample:stop_sample
                ] = gaussian_filter1d(
                    input_schroeder_db[
                        first_sample:stop_sample
                    ],
                    sigma=3,
                    mode="nearest",
                )

        schroeder_ax.plot(
            time_s,
            input_schroeder_smoothed,
            color="#0072B2",
            linewidth=2,
            label="Input ISM",
        )

    schroeder_ax.plot(
        time_s,
        gaussian_filter1d(
            final_schroeder_db,
            sigma=3,
            mode="nearest",
        ),
        color="#B20000",
        linewidth=2,
        label="Final RIR",
    )

    schroeder_ax.axvline(
        mixing_time_s,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Mixing time",
    )

    schroeder_ax.set_xlim(
        time_s[0],
        (
            time_s[-1]
            if num_samples > 1
            else 1.0 / float(fs)
        ),
    )
    schroeder_ax.set_ylim(-80, 5)
    schroeder_ax.set_title(
        "Schroeder energy decay",
        fontsize=font_sizes["title"],
    )
    schroeder_ax.set_xlabel(
        "Time [s]",
        fontsize=font_sizes["label"],
    )
    schroeder_ax.set_ylabel(
        "Normalized cumulative energy [dB]",
        fontsize=font_sizes["label"],
    )
    schroeder_ax.tick_params(
        axis="both",
        labelsize=font_sizes["tick"],
    )
    schroeder_ax.grid(
        True,
        which="both",
        alpha=0.3,
    )
    schroeder_legend = schroeder_ax.legend(
        fontsize=font_sizes["legend"]
    )
    _set_legend_linewidths(
        schroeder_legend,
        schroeder_legend_linewidths,
    )

    # ------------------------------------------------------------------
    # RT profile
    # ------------------------------------------------------------------
    rt_profile = (
        selected_rt_profile
        if selected_rt_profile is not None
        else diagnostics.get("rt_profile")
    )

    center_freqs = (
        selected_center_freqs
        if selected_center_freqs is not None
        else diagnostics.get("centerFreqs")
    )

    if rt_profile is None:
        rt_ax.text(
            0.5,
            0.5,
            "No RT profile available",
            ha="center",
            va="center",
            fontsize=font_sizes["annotation"],
        )
        rt_ax.set_axis_off()
    else:
        rt_profile = np.asarray(
            rt_profile,
            dtype=float,
        ).reshape(-1)

        if center_freqs is None:
            center_freqs = np.arange(
                1,
                rt_profile.size + 1,
                dtype=float,
            )
            x_label = "Band index"
            use_log_x = False
        else:
            center_freqs = np.asarray(
                center_freqs,
                dtype=float,
            ).reshape(-1)
            x_label = "Center frequency [Hz]"
            use_log_x = True

        count = min(
            rt_profile.size,
            center_freqs.size,
        )

        valid = (
            np.isfinite(rt_profile[:count])
            & (rt_profile[:count] > 0)
            & np.isfinite(center_freqs[:count])
            & (center_freqs[:count] > 0)
        )

        if np.any(valid):
            plot_method = (
                rt_ax.semilogx
                if use_log_x
                else rt_ax.plot
            )

            plot_method(
                center_freqs[:count][valid],
                rt_profile[:count][valid],
                color="#0072B2",
                linewidth=1.5,
                marker="o",
            )

            rt_ax.set_title(
                "Selected RT profile",
                fontsize=font_sizes["title"],
            )
            rt_ax.set_xlabel(
                x_label,
                fontsize=font_sizes["label"],
            )
            rt_ax.set_ylabel(
                "$T_{60}$ [s]",
                fontsize=font_sizes["label"],
            )
            rt_ax.tick_params(
                axis="both",
                labelsize=font_sizes["tick"],
            )
            rt_ax.grid(
                True,
                which="both",
                alpha=0.3,
            )
        else:
            rt_ax.text(
                0.5,
                0.5,
                "No valid RT values",
                ha="center",
                va="center",
                fontsize=font_sizes["annotation"],
            )
            rt_ax.set_axis_off()

    # ------------------------------------------------------------------
    # Individual RIR-channel plots
    # ------------------------------------------------------------------
    waveform_peak = float(
        np.max(np.abs(rir))
    )
    waveform_limit = (
        1.05 * waveform_peak
        if waveform_peak > 0
        else 1.0
    )

    waveform_xmax = (
        time_s[-1]
        if num_samples > 1
        else 1.0 / float(fs)
    )

    for channel in range(num_channels):
        row, column = divmod(
            channel,
            waveform_columns,
        )

        ax = fig.add_subplot(
            waveform_grid[row, column]
        )

        ax.plot(
            time_s,
            rir[channel],
            linewidth=0.55,
            color="#0072B2",
        )

        ax.axvline(
            mixing_time_s,
            linestyle="--",
            linewidth=1.5,
            color="black",
        )

        #if fade_end_s > mixing_time_s:
        #    ax.axvspan(
        #        mixing_time_s,
        #        fade_end_s,
        #        alpha=0.22,
        #        color="0.75",
        #    )

        ax.set_xlim(
            time_s[0],
            waveform_xmax,
        )
        ax.set_ylim(
            -waveform_limit,
            waveform_limit,
        )
        ax.set_title(
            f"RIR channel {channel}",
            fontsize=font_sizes["waveform_title"],
        )
        ax.grid(
            True,
            alpha=0.25,
        )
        ax.tick_params(
            labelsize=font_sizes["waveform_tick"],
        )
        ax.ticklabel_format(
            axis="y",
            style="sci",
            scilimits=(-2, 2),
        )

        if row == waveform_rows - 1:
            ax.set_xlabel(
                "Time [s]",
                fontsize=font_sizes["waveform_label"],
            )

        if column == 0:
            ax.set_ylabel(
                "Amplitude",
                fontsize=font_sizes["waveform_label"],
            )

    for empty_index in range(
        num_channels,
        waveform_rows * waveform_columns,
    ):
        row, column = divmod(
            empty_index,
            waveform_columns,
        )
        ax = fig.add_subplot(
            waveform_grid[row, column]
        )
        ax.set_axis_off()

    # ------------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------------
    fig.tight_layout(
        rect=(0, 0, 1, 0.96)
    )

    plot_dir = os.path.join(
        output_path,
        "rir_plots",
    )
    os.makedirs(
        plot_dir,
        exist_ok=True,
    )

    filename = os.path.join(
        plot_dir,
        f"{name}_rir_visualization.pdf",
    )

    fig.savefig(
        filename,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    return filename


def simulate_room(
    room_idx: int,
    pos_idx: int,
    output_path: str,
    physical_params: list,
    array_type: str = "ambisonic",
    ambi_order: int = 3,
    sample_rate: int = 16000,
    em32_directivity_path: str = None,
    em32_radius: float = 0.042,
    amp_threshold: float = 15.0,
    em32_filter_length: int = 1024,
    ism_order: int = 12,
    export_mode: str = "ism_late",
    shoebox: bool = False,
    max_displacement_corners: float = 0.5,
    use_rand_ism: bool = False,
    max_rand_disp: float = 0.1,
    save_plots: bool = False,
) -> Union[dict, None]:
    """
    Simulates a single room impulse response (RIR) for a given room configuration
    and source/microphone position using pyroomacoustics. Returns the RIR array and
    acoustic metadata to the parent process. Includes error handling.

    Args:
        room_idx: Index of the current room simulation.
        pos_idx: Index of the current source/microphone position within the room.
        output_path: Output directory. Pickle writing is handled by the parent process.
        ambisonic_microphone: Tuple defining the Ambisonic microphone array
                              (positions, sample_rate, directivities).
        physical_params: A list containing the physical parameters of the room and
                         source/mic positions:
                         [Lx, Ly, Lz, materials, Sx, Sy, Sz, Rx, Ry, Rz].

    Returns:
        dict: A dictionary containing metadata and acoustic parameters for the
              generated RIR if successful.
        None: If an error occurred during simulation.
    """
    valid_export_modes = {"ism", "ism_late"}
    if export_mode not in valid_export_modes:
        raise ValueError(
            f"export_mode must be one of {sorted(valid_export_modes)}, got {export_mode!r}"
        )
    if ism_order < 0:
        raise ValueError(f"ism_order must be >= 0, got {ism_order}")

    room_name = f"R{room_idx:04d}_P{pos_idx}"
    rir_name = f"{room_name}_IR"

    # Extract physical parameters with clear variable names
    (
        Lx, Ly, Lz, Sx, Sy, Sz, Rx, Ry, Rz, rt60,
        e_absorption_rest, e_absorption_selected_wall, centerFreqs,
        idxDifferentWall, floor_corners, volume, array_azimuth_rad,
    ) = physical_params

    logging.info(f"Starting simulation for {room_name} with params: "
                 f"Dims=({Lx[0]:.2f}, {Ly[0]:.2f}, {Lz[0]:.2f}), "
                 f"Src=({Sx[0]:.2f}, {Sy[0]:.2f}, {Sz[0]:.2f}), "
                 f"Mic=({Rx[0]:.2f}, {Ry[0]:.2f}, {Rz[0]:.2f})")

    try:
        # Ensure parameters are simple floats for pyroomacoustics
        room_dims = [Lx[0], Ly[0], Lz[0]]
        source_pos = [Sx[0], Sy[0], Sz[0]]
        mic_center_pos = [Rx[0], Ry[0], Rz[0]]

        # Expensive rotated-directivity/filter generation happens in parallel
        # inside worker processes instead of blocking the serial room loop.
        ambisonic_microphone = _get_worker_array(
            array_type=array_type,
            ambi_order=ambi_order,
            sample_rate=sample_rate,
            array_azimuth_rad=array_azimuth_rad,
            em32_directivity_path=em32_directivity_path,
            em32_radius=em32_radius,
        )

        # Direction of arrival in the microphone-local coordinate frame.
        # This explicitly accounts for the microphone-array yaw orientation.
        accdoa, doa_az_deg, doa_el_deg = doa_in_mic_frame(
            source_pos,
            mic_center_pos,
            array_azimuth_rad,
        )

        material_properties = [
            {
                "description": "uniform_material",
                "coeffs": e_absorption_rest,
                "center_freqs": centerFreqs,
            }
            for _ in range(6)
        ]
        material_properties[idxDifferentWall] = {
            "description": "uniform_material",
            "coeffs": e_absorption_selected_wall,
            "center_freqs": centerFreqs,
        }

        materials_side_walls = pra.make_materials(
            *((material_properties[index],) for index in range(4))
        )
        floor_and_ceiling = pra.make_materials(
            floor=(material_properties[4],),
            ceiling=(material_properties[5],),
        )

        wall_keys = ["west", "east", "south", "north", "floor", "ceiling"]
        materials_dict = {
            wall: (material_properties[index],)
            for index, wall in enumerate(wall_keys)
        }

        def build_room(max_order):
            """Build the same physical room at a requested ISM order."""
            if shoebox:
                return pra.ShoeBox(
                    p=room_dims,
                    materials=pra.make_materials(**materials_dict),
                    fs=ambisonic_microphone[1],
                    max_order=max_order,
                    humidity=0.45,
                    air_absorption=True,
                    use_rand_ism=use_rand_ism,
                    max_rand_disp=max_rand_disp,
                )

            built_room = pra.Room.from_corners(
                floor_corners,
                max_order=max_order,
                materials=materials_side_walls,
                fs=ambisonic_microphone[1],
                humidity=0.45,
                air_absorption=True,
                use_rand_ism=use_rand_ism,
                max_rand_disp=max_rand_disp,
            )
            built_room.extrude(room_dims[2], materials=floor_and_ceiling)
            return built_room

        room = build_room(ism_order)

        geometry_metadata = get_room_geometry_metadata(
            room_dims, floor_corners, shoebox
        )

        print(
            f"RT60: {rt60}, Absorption Selected Wall: "
            f"{e_absorption_selected_wall}, Absorption Rest: {e_absorption_rest}, "
            f"Selected Wall Index: {idxDifferentWall}, Volume: {volume}, "
            f"Room Dims: {room_dims}, Source Pos: {source_pos}, "
            f"Receiver Pos: {mic_center_pos}"
        )


        # Add source and microphone array
        room.add_source(source_pos)
        room.add_microphone_array(
            pra.MicrophoneArray(
                ambisonic_microphone[0] + np.array(mic_center_pos)[:, None], # Adjust mic positions relative to center
                ambisonic_microphone[1], # Sample rate
                ambisonic_microphone[2], # Directivities
            )
        )

        # Compute RIR using Image Source Model
        room.image_source_model()

        if use_rand_ism:
            removed_images = remove_image_sources_before_direct(room)
            if any(removed_images):
                logging.info(
                    "Removed premature randomized image sources per source: %s",
                    removed_images,
                )

        room.compute_rir()

        # Post-processing RIRs
        # Find maximum length across all channels to pad
        max_len = np.array([len(r[0]) for r in room.rir]).max()
        # Determine target length for padding (pad to the nearest second)
        target_len = int(np.ceil(max_len / room.fs) * room.fs)

        # Stack and pad RIRs
        RIR = np.stack(
            [np.pad(r[0], (0, target_len - r[0].shape[0])) for r in room.rir],
            axis=0, # Stack along a new dimension (channels)
        )


        '''
        # Normalize RIR by the maximum absolute value across all channels
        max_rir_val = np.max(np.abs(RIR))
        if max_rir_val > 0:
            RIR = RIR / max_rir_val
        else:
             # Handle cases where the RIR might be all zeros (e.g., source/mic outside room)
             logging.warning(f"RIR for {room_name} is all zeros or near zero. Skipping acoustic parameter calculation and file writing.")
             return {
                "Name": rir_name,
                "RT": np.nan, # Use NaN for undefined values
                "c50": np.nan,
                "x_room": Lx[0], "y_room": Ly[0], "z_room": Lz[0],
                "x_source": Sx[0], "y_source": Sy[0], "z_source": Sz[0],
                "x_mic": Rx[0], "y_mic": Ry[0], "z_mic": Rz[0],
                "status": "Warning: RIR all zeros"
             }
        '''

        # The previous code generated a complete stochastic late-reverb tail
        # but did not assign it back to RIR. That expensive no-op is omitted.

        if array_type == "em32":
            if em32_directivity_path is None:
                raise ValueError("em32_directivity_path is required for EM32 simulation")
            RIR = convert_em32_to_sh(
                RIR,
                order_sht=ambi_order,
                sample_rate=room.fs,
                amp_threshold=amp_threshold,
                radius=em32_radius,
                filter_length=em32_filter_length,
            )
        else:
            # Preserve the original ideal-HOA post-filtering path.
            RIR = apply_tikhonov_difference_filters(
                RIR, room.fs, R=em32_radius, Nmic=32,
                amp_threshold=amp_threshold, c=343.0
            )

        # Select exactly what is exported.
        diagnostics = None
        if export_mode == "ism":
            # Export only the image-source-model RIR. No direct-only room and no
            # stochastic late reverberation are needed.
            pass
        elif export_mode == "ism_late":
            # Keep the requested-order ISM as the early part.  A separate
            # order-12 ISM is used only as the late-energy reference and is
            # never included directly in the returned/saved RIR.
            if ism_order == LATE_REFERENCE_ISM_ORDER:
                late_reference_rir = RIR
            else:
                late_reference_room = build_room(LATE_REFERENCE_ISM_ORDER)
                late_reference_room.add_source(source_pos)
                late_reference_room.add_microphone_array(
                    pra.MicrophoneArray(
                        ambisonic_microphone[0]
                        + np.array(mic_center_pos)[:, None],
                        ambisonic_microphone[1],
                        ambisonic_microphone[2],
                    )
                )
                late_reference_room.image_source_model()

                if use_rand_ism:
                    removed_images = remove_image_sources_before_direct(
                        late_reference_room
                    )
                    if any(removed_images):
                        logging.info(
                            "Removed premature randomized image sources from "
                            "order-%d late reference per source: %s",
                            LATE_REFERENCE_ISM_ORDER,
                            removed_images,
                        )

                late_reference_room.compute_rir()
                reference_max_len = max(
                    len(microphone_rirs[0])
                    for microphone_rirs in late_reference_room.rir
                )
                reference_target_len = int(
                    np.ceil(reference_max_len / late_reference_room.fs)
                    * late_reference_room.fs
                )
                late_reference_rir = np.stack(
                    [
                        np.pad(
                            microphone_rirs[0],
                            (0, reference_target_len - len(microphone_rirs[0])),
                        )
                        for microphone_rirs in late_reference_room.rir
                    ],
                    axis=0,
                )

                # Put the reference in the same SH domain as the exported RIR
                # before measuring its channel-0 late energy.
                if array_type == "em32":
                    late_reference_rir = convert_em32_to_sh(
                        late_reference_rir,
                        order_sht=ambi_order,
                        sample_rate=late_reference_room.fs,
                        amp_threshold=amp_threshold,
                        radius=em32_radius,
                        filter_length=em32_filter_length,
                    )
                else:
                    late_reference_rir = apply_tikhonov_difference_filters(
                        late_reference_rir,
                        late_reference_room.fs,
                        R=em32_radius,
                        Nmic=32,
                        amp_threshold=amp_threshold,
                        c=343.0,
                    )

                logging.info(
                    "Computed internal order-%d ISM reference for late "
                    "reverberation estimation in %s; exported early ISM order is %d.",
                    LATE_REFERENCE_ISM_ORDER,
                    room_name,
                    ism_order,
                )

            RIR, RIR_early, RIR_late, diagnostics = addStochasticLateReverbPyfar(
                srir=RIR,
                late_reference_srir=late_reference_rir,
                rt=rt60,
                fs=room.fs,
                volume=volume,
                centerFreqs=centerFreqs,
                num_fractions=1,
                filter_order=14,
                ism_order=ism_order
            )

        # Pad final RIR to the next full second
        current_len = RIR.shape[1]
        target_len = int(np.ceil(current_len / room.fs) * room.fs)

        if target_len > current_len:
            RIR = np.pad(
                RIR,
                pad_width=((0, 0), (0, target_len - current_len)),
                mode="constant"
            )

        # Calculate Acoustic Parameters (RT60 and C50)
        # Calculate RT60 from the first channel (often representative)
        # Add a check if the RIR is long enough for RT60 calculation
        rt60_sim = np.nan # Default to NaN
        if len(RIR[0]) > room.fs * 0.1: # Ensure at least 100ms for decay calculation
             rt60_sim = pra.experimental.rt60.measure_rt60(RIR[0], fs=room.fs, decay_db=30)
        else:
             logging.warning(f"RIR for {room_name} too short ({len(RIR[0])} samples) for RT60 calculation.")


        # Calculate C50 from the first channel
        c50 = np.nan # Default to NaN
        # Ensure the RIR is long enough to have samples at 50ms
        if len(RIR[0]) > int(room.fs * 0.05):
            early_energy = np.sum(np.square(np.abs(RIR[0, : int(room.fs * 0.05)])))
            late_energy = np.sum(np.square(np.abs(RIR[0, int(room.fs * 0.05) : ])))

            if late_energy > 0: # Avoid division by zero
                 c50 = 10 * np.log10(early_energy / late_energy)
            else:
                 logging.warning(f"Late energy is zero for C50 calculation in {room_name}.")
                 c50 = np.inf # Or handle as appropriate for your analysis

        else:
             logging.warning(f"RIR for {room_name} too short for C50 calculation (needs > 50ms).")


        logging.info(f"Successfully simulated {room_name}. RT60: {rt60_sim:.2f}, C50: {c50:.2f}")

        # Save diagnostic visualization of final RIR

        if save_plots == True:
            try:
                save_rir_visualization(
                    rir=RIR,
                    fs=room.fs,
                    output_path=output_path,
                    name=rir_name,
                    diagnostics=diagnostics,
                    selected_rt_profile=rt60,
                    selected_center_freqs=centerFreqs,
                )
            except Exception as e:
                logging.warning(f"Could not save RIR visualization for {rir_name}: {e}")
        
        
        

        # Return metadata and results
        # Frequency-dependent metadata
        rt_profile_metadata = {
            f"rt60_{int(freq)}Hz": float(rt)
            for freq, rt in zip(centerFreqs, rt60)
        }
        return {
            "Name": rir_name,
            "room_idx": int(room_idx),
            "position_id": str(pos_idx),
            "sample_rate": int(room.fs),
            "rir": np.ascontiguousarray(RIR, dtype=np.float32),
            "RT": rt60_sim,
            "c50": c50,
            "x_room": float(Lx[0]),
            "y_room": float(Ly[0]),
            "z_room": float(Lz[0]),
            **geometry_metadata,
            "max_displacement_corners": float(max_displacement_corners),
            "use_rand_ism": bool(use_rand_ism),
            "max_rand_ism_displacement": float(max_rand_disp),
            "room_volume_m3": float(volume),
            "x_source": round(float(Sx[0]), 2),
            "y_source": round(float(Sy[0]), 2),
            "z_source": round(float(Sz[0]), 2),
            "x_mic": round(float(Rx[0]), 2),
            "y_mic": round(float(Ry[0]), 2),
            "z_mic": round(float(Rz[0]), 2),
            "mic_orientation_az_deg": float(np.degrees(array_azimuth_rad)),
            "accdoa_x": float(accdoa[0]),
            "accdoa_y": float(accdoa[1]),
            "accdoa_z": float(accdoa[2]),
            "az_deg": float(doa_az_deg),
            "el_deg": float(doa_el_deg),
            **rt_profile_metadata,
            "ism_order": int(ism_order),
            "late_reference_ism_order": (
                LATE_REFERENCE_ISM_ORDER if export_mode == "ism_late" else None
            ),
            "export_mode": export_mode,
            "status": "Success"
        }

    except ValueError as ve:
        error_msg = f"ValueError for {room_name}: {ve}"
        logging.error(error_msg)
        return {
            "Name": rir_name,
            "status": "Error: ValueError",
            "error_message": str(ve),
            "x_room": Lx[0], "y_room": Ly[0], "z_room": Lz[0],
            "x_source": Sx[0] if 'Sx' in locals() else np.nan, # Use locals() to check if variable exists
            "y_source": Sy[0] if 'Sy' in locals() else np.nan,
            "z_source": Sz[0] if 'Sz' in locals() else np.nan,
            "x_mic": Rx[0] if 'Rx' in locals() else np.nan,
            "y_mic": Ry[0] if 'Ry' in locals() else np.nan,
            "z_mic": Rz[0] if 'Rz' in locals() else np.nan,
            "mic_orientation_az_deg": float(np.degrees(array_azimuth_rad)) if 'array_azimuth_rad' in locals() else np.nan,
            "accdoa_x": float(accdoa[0]) if 'accdoa' in locals() else np.nan,
            "accdoa_y": float(accdoa[1]) if 'accdoa' in locals() else np.nan,
            "accdoa_z": float(accdoa[2]) if 'accdoa' in locals() else np.nan,
            "az_deg": float(doa_az_deg) if 'doa_az_deg' in locals() else np.nan,
            "el_deg": float(doa_el_deg) if 'doa_el_deg' in locals() else np.nan,
        }
    except Exception as e:
        # Catch any other exceptions
        error_msg = f"An unexpected error occurred during simulation for {room_name}: {e}"
        logging.error(error_msg, exc_info=True) # Log traceback for unexpected errors
        return {
            "Name": rir_name,
            "status": "Error: Unexpected Exception",
            "error_message": str(e),
            "x_room": Lx[0], "y_room": Ly[0], "z_room": Lz[0],
             "x_source": Sx[0] if 'Sx' in locals() else np.nan,
            "y_source": Sy[0] if 'Sy' in locals() else np.nan,
            "z_source": Sz[0] if 'Sz' in locals() else np.nan,
            "x_mic": Rx[0] if 'Rx' in locals() else np.nan,
            "y_mic": Ry[0] if 'Ry' in locals() else np.nan,
            "z_mic": Rz[0] if 'Rz' in locals() else np.nan,
            "mic_orientation_az_deg": float(np.degrees(array_azimuth_rad)) if 'array_azimuth_rad' in locals() else np.nan,
            "accdoa_x": float(accdoa[0]) if 'accdoa' in locals() else np.nan,
            "accdoa_y": float(accdoa[1]) if 'accdoa' in locals() else np.nan,
            "accdoa_z": float(accdoa[2]) if 'accdoa' in locals() else np.nan,
            "az_deg": float(doa_az_deg) if 'doa_az_deg' in locals() else np.nan,
            "el_deg": float(doa_el_deg) if 'doa_el_deg' in locals() else np.nan,
        }


def main(
    output_path: str,
    num_rooms: int,
    num_positions: int,
    ambi_order: int = 3,
    sample_rate: int = 16000,
    num_sources_per_receiver: int = 3,
    array_type: str = "ambisonic",
    em32_directivity_path: str = "Eigenmike_em32_IRs.npz",
    em32_radius: float = 0.042,
    amp_threshold: float = 15.0,
    em32_filter_length: int = 1024,
    ism_order: int = 12,
    export_mode: str = "ism_late",
    shoebox: bool = False,
    max_displacement_corners: float = 0.5,
    use_rand_ism: bool = False,
    max_rand_disp: float = 0.1,
    save_plots: bool = False,
):
    """
    Main function to generate a dataset of RIRs with varying room acoustics
    and source/microphone positions.
    Args:
        output_path: Directory to save one pickle per room and metadata CSV files.
        num_rooms: The number of different room geometries to simulate.
        num_positions: The number of different source/microphone positions
                       within each room geometry.
        ambi_order: The Ambisonic order for the microphone array.
        sample_rate: The sample rate for the simulations.
    """
    start = time.time() # Start time for performance measurement

    start_room, end_room = get_room_range(num_rooms)

    # --- NEU: Realistische RT-Profile laden ---
    rtProfiles = np.load('rtProfilesMit.npy')
    centerFreqs = np.asanyarray(125 * 2**np.arange(0, 8)).flatten()
    numRtProfiles = rtProfiles.shape[0]
    logging.info(f"Loaded {numRtProfiles} RT60 profiles")

    c = 343 # Speed of sound in m/s
    max_displacement_m = max_displacement_corners
    anisotropic = False #np.random.rand() > 0.5
    anisotropicType = np.random.rand() > 0.5 # should it be more or less absorption

    logging.info(
        f"SLURM task processing rooms {start_room} to {end_room-1}"
    )

    logging.info(f"Starting RIR generation: {num_rooms} rooms, {num_positions} positions per room, "
                 f"Ambisonic Order {ambi_order}, Sample Rate {sample_rate}")

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Validate that an Ambisonic array can be constructed. A separately rotated
    # array is created once for each room below.
    try:
        if array_type == "em32":
            create_em32_array(
                em32_directivity_path=em32_directivity_path,
                sample_rate=sample_rate,
                azimuth_rotation=0.0,
                radius=em32_radius,
            )
            logging.info(
                "EM32 measured array validated; output will be encoded to "
                f"order {ambi_order} ACN/N3D SH ({(ambi_order + 1) ** 2} channels)."
            )
        else:
            create_ambisonic_array(
                order_of_ambisonics=ambi_order,
                sample_rate=sample_rate,
                azimuth_rotation=0.0,
            )
            logging.info(
                f"Ambisonic array configuration validated for order {ambi_order} "
                f"({(ambi_order + 1) ** 2} channels)."
            )
    except Exception as e:
        logging.critical(f"Failed to create Ambisonic microphone array: {e}")
        return

    all_results = []  # Lightweight metadata rows only; no RIR arrays.
    room_results = {}  # Completed RIRs grouped by room until each pickle is written.
    expected_rirs_per_room = num_positions * num_sources_per_receiver

    # Use ProcessPoolExecutor for parallel processing
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        # Outer loop for different room geometries
        i = start_room
        with tqdm(total=end_room - start_room, desc="Generating Rooms") as pbar:
            while i < end_room:

                rtProfileIdx = 1 #np.random.randint(0, numRtProfiles)
                rt60 = rtProfiles[rtProfileIdx, :]

                #rt60 = random_rt_profile_with_long_band(rtProfiles,min_rt=1.0)

                while True:
                    # Generate room dimensions
                    Lx, Ly, Lz = get_random_dimensions()
                    room_dims = [Lx[0], Ly[0], Lz[0]]
                    try:
                        if shoebox:
                            floor_corners = None
                            room_test = pra.ShoeBox(
                                room_dims,
                                fs=sample_rate,
                                use_rand_ism=False,
                                max_order=0,
                            )

                        else:
                            room_shoebox = pra.ShoeBox(
                                room_dims,
                                fs=sample_rate,
                                use_rand_ism=False,
                                max_order=0,
                            )
                            floor_corners_shoebox = room_shoebox.walls[4].corners
                            floor_corners = floor_corners_shoebox[:2, :] + (np.random.rand(2, 4) * max_displacement_m)

                            room_test = pra.Room.from_corners(
                                floor_corners,
                                max_order=0,
                                fs=sample_rate,
                            )

                            room_test.extrude(room_dims[2])
                            room_test.set_sound_speed(c)

                        volume = float(room_test.get_volume())

                        # Same function for every room geometry
                        e_absorption = inverse_eyring(rt60, room_test, c=c)

                        # random anisotropy index between 0.5 and 1.5,
                        # which means 50% more or less absorption on one of the walls
                        #anisotropy_index = np.random.rand() + 0.5

                        if  anisotropic: # for half the samples, make
                            anisotropy_index  = np.random.rand() * 5 + 5 # between 5 and 15 times
                            if anisotropicType:
                                anisotropy_index  = 1 / anisotropy_index # more or less absorption
                        else: # for the other half, don't
                            anisotropy_index = 1

                        # random wall to have a different absorption
                        idxDifferentWall = np.random.randint(0, 6)

                        # get surface are as
                        area = [room_test.wall_area(room_test.walls[iWall]) for iWall in range(6)]

                        area_all = np.sum(np.array(area))
                        area_selected_wall = np.array(area[idxDifferentWall])
                        area_rest = area_all - area_selected_wall

                        e_absorption_rest = np.array(e_absorption * area_rest / (anisotropy_index * area_selected_wall + area_rest))
                        e_absorption_selected_wall = np.array(anisotropy_index * e_absorption_rest)

                        concat_absorption_values = np.hstack((e_absorption_rest, e_absorption_selected_wall))

                        if any(concat_absorption_values < np.array(0.01)) | any(concat_absorption_values > np.array(0.99)):
                            print('absorption values out of bounds, need to find a different room size')
                            continue # regenerate room dimensions and absorption values

                        break

                    except ValueError:
                        print('need to find a different room size')
                        continue
                
                # Inner loop for different source/mic positions within the current room
                accepted_receivers = []
                accepted_sources_per_receiver = []
                j = 0
                tries_receiver = 0
                position_generation_failed = False
                while j < num_positions and tries_receiver < 1000:
                    min_source_angle_difference = 15  # choose constraint
                    min_dist_from_receiver = 1.0 # in meters
                    max_dist_from_receiver = 50.0 # in meters
                    
                    tries_receiver += 1
                    try:
                        _, _, _, Rx, Ry, Rz = get_random_positions(
                            Lx[0], Ly[0], Lz[0], min_dist_from_wall=0.5,
                            floor_corners=floor_corners,
                        )
                    except ValueError as exc:
                        print(
                            f'Could not generate a valid receiver position for '
                            f'room {i}: {exc} Regenerating room dimensions.'
                        )
                        position_generation_failed = True
                        break
                    receiver = np.array([Rx[0], Ry[0], Rz[0]])
                    # Reject if receiver already exists
                    if any(np.allclose(receiver, r) for r in accepted_receivers):
                        continue

                    accepted_dirs = []
                    accepted_sources = []

                    k = 0
                    tries_source = 0
                    while k < num_sources_per_receiver and tries_source < 1000:
                        tries_source += 1
                        try:
                            Sx, Sy, Sz, _, _, _ = get_random_positions(
                                Lx[0], Ly[0], Lz[0], min_dist_from_wall=0.5,
                                floor_corners=floor_corners,
                            )
                        except ValueError as exc:
                            print(
                                f'Could not generate a valid source position for '
                                f'room {i}: {exc} Regenerating room dimensions.'
                            )
                            position_generation_failed = True
                            break
                        source = np.array([Sx[0], Sy[0], Sz[0]])

                        # avoid degenerate case
                        if np.allclose(source, receiver):
                            continue

                        dist = np.linalg.norm(source - receiver)
                        if dist < min_dist_from_receiver or dist > max_dist_from_receiver:
                            continue

                        direction = doa_unit_vector(source, receiver)

                        # check angular constraint
                        if is_too_close(direction, np.array(accepted_dirs), min_source_angle_difference):
                            continue

                        accepted_dirs.append(direction)
                        accepted_sources.append(source)
                        k += 1

                    if position_generation_failed:
                        break

                    if len(accepted_sources) < num_sources_per_receiver:
                        print(f'Could only find {len(accepted_sources)} sources for receiver {j} in room {i} after {tries_source} tries, regenerating receiver position.')
                        continue # regenerate source position for this room and position index

                    accepted_receivers.append(receiver)
                    accepted_sources_per_receiver.append(accepted_sources)
                    j += 1 # only increment position index if we have accepted sources

                if position_generation_failed:
                    continue # generate new room dimensions for the same room index

                if len(accepted_receivers) < num_positions:
                    print(f'Could only find {len(accepted_receivers)} receivers for room {i} after {tries_receiver} tries, regenerating room.')
                    continue # find a different room and start over with position generation

                for k, receiver in enumerate(accepted_receivers):
                    Rx = np.array([receiver[0]])
                    Ry = np.array([receiver[1]])
                    Rz = np.array([receiver[2]])

                    # Draw a new microphone-array yaw for every receiver position.
                    # All sources belonging to this receiver share the same orientation.
                    array_azimuth_rad = float(np.random.uniform(0.0, 2.0 * np.pi))
                    logging.info(
                        f"Room {i}, receiver position {k}: microphone-array azimuth = "
                        f"{np.degrees(array_azimuth_rad):.3f} deg "
                        f"({array_azimuth_rad:.6f} rad)"
                    )

                    for l, source in enumerate(accepted_sources_per_receiver[k]):
                        Sx = np.array([source[0]])
                        Sy = np.array([source[1]])
                        Sz = np.array([source[2]])

                        physical_params = [
                            Lx, Ly, Lz, Sx, Sy, Sz, Rx, Ry, Rz, rt60,
                            e_absorption_rest, e_absorption_selected_wall, centerFreqs,
                            idxDifferentWall, floor_corners, volume, array_azimuth_rad,
                        ]

                        futures.append(
                            executor.submit(
                                simulate_room,
                                i,
                                f"{k}_S{l}",
                                output_path,
                                physical_params,
                                array_type,
                                ambi_order,
                                sample_rate,
                                em32_directivity_path,
                                em32_radius,
                                amp_threshold,
                                em32_filter_length,
                                ism_order,
                                export_mode,
                                shoebox,
                                max_displacement_corners,
                                use_rand_ism,
                                max_rand_disp,
                                save_plots
                            )
                        )
                i += 1
                pbar.update(1)

        # Process the results as they complete. Each worker returns one RIR; the
        # parent groups all receiver/source combinations belonging to one room and
        # writes exactly one pickle for that room.
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Results"):
            result = future.result()
            if not result:
                continue

            if result.get("status") != "Success":
                all_results.append(result)
                continue

            room_idx = int(result["room_idx"])
            position_id = str(result["position_id"])

            room_entry = room_results.setdefault(
                room_idx,
                {
                    "room_idx": room_idx,
                    "sample_rate": int(result["sample_rate"]),
                    "rirs": {},
                },
            )

            # Store channels x samples without transposing or WAV quantization.
            metadata = {
                key: value
                for key, value in result.items()
                if key not in {"rir", "room_idx", "position_id", "sample_rate"}
            }
            room_entry["rirs"][position_id] = {
                "rir": result["rir"],
                "metadata": metadata,
            }

            # The CSV contains metadata only. Keeping the array out prevents pandas
            # from serializing large ndarray representations into cells.
            all_results.append(metadata_row(result))

            if len(room_entry["rirs"]) == expected_rirs_per_room:
                pickle_path = os.path.join(output_path, f"R{room_idx:04d}.pkl")
                with open(pickle_path, "wb") as pickle_file:
                    pickle.dump(room_entry, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

                logging.info(
                    "Saved room %d with %d RIRs to %s",
                    room_idx,
                    expected_rirs_per_room,
                    pickle_path,
                )
                del room_results[room_idx]

    # Save any room for which one or more simulations failed. These files are
    # explicitly marked incomplete instead of silently discarding valid RIRs.
    for room_idx, room_entry in room_results.items():
        pickle_path = os.path.join(output_path, f"R{room_idx:04d}_incomplete.pkl")
        with open(pickle_path, "wb") as pickle_file:
            pickle.dump(room_entry, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        logging.warning(
            "Saved incomplete room %d with %d of %d RIRs to %s",
            room_idx,
            len(room_entry["rirs"]),
            expected_rirs_per_room,
            pickle_path,
        )

    # Separate successful results from errors for the main data CSV
    successful_results = [res for res in all_results if res and res.get("status") == "Success"]
    error_results = [res for res in all_results if res and res.get("status") != "Success"]

    # Save successful results to the main data CSV
    if successful_results:
        data_df = pd.DataFrame(successful_results)

        # Keep CSV rows in deterministic room/receiver/source order.
        # Expected identifier format: R0000_P<receiver>_S<source>_IR
        sort_keys = data_df["Name"].str.extract(
            r"^R(?P<room>\d+)_P(?P<receiver>\d+)_S(?P<source>\d+)_IR$"
        ).astype("Int64")
        data_df = (
            data_df.assign(
                _room_order=sort_keys["room"],
                _receiver_order=sort_keys["receiver"],
                _source_order=sort_keys["source"],
            )
            .sort_values(
                ["_room_order", "_receiver_order", "_source_order"],
                kind="stable",
                na_position="last",
            )
            .drop(columns=["_room_order", "_receiver_order", "_source_order"])
            .reset_index(drop=True)
        )

        # Keep metadata columns in a predictable order.
        rt_columns = [col for col in data_df.columns if col.startswith("rt60_")]
        ordered_columns = [
            "Name", "room_idx", "position_id", "sample_rate", "RT", "c50",
            "ism_order", "late_reference_ism_order", "export_mode",
            "x_room", "y_room", "z_room",
            "is_shoebox", "max_displacement_corners", "use_rand_ism", "max_rand_ism_displacement", 
            "room_volume_m3", "room_corners",
            "x_source", "y_source", "z_source",
            "x_mic", "y_mic", "z_mic",
            "mic_orientation_az_deg",
            "accdoa_x", "accdoa_y", "accdoa_z",
            "az_deg", "el_deg",
            *rt_columns,
            "status",
        ]
        data_df = data_df[[col for col in ordered_columns if col in data_df.columns]]

        task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
        data_output_path = os.path.join(output_path, f"Generated_HOA_SRIR_data_task_{task_id}.csv")
        data_df.to_csv(data_output_path, index=False)
        logging.info(f"Successfully saved metadata for {len(successful_results)} RIRs to {data_output_path}")
    else:
        logging.warning("No successful RIR simulations to save metadata for.")

    # Save error log to a separate CSV
    if error_results:
        error_df = pd.DataFrame(error_results)
        error_log_path = os.path.join(output_path, "rir_error_log.csv")
        error_df.to_csv(error_log_path, index=False)
        logging.warning(f"Encountered {len(error_results)} errors during RIR generation. "
                        f"Error details saved to {error_log_path}")
    else:
        logging.info("No errors reported during RIR generation.")

    logging.info("RIR generation process completed.")

    # Print the summary and performance information to stderr
    print(file=sys.stderr)
    rusage_s = resource.getrusage(resource.RUSAGE_SELF)
    rusage_c = resource.getrusage(resource.RUSAGE_CHILDREN)
    print(f'Walltime {time.time() - start:.2f} s', file=sys.stderr)
    print(f'User time: {rusage_s.ru_utime + rusage_c.ru_utime:.2f} s ({rusage_s.ru_utime:.2f} + {rusage_c.ru_utime:.2f})', file=sys.stderr)
    print(f'System time: {rusage_s.ru_stime + rusage_c.ru_stime:.2f} s ({rusage_s.ru_stime:.2f} + {rusage_c.ru_stime:.2f})', file=sys.stderr)
    print(f'MaxRSS: {(rusage_s.ru_maxrss + rusage_c.ru_maxrss)/2**20:.3f} GiB ({rusage_s.ru_maxrss/2**20:.3f} + {rusage_c.ru_maxrss/2**20:.3f})', file=sys.stderr)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate Room Impulse Responses (RIRs) using pyroomacoustics.")

    # Add arguments for each parameter
    parser.add_argument(
        "--output_path",
        type=str,
        default="./Generated_HOA_IRs/", # Default value
        help="Directory to save one pickle per room and metadata CSV files."
    )
    parser.add_argument(
        "--num_rooms",
        type=int,
        default=2, # Default value
        help="Number of unique room geometries to simulate."
    )
    parser.add_argument(
        "--num_positions",
        type=int,
        default=10, # Default value
        help="Number of different source/mic positions within each room geometry."
    )
    parser.add_argument(
        "--ambi_order",
        type=int,
        default=3, # Default value (changed from 4 in your example to match previous code)
        help="The Ambisonic order for the microphone array (e.g., 0, 1, 2, 3, ...)."
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000, # Default value
        help="The sample rate for the simulations (e.g., 16000, 48000)."
    )
    parser.add_argument(
        "--num_sources_per_receiver",
        type=int,
        default=3,
        help="Number of random source positions per receiver position."
    )
    parser.add_argument(
        "--array_type",
        choices=("ambisonic", "em32"),
        default="ambisonic",
        help="Use the ideal HOA array or the measured Eigenmike EM32 array."
    )
    parser.add_argument(
        "--em32_directivity_path",
        type=str,
        default="Eigenmike_em32_IRs.npz",
        help="Path to Eigenmike_em32_IRs.npz (required when --array_type em32)."
    )
    parser.add_argument(
        "--em32_radius",
        type=float,
        default=0.042,
        help="Eigenmike capsule radius in metres."
    )
    parser.add_argument(
        "--amp_threshold",
        type=float,
        default=15.0,
        help="Maximum regularized SHT-filter amplification in dB."
    )

    parser.add_argument(
        "--em32_filter_length",
        type=int,
        default=1024,
        help="Even FIR/FFT length for the theoretical rigid-sphere EM32-to-SH encoder."
    )

    parser.add_argument(
        "--ism_order",
        type=int,
        default=12,
        help=(
            "Maximum image-source order of the exported early ISM. In "
            "'ism_late' mode, an internal order-12 ISM is additionally used "
            "only to estimate the stochastic late-reverberation level."
        ),
    )
    parser.add_argument(
        "--export_mode",
        choices=("ism", "ism_late"),
        default="ism_late",
        help=(
            "RIR to export: 'ism' exports only the ISM RIR; 'ism_late' "
            "exports ISM early + stochastic late reverb (+ mixing time); "
        ),
    )
    parser.add_argument(
        "--shoebox",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use rectangular ShoeBox rooms instead of randomized polygonal rooms.",
    )

    parser.add_argument(
        "--max_displacement_corners",
        type=float,
        default=0.5,
        help="Maximum non shoebox corner displacement in metres.",
    )

    parser.add_argument(
        "--use_rand_ism",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable randomized image-source positions.",
    )

    parser.add_argument(
        "--save_plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable saving of plots.",
    )

    parser.add_argument(
        "--max_rand_disp",
        type=float,
        default=0.1,
        help="Maximum random image-source displacement in metres.",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(
        output_path=args.output_path,
        num_rooms=args.num_rooms,
        num_positions=args.num_positions,
        ambi_order=args.ambi_order,
        sample_rate=args.sample_rate,
        num_sources_per_receiver=args.num_sources_per_receiver,
        array_type=args.array_type,
        em32_directivity_path=args.em32_directivity_path,
        em32_radius=args.em32_radius,
        amp_threshold=args.amp_threshold,
        em32_filter_length=args.em32_filter_length,
        ism_order=args.ism_order,
        export_mode=args.export_mode,
        shoebox=args.shoebox,
        max_displacement_corners=args.max_displacement_corners,
        use_rand_ism=args.use_rand_ism,
        max_rand_disp=args.max_rand_disp,
        save_plots=args.save_plots
    )
