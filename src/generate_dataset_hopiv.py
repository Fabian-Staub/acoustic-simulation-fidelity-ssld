# Corrected: HO-PIV + per-HOA-channel log-mel spectrograms,
# using a true mel axis, e.g. n_mels=128.
#
# Main change:
#   - HO-PIV is first computed on the STFT frequency bins: (50, 513, 2*(Q-1))
#   - HO-PIV is then mel-binned to:                      (50, n_mels, 2*(Q-1))
#   - Per-HOA-channel log-mel is computed as:             (50, n_mels, Q)
#   - Final feature is:                                   (50, n_mels, 2*(Q-1)+Q)
#
# This avoids librosa's "Empty filters detected" warning caused by n_mels=513
# at fs=16 kHz / n_fft=1024.

import os
import glob
import argparse
import json
import pickle
from functools import lru_cache
from collections.abc import Mapping
import re

import numpy as np
import pandas as pd
import librosa
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.io import wavfile
from scipy.fft import next_fast_len, rfft, irfft


# =========================
# Fixed configuration
# =========================

N_SPEAKERS = 3

# .pt output format:
#     (X, accdoa)
# where
#     X.shape      == (N, 50, n_mels, feature_channels)
#     accdoa.shape == (N, 50, 3, 3)
# No metadata dictionaries are stored in .pt files.


# =========================
# VAD / repeated speech track
# =========================

def get_timit_phn_path(wav_path):
    """Return the TIMIT .PHN path corresponding to a *.WAV.wav file."""
    wav_path = os.fspath(wav_path)
    suffix = ".WAV.wav"
    if not wav_path.lower().endswith(suffix.lower()):
        raise ValueError(f"Expected a TIMIT file ending in {suffix!r}: {wav_path}")
    return wav_path[:-len(suffix)] + ".PHN"


@lru_cache(maxsize=256)
def _load_timit_speech_and_activity_cached(wav_path, fs):
    """
    Load a mono 16-kHz TIMIT utterance and derive sample-level activity from
    its companion .PHN file. TIMIT interval end indices are exclusive.
    """
    fs_in, speech = wavfile.read(wav_path)
    if int(fs_in) != int(fs):
        raise ValueError(
            f"TIMIT speech must be {fs} Hz, but {wav_path} is {fs_in} Hz."
        )

    speech = np.asarray(speech)
    if speech.ndim != 1:
        raise ValueError(f"Expected mono TIMIT audio, got shape {speech.shape}.")

    if speech.dtype.kind in {"i", "u"}:
        if speech.dtype.kind == "u":
            info = np.iinfo(speech.dtype)
            midpoint = (info.max + 1) / 2.0
            speech = (speech.astype(np.float32) - midpoint) / midpoint
        else:
            scale = float(max(abs(np.iinfo(speech.dtype).min), np.iinfo(speech.dtype).max))
            speech = speech.astype(np.float32) / scale
    else:
        speech = speech.astype(np.float32, copy=False)

    peak = float(np.max(np.abs(speech))) if speech.size else 0.0
    if peak > 0.0:
        speech = speech / peak

    phn_path = get_timit_phn_path(wav_path)
    if not os.path.exists(phn_path):
        raise FileNotFoundError(f"Missing TIMIT phone annotation: {phn_path}")

    activity = np.zeros(len(speech), dtype=bool)
    silence_phones = {"h#", "pau", "epi"}

    with open(phn_path, "r", encoding="ascii") as phn_file:
        for line_number, line in enumerate(phn_file, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) != 3:
                raise ValueError(
                    f"Malformed PHN line {line_number} in {phn_path}: {line!r}"
                )
            start_sample, end_sample = int(fields[0]), int(fields[1])
            phone = fields[2].lower()
            start_sample = max(0, min(start_sample, len(speech)))
            end_sample = max(start_sample, min(end_sample, len(speech)))
            if phone not in silence_phones:
                activity[start_sample:end_sample] = True

    return (
        np.ascontiguousarray(speech, dtype=np.float32),
        np.ascontiguousarray(activity, dtype=bool),
        phn_path,
    )


def load_timit_speech_and_activity(wav_path, fs=16000):
    return _load_timit_speech_and_activity_cached(os.fspath(wav_path), int(fs))


def make_repeated_speech_track(
    speech,
    speech_vad_mask,
    fs,
    rng,
    max_pause_seconds=0.5,
):
    """
    Repeat one TIMIT utterance twice and add independent random silences at
    the beginning, between repetitions, and at the end, as described by
    Poschadel et al. Each silence is uniformly sampled from [0, 0.5] s.
    """
    speech = np.asarray(speech, dtype=np.float32)
    speech_vad_mask = np.asarray(speech_vad_mask, dtype=bool)

    if speech.shape != speech_vad_mask.shape:
        raise ValueError(
            f"Speech and PHN activity must have equal shape, got "
            f"{speech.shape} and {speech_vad_mask.shape}."
        )

    max_pause_seconds = max(0.0, float(max_pause_seconds))
    pause_lengths = [
        int(round(rng.uniform(0.0, max_pause_seconds) * fs))
        for _ in range(3)
    ]
    pause_start, pause_middle, pause_end = [
        np.zeros(length, dtype=np.float32) for length in pause_lengths
    ]
    inactive_start, inactive_middle, inactive_end = [
        np.zeros(length, dtype=bool) for length in pause_lengths
    ]

    track = np.concatenate(
        [pause_start, speech, pause_middle, speech, pause_end]
    ).astype(np.float32)
    sample_mask = np.concatenate(
        [inactive_start, speech_vad_mask, inactive_middle,
         speech_vad_mask, inactive_end]
    )

    pauses_s = {
        "pause_start_s": pause_lengths[0] / float(fs),
        "pause_middle_s": pause_lengths[1] / float(fs),
        "pause_end_s": pause_lengths[2] / float(fs),
    }
    return track, sample_mask, pauses_s


# =========================
# HOA helpers
# =========================

def infer_hoa_order(num_channels):
    sqrt_channels = int(round(np.sqrt(num_channels)))

    if sqrt_channels * sqrt_channels != num_channels:
        raise ValueError(
            f"Invalid HOA channel count: {num_channels}. "
            f"Expected a perfect square: 1, 4, 9, 16, 25, ..."
        )

    order = sqrt_channels - 1

    if order < 0:
        raise ValueError(
            f"Invalid HOA order inferred from {num_channels} channels."
        )

    return order


def truncate_hoa_to_order(signal, target_order, signal_name="HOA signal"):
    """
    Keep only the ACN channels required for ``target_order``.

    HOA order N uses the first (N + 1)^2 ACN channels. Higher-order
    channels are ignored. An input with insufficient order raises an error.
    """
    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim != 2:
        raise ValueError(
            f"Expected {signal_name} with shape (channels, samples), "
            f"got {signal.shape}."
        )

    target_order = int(target_order)
    if target_order < 0:
        raise ValueError(
            f"HOA order must be non-negative, got {target_order}."
        )

    available_channels = int(signal.shape[0])
    available_order = infer_hoa_order(available_channels)
    required_channels = (target_order + 1) ** 2

    if available_channels < required_channels:
        raise ValueError(
            f"{signal_name} contains {available_channels} channels "
            f"(HOA order {available_order}), but requested order "
            f"{target_order} requires {required_channels} channels."
        )

    return np.ascontiguousarray(
        signal[:required_channels, :],
        dtype=np.float32,
    )


def check_same_hoa_channel_count(signals):
    channel_counts = [sig.shape[0] for sig in signals]
    unique_channel_counts = sorted(set(channel_counts))

    if len(unique_channel_counts) != 1:
        raise ValueError(
            f"All RIRs must have the same HOA channel count. "
            f"Got channel counts: {unique_channel_counts}"
        )

    return unique_channel_counts[0]


# =========================
# Audio helpers
# =========================

@lru_cache(maxsize=256)
def _load_mono_speech_cached(path, fs):
    speech, _ = librosa.load(path, sr=fs, mono=True)
    speech = np.asarray(speech, dtype=np.float32)

    peak = np.max(np.abs(speech))

    if peak > 0:
        speech = speech / peak

    return np.ascontiguousarray(speech, dtype=np.float32)


def load_mono_speech(path, fs):
    """
    Cached speech loading/resampling.

    The returned array must be treated as read-only. Functions in this script
    do not modify it in place.
    """
    return _load_mono_speech_cached(os.fspath(path), int(fs))


@lru_cache(maxsize=1024)
def _read_wav_float_cached(path, target_fs):
    """
    Reads and optionally resamples a WAV once, then caches the float32 result.

    Returns:
        audio with shape (channels, samples)
    """
    fs_in, audio = wavfile.read(path)
    audio = np.asarray(audio)

    if audio.dtype.kind in {"i", "u"}:
        if audio.dtype.kind == "u":
            # Unsigned PCM is normally centered around half the integer range.
            info = np.iinfo(audio.dtype)
            midpoint = (info.max + 1) / 2.0
            audio = (audio.astype(np.float32) - midpoint) / midpoint
        else:
            scale = float(max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max))
            audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32, copy=False)

    if audio.ndim == 1:
        audio = audio[None, :]
    else:
        audio = audio.T

    if fs_in != target_fs:
        audio = librosa.resample(
            audio,
            orig_sr=fs_in,
            target_sr=target_fs,
            axis=1,
        ).astype(np.float32, copy=False)

    return np.ascontiguousarray(audio, dtype=np.float32)


def read_wav_float(path, target_fs):
    """
    Cached mono or multichannel WAV loading.

    The returned array must be treated as read-only.
    """
    return _read_wav_float_cached(os.fspath(path), int(target_fs))


def estimate_rir_delay_samples(rir, threshold_db=-40.0):
    rir = np.asarray(rir, dtype=np.float32)

    if rir.ndim != 2 or rir.shape[1] == 0:
        return 0

    envelope = np.sqrt(np.mean(rir ** 2, axis=0))
    peak = float(np.max(envelope))

    if peak <= 0.0:
        return 0

    threshold = peak * (10.0 ** (float(threshold_db) / 20.0))
    above = np.where(envelope >= threshold)[0]

    if len(above) == 0:
        return 0

    return int(above[0])


def _row_has_key(rir_row, key):
    """Return True for either pandas Series or generic mapping rows."""
    if isinstance(rir_row, pd.Series):
        return key in rir_row.index
    if isinstance(rir_row, Mapping):
        return key in rir_row
    return hasattr(rir_row, key)


def _finite_float(value):
    """Convert a scalar to float; return None for missing/non-finite values."""
    if value is None:
        return None
    try:
        # Avoid pd.isna(array) producing an array-valued truth expression.
        if np.ndim(value) != 0:
            return None
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None


def get_rt60_seconds_from_rir_row(rir_row, default_rt60=0.0):
    """Read a broadband RT value, with frequency-band fallbacks.

    The supplied room metadata uses ``RT`` as the broadband value and fields
    such as ``rt60_1000Hz`` for individual octave/frequency bands.
    """
    direct_keys = (
        "rt60_normalized", "RT60", "rt60", "RT", "rt",
        "rt60_1000Hz", "rt60_500Hz", "rt60_2000Hz",
    )
    for key in direct_keys:
        if _row_has_key(rir_row, key):
            value = _finite_float(rir_row.get(key))
            if value is not None and value >= 0.0:
                return value

    band_values = []
    keys = rir_row.index if isinstance(rir_row, pd.Series) else rir_row.keys()
    for key in keys:
        if re.fullmatch(r"rt60_\d+Hz", str(key), flags=re.IGNORECASE):
            value = _finite_float(rir_row.get(key))
            if value is not None and value >= 0.0:
                band_values.append(value)

    if band_values:
        return float(np.median(band_values))
    return float(default_rt60)


def get_rt60_bands_from_rir_row(rir_row):
    """Return valid ``{center_frequency_hz: rt60_seconds}`` CSV values."""
    bands = {}
    keys = rir_row.index if isinstance(rir_row, pd.Series) else rir_row.keys()
    for key in keys:
        match = re.fullmatch(r"rt60_(\d+)Hz", str(key), flags=re.IGNORECASE)
        if match is None:
            continue
        value = _finite_float(rir_row.get(key))
        if value is not None and value > 0.0:
            bands[int(match.group(1))] = float(value)
    return dict(sorted(bands.items()))


def average_rt60_bands_from_rir_rows(rir_rows):
    """Average each RT60 band across the source rows at one position."""
    values_by_band = {}
    for rir_row in rir_rows:
        if rir_row is None:
            continue
        for center_hz, rt60_s in get_rt60_bands_from_rir_row(rir_row).items():
            values_by_band.setdefault(center_hz, []).append(rt60_s)
    return {
        center_hz: float(np.mean(values))
        for center_hz, values in sorted(values_by_band.items())
    }


def delay_and_extend_activity_mask(
    mask,
    delay_samples,
    extension_samples,
    target_len=None,
):
    mask = np.asarray(mask, dtype=bool)

    if target_len is None:
        target_len = len(mask)

    target_len = int(target_len)
    delay_samples = max(0, int(delay_samples))
    extension_samples = max(0, int(extension_samples))

    corrected = np.zeros(target_len, dtype=bool)

    if target_len <= 0 or len(mask) == 0 or delay_samples >= target_len:
        return corrected

    padded = np.concatenate([[False], mask, [False]])
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    for start, end in zip(starts, ends):
        corrected_start = start + delay_samples
        corrected_end = end + delay_samples + extension_samples

        corrected_start = max(0, min(corrected_start, target_len))
        corrected_end = max(0, min(corrected_end, target_len))

        if corrected_end > corrected_start:
            corrected[corrected_start:corrected_end] = True

    return corrected


def convolve_mono_with_multichannel_rir(
    track,
    rir,
    keep_original_length=True,
):
    """
    FFT convolution of one mono track with all RIR channels at once.

    This is faster than calling scipy.signal.fftconvolve independently for
    every HOA channel because the speech FFT is computed only once.
    """
    track = np.asarray(track, dtype=np.float32)
    rir = np.asarray(rir, dtype=np.float32)

    if track.ndim != 1:
        raise ValueError(
            f"Expected mono track with shape (samples,), "
            f"but got shape {track.shape}."
        )

    if rir.ndim != 2:
        raise ValueError(
            f"Expected RIR with shape (channels, samples), "
            f"but got shape {rir.shape}."
        )

    infer_hoa_order(rir.shape[0])

    if track.size == 0 or rir.shape[1] == 0:
        output_len = track.size if keep_original_length else 0
        return np.zeros((rir.shape[0], output_len), dtype=np.float32)

    full_len = track.shape[0] + rir.shape[1] - 1
    fft_len = next_fast_len(full_len)

    track_fft = rfft(track, n=fft_len)
    rir_fft = rfft(rir, n=fft_len, axis=1)

    convolved = irfft(
        rir_fft * track_fft[None, :],
        n=fft_len,
        axis=1,
    )[:, :full_len]

    if keep_original_length:
        convolved = convolved[:, :track.shape[0]]

    return np.ascontiguousarray(convolved, dtype=np.float32)


def pad_multichannel_to_length(signal, target_len):
    signal = np.asarray(signal, dtype=np.float32)

    if signal.shape[1] >= target_len:
        return signal[:, :target_len].astype(np.float32)

    pad_len = target_len - signal.shape[1]

    return np.pad(
        signal,
        ((0, 0), (0, pad_len)),
        mode="constant",
    ).astype(np.float32)


def pad_bool_mask_to_length(mask, target_len):
    mask = np.asarray(mask, dtype=bool)

    if len(mask) >= target_len:
        return mask[:target_len]

    return np.pad(
        mask,
        (0, target_len - len(mask)),
        mode="constant",
        constant_values=False,
    )




def mean_active_power(signal, activity_mask, channel=0, eps=1e-12):
    """Mean-square power of one HOA channel over active samples only."""
    signal = np.asarray(signal, dtype=np.float32)
    activity_mask = np.asarray(activity_mask, dtype=bool)

    if signal.ndim != 2:
        raise ValueError(
            f"Expected signal with shape (channels, samples), got {signal.shape}."
        )
    if signal.shape[1] != activity_mask.size:
        raise ValueError(
            f"Signal and activity mask length differ: "
            f"{signal.shape[1]} != {activity_mask.size}."
        )
    if not 0 <= int(channel) < signal.shape[0]:
        raise ValueError(f"Invalid channel index {channel} for {signal.shape[0]} channels.")

    active = signal[int(channel), activity_mask].astype(np.float64)
    if active.size == 0:
        return 0.0

    power = float(np.mean(active * active))
    return power if np.isfinite(power) and power > eps else 0.0


def normalize_tracks_to_relative_sir(
    tracks,
    activity_masks,
    rng,
    sir_min_db=0.0,
    sir_max_db=10.0,
    reference_index=0,
    power_channel=0,
    eps=1e-12,
):
    """
    Scale every non-reference speaker relative to one reference speaker.

    Power is measured only over each speaker's own active samples, using the
    selected HOA channel (channel 0 / W by default). Thus inserted pauses do
    not affect the SIR normalization.
    """
    if len(tracks) != len(activity_masks):
        raise ValueError("tracks and activity_masks must have equal length.")
    if not tracks:
        return [], []

    reference_index = int(reference_index)
    if not 0 <= reference_index < len(tracks):
        raise ValueError(f"Invalid reference_index: {reference_index}.")

    scaled = [np.asarray(track, dtype=np.float32).copy() for track in tracks]
    powers_before = [
        mean_active_power(track, mask, channel=power_channel, eps=eps)
        for track, mask in zip(scaled, activity_masks)
    ]

    reference_power = powers_before[reference_index]
    if reference_power <= eps:
        raise ValueError(
            "Reference speaker has no measurable power in active samples."
        )

    normalization_info = []
    for speaker_index, (track, power) in enumerate(zip(scaled, powers_before)):
        if speaker_index == reference_index:
            target_sir_db = 0.0
            gain = 1.0
        elif power <= eps:
            target_sir_db = None
            gain = 0.0
            track *= 0.0
        else:
            target_sir_db = float(rng.uniform(sir_min_db, sir_max_db))
            gain = float(np.sqrt(
                reference_power / (power * 10.0 ** (target_sir_db / 10.0))
            ))
            track *= np.float32(gain)

        power_after = mean_active_power(
            track, activity_masks[speaker_index], channel=power_channel, eps=eps
        )
        achieved_sir_db = None
        if speaker_index != reference_index and power_after > eps:
            achieved_sir_db = float(
                10.0 * np.log10(reference_power / power_after)
            )

        normalization_info.append({
            "speaker_index": int(speaker_index),
            "reference_index": int(reference_index),
            "target_sir_db": target_sir_db,
            "achieved_sir_db": achieved_sir_db,
            "gain": float(gain),
            "active_power_before": float(power),
            "active_power_after": float(power_after),
            "power_channel": int(power_channel),
        })

    return scaled, normalization_info

def normalize_peak(signal, peak_value=0.9):
    signal = np.asarray(signal, dtype=np.float32)

    max_peak = np.max(np.abs(signal))

    if max_peak > 0:
        signal = peak_value * signal / max_peak

    return signal.astype(np.float32)


# =========================
# Diffuse babble-noise helpers
# =========================

def repeat_to_length(signal, target_len, random_offset=0, eps=1e-12):
    """Repeat a mono signal until ``target_len`` samples are available."""
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    target_len = int(target_len)
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if signal.size == 0 or float(np.max(np.abs(signal))) <= eps:
        return np.zeros(target_len, dtype=np.float32)

    offset = int(random_offset) % signal.size
    if offset:
        signal = np.concatenate([signal[offset:], signal[:offset]])
    repeats = int(np.ceil(target_len / signal.size))
    return np.tile(signal, repeats)[:target_len].astype(np.float32, copy=False)


def create_repeated_babble_noise(
    speech_files,
    target_len,
    fs,
    rng,
    num_speakers=50,
):
    """
    Create an anechoic babble track by summing randomly selected TIMIT speakers.

    Every selected utterance is repeated to the full mixture length. A random
    circular starting offset prevents repetition boundaries from lining up.
    The sum is divided by sqrt(num_speakers), preserving approximately stable
    variance while retaining dense, practically pause-free babble.
    """
    if not speech_files:
        raise ValueError("No speech files available for babble generation.")
    num_speakers = int(num_speakers)
    if num_speakers <= 0:
        return np.zeros(int(target_len), dtype=np.float32), []

    replace = len(speech_files) < num_speakers
    selected = rng.choice(speech_files, size=num_speakers, replace=replace)
    babble = np.zeros(int(target_len), dtype=np.float64)

    for path in selected:
        utterance = load_mono_speech(path, fs)
        offset = int(rng.integers(0, max(1, len(utterance))))
        repeated = repeat_to_length(utterance, target_len, random_offset=offset)
        rms = float(np.sqrt(np.mean(repeated.astype(np.float64) ** 2)))
        if rms > 1e-12:
            repeated = repeated / np.float32(rms)
        babble += repeated.astype(np.float64)

    babble /= np.sqrt(float(num_speakers))
    babble -= float(np.mean(babble))
    return babble.astype(np.float32), [os.fspath(x) for x in selected]


def create_frequency_dependent_diffuse_rir(
    num_channels,
    rt60_by_band_s,
    fs,
    rng,
    max_duration_s=4.0,
):
    """
    Create an N3D/ACN diffuse late-reverberation model from band RT60s.

    Each FFT bin is assigned to the closest supplied octave-band center on a
    logarithmic frequency axis. Independent band-limited Gaussian noise is then
    multiplied by that band's -60 dB decay envelope. Bands above Nyquist are
    retained in metadata but do not contribute samples at this sample rate.
    """
    num_channels = int(num_channels)
    infer_hoa_order(num_channels)

    bands = {
        int(center_hz): float(rt60_s)
        for center_hz, rt60_s in dict(rt60_by_band_s).items()
        if int(center_hz) > 0
        and np.isfinite(float(rt60_s))
        and float(rt60_s) > 0.0
    }
    if not bands:
        raise ValueError("At least one positive finite per-band RT60 is required.")

    fs = int(fs)
    max_rt60_s = max(bands.values())
    rir_len = max(2, int(round(min(float(max_duration_s), max_rt60_s) * fs)))
    t = np.arange(rir_len, dtype=np.float64) / float(fs)

    centers_hz = np.asarray(sorted(bands), dtype=np.float64)
    frequencies_hz = np.fft.rfftfreq(rir_len, d=1.0 / float(fs))
    positive_frequencies = np.maximum(frequencies_hz, np.finfo(np.float64).tiny)
    distances = np.abs(
        np.log2(positive_frequencies[:, None] / centers_hz[None, :])
    )
    bin_band_indices = np.argmin(distances, axis=1)

    rir = np.zeros((num_channels, rir_len), dtype=np.float64)
    for band_index, center_hz in enumerate(centers_hz.astype(int)):
        band_mask = bin_band_indices == band_index
        if not np.any(band_mask):
            continue
        noise_spectrum = rfft(
            rng.standard_normal((num_channels, rir_len)), axis=1
        )
        noise_spectrum[:, ~band_mask] = 0.0
        band_noise = irfft(noise_spectrum, n=rir_len, axis=1)

        # Amplitude reaches -60 dB (= 1/1000) at this band's RT60.
        envelope = np.exp(-np.log(1000.0) * t / bands[center_hz])
        rir += band_noise * envelope[None, :]

    # Remove any impulse-like common component and equalize channel energy.
    rir[:, 0] = 0.0
    rms = np.sqrt(np.mean(rir * rir, axis=1, keepdims=True))
    rir /= np.maximum(rms, 1e-12)
    return np.ascontiguousarray(rir, dtype=np.float32)


def add_diffuse_babble_noise(
    clean_ambisonics,
    speech_activity_mask,
    speech_files,
    rt60_by_band_s,
    fs,
    rng,
    num_babble_speakers=50,
    snr_db=20.0,
):
    """Add diffuse babble at the requested active-speech SNR.

    Power is measured in the N3D omnidirectional channel (ACN 0) only over
    samples where at least one target speaker is active. Despite the historical
    variable name "STI" in the request, this is an SNR/power calculation, not
    the Speech Transmission Index metric.
    """
    clean = np.asarray(clean_ambisonics, dtype=np.float32)
    activity = np.asarray(speech_activity_mask, dtype=bool)
    if clean.ndim != 2 or clean.shape[1] != activity.size:
        raise ValueError("clean_ambisonics and speech_activity_mask mismatch.")

    dry_babble, selected = create_repeated_babble_noise(
        speech_files=speech_files, target_len=clean.shape[1], fs=fs, rng=rng,
        num_speakers=num_babble_speakers,
    )
    diffuse_rir = create_frequency_dependent_diffuse_rir(
        num_channels=clean.shape[0], rt60_by_band_s=rt60_by_band_s,
        fs=fs, rng=rng,
    )
    diffuse_babble = convolve_mono_with_multichannel_rir(
        dry_babble, diffuse_rir, keep_original_length=True
    )

    if not np.any(activity):
        gain = 0.0
        speech_power = babble_power = 0.0
    else:
        speech_power = mean_active_power(clean, activity, channel=0)
        babble_power = mean_active_power(diffuse_babble, activity, channel=0)
        if speech_power <= 1e-12 or babble_power <= 1e-12:
            gain = 0.0
        else:
            gain = float(np.sqrt(
                speech_power / (babble_power * 10.0 ** (float(snr_db) / 10.0))
            ))
    diffuse_babble *= np.float32(gain)
    mixed = clean + diffuse_babble
    achieved_snr = None
    scaled_babble_power = mean_active_power(diffuse_babble, activity, channel=0)
    if speech_power > 1e-12 and scaled_babble_power > 1e-12:
        achieved_snr = float(10.0 * np.log10(speech_power / scaled_babble_power))

    info = {
        "enabled": True,
        "num_babble_speakers": int(num_babble_speakers),
        "rt60_by_band_s": {
            f"{int(center_hz)}Hz": float(rt60_s)
            for center_hz, rt60_s in sorted(rt60_by_band_s.items())
        },
        "target_snr_db": float(snr_db),
        "achieved_snr_db": achieved_snr,
        "gain": float(gain),
        "speech_active_power_omni": float(speech_power),
        "babble_active_power_omni_before": float(babble_power),
        "babble_active_power_omni_after": float(scaled_babble_power),
        "selected_speech_files": selected,
    }
    return mixed.astype(np.float32), diffuse_babble.astype(np.float32), info


# =========================
# Visualization helper
# =========================

def plot_first_realigned_multitrack_sample(
    convolved_tracks,
    speaker_activity_masks,
    speaker_metadata,
    fs,
    output_path=None,
    max_seconds=None,
):
    """
    Plot the first synthesized multitrack example after RIR/VAD realignment.

    Each row shows the omnidirectional HOA channel (channel 0) of one
    reverberant speaker track. Corrected voice-active samples are shaded.

    All subplots use the same y-axis range, determined by the loudest
    displayed audio track.
    """
    if len(convolved_tracks) != len(speaker_activity_masks):
        raise ValueError(
            "convolved_tracks and speaker_activity_masks must have equal length."
        )

    n_tracks = len(convolved_tracks)
    if n_tracks == 0:
        return

    n_samples = min(track.shape[1] for track in convolved_tracks)
    if max_seconds is not None:
        n_samples = min(n_samples, int(float(max_seconds) * fs))

    time_s = np.arange(n_samples, dtype=np.float32) / float(fs)

    # ------------------------------------------------------------------
    # Determine a common y-axis range from the loudest displayed track.
    # ------------------------------------------------------------------
    max_amplitude = max(
        float(np.max(np.abs(np.asarray(track[0, :n_samples]))))
        for track in convolved_tracks
    )

    # Avoid a degenerate y-axis for silent tracks.
    if max_amplitude <= 0:
        max_amplitude = 1.0

    # Add a little visual headroom.
    y_limit = 1.08 * max_amplitude

    # Slightly more compact figure while keeping text readable when
    # displayed at a small size.
    fig, axes = plt.subplots(
        n_tracks,
        1,
        figsize=(14, 2 * n_tracks),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for speaker_idx, (track, mask, meta) in enumerate(
        zip(convolved_tracks, speaker_activity_masks, speaker_metadata)
    ):
        waveform = np.asarray(track[0, :n_samples], dtype=np.float32)
        active = np.asarray(mask[:n_samples], dtype=bool)

        ax = axes[speaker_idx]

        ax.plot(
            time_s,
            waveform,
            linewidth=1.0,
            zorder=2,
        )

        # Shade the entire vertical extent of the waveform plot.
        ax.fill_between(
            time_s,
            -y_limit,
            y_limit,
            where=active,
            alpha=0.18,
            step="pre",
            linewidth=0,
            label="Adjusted voice activity region",
            zorder=1,
        )

        ax.set_ylim(-y_limit, y_limit)

        ax.set_ylabel(
            f"Amplitude",
            fontsize=14,
            fontweight="regular",
        )

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=12,
        )

        ax.grid(
            True,
            alpha=0.22,
            linewidth=0.7,
        )

        delay_ms = 1000.0 * float(meta.get("rir_delay_s", 0.0))
        extension_ms = 1000.0 * float(meta.get("vad_extension_s", 0.0))

        ax.set_title(
            f"Speaker track {speaker_idx + 1}",
            fontsize=14,
            fontweight="semibold",
            pad=6,
        )

        ax.legend(
            loc="upper right",
            fontsize=14,
            framealpha=0.9,
            ncol=2,
        )

    axes[-1].set_xlabel(
        "Time [s]",
        fontsize=14,
        fontweight="regular",
    )


    if output_path is not None:
        output_path = os.fspath(output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(
            output_path,
            dpi=220,
            bbox_inches="tight",
        )
        print(f"Saved first-sample waveform/VAD plot to: {output_path}")

    plt.show()
    plt.close(fig)

# =========================
# HO-PIV / ACCDOA helpers
# =========================

def get_accdoa_from_rir_row(rir_row):
    """Read the three-component ACCDOA vector from the room metadata.

    Supported layouts:
      1. ``accdoa=[x, y, z]``
      2. ``accdoa={x: ..., y: ..., z: ...}``
      3. Separate scalar fields ``accdoa_x``, ``accdoa_y``, ``accdoa_z``

    Layout 3 is the format used by the metadata supplied with this request.
    """
    name = rir_row.get("Name", "<unknown>")
    value = rir_row.get("accdoa", None)

    if value is None and all(
        _row_has_key(rir_row, key)
        for key in ("accdoa_x", "accdoa_y", "accdoa_z")
    ):
        value = [
            rir_row.get("accdoa_x"),
            rir_row.get("accdoa_y"),
            rir_row.get("accdoa_z"),
        ]

    if isinstance(value, Mapping):
        component_sets = (
            ("x", "y", "z"),
            ("accdoa_x", "accdoa_y", "accdoa_z"),
        )
        for keys in component_sets:
            if all(key in value for key in keys):
                value = [value[key] for key in keys]
                break

    if isinstance(value, str):
        stripped = value.strip().strip("[]()")
        value = np.fromstring(stripped.replace(",", " "), sep=" ")

    if value is None:
        raise KeyError(
            f"RIR row {name} is missing ACCDOA metadata. Expected either "
            "'accdoa' or the fields 'accdoa_x', 'accdoa_y', 'accdoa_z'."
        )

    accdoa = np.asarray(value, dtype=np.float32).reshape(-1)
    if accdoa.size != 3 or not np.isfinite(accdoa).all():
        raise ValueError(
            f"RIR row {name} has invalid ACCDOA metadata {value!r}; "
            "expected three finite components."
        )

    # ACCDOA direction vectors should have norm <= 1. Small numerical excess is
    # harmless; a clearly invalid vector is rejected instead of silently scaled.
    norm = float(np.linalg.norm(accdoa))
    if norm > 1.01:
        raise ValueError(
            f"RIR row {name} has ACCDOA norm {norm:.6f} > 1.01: {accdoa.tolist()}"
        )

    return np.ascontiguousarray(accdoa, dtype=np.float32)


def get_az_el_deg_from_rir_row(rir_row):
    """Read azimuth and elevation in degrees directly from PKL metadata."""
    name = rir_row.get("Name", "<unknown>")
    key_pairs = (("az", "el"), ("az_deg", "el_deg"))

    for az_key, el_key in key_pairs:
        az_value = rir_row.get(az_key, None)
        el_value = rir_row.get(el_key, None)
        if pd.notna(az_value) and pd.notna(el_value):
            az_deg = float(az_value)
            el_deg = float(el_value)
            if np.isfinite(az_deg) and np.isfinite(el_deg):
                return az_deg, el_deg

    raise KeyError(
        f"RIR row {name} is missing finite PKL az/el metadata "
        "('az' and 'el', or 'az_deg' and 'el_deg')."
    )


def optional_float_from_rir_row(rir_row, key):
    """Return an optional finite scalar metadata value as float."""
    return _finite_float(rir_row.get(key, None))


def public_metadata_from_rir_row(rir_row):
    """Copy scalar/list metadata while excluding internal loader columns."""
    result = {}
    items = rir_row.items() if hasattr(rir_row, "items") else []
    excluded = {"r", "p", "s", "original_room_idx", "rt60_normalized"}
    for key, value in items:
        key = str(key)
        if key.startswith("_") or key in excluded:
            continue
        if isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, np.ndarray):
            if value.ndim == 0:
                value = value.item()
            else:
                value = value.tolist()
        result[key] = value
    return result


def make_sample_id(room, position):
    return f"R{int(room):04d}_P{int(position):02d}"


@lru_cache(maxsize=32)
def _cached_hann_window(frame_length):
    return np.hanning(int(frame_length)).astype(np.float32)


def stft_hoa_fixed_num_frames(
    hoa_signal,
    num_frames=50,
    hop_length=320,
    frame_length=640,
    n_fft=1024,
):
    """
    Vectorized fixed-frame STFT.

    Input:
        hoa_signal:
            shape (channels, samples)

    Output:
        complex STFT:
            shape (time_frames, frequency_bins, channels)
    """
    hoa_signal = np.asarray(hoa_signal, dtype=np.float32)

    if hoa_signal.ndim != 2:
        raise ValueError(
            f"Expected HOA signal with shape (channels, samples), "
            f"but got shape {hoa_signal.shape}."
        )

    if num_frames <= 0:
        return np.empty(
            (0, n_fft // 2 + 1, hoa_signal.shape[0]),
            dtype=np.complex64,
        )

    required_len = (num_frames - 1) * hop_length + frame_length
    num_samples = hoa_signal.shape[1]

    if num_samples < required_len:
        work = np.pad(
            hoa_signal,
            ((0, 0), (0, required_len - num_samples)),
            mode="constant",
        )
    else:
        work = hoa_signal[:, :required_len]

    frames = np.lib.stride_tricks.sliding_window_view(
        work,
        window_shape=frame_length,
        axis=1,
    )[:, ::hop_length, :][:, :num_frames, :]

    window = _cached_hann_window(frame_length)
    windowed = frames * window[None, None, :]

    spec = np.fft.rfft(windowed, n=n_fft, axis=-1)

    # (Q, T, F) -> (T, F, Q)
    return np.ascontiguousarray(
        spec.transpose(1, 2, 0),
        dtype=np.complex64,
    )


@lru_cache(maxsize=32)
def _make_mel_basis_cached(
    fs,
    n_fft,
    n_mels,
    fmin,
    fmax,
):
    mel_basis = librosa.filters.mel(
        sr=int(fs),
        n_fft=int(n_fft),
        n_mels=int(n_mels),
        fmin=float(fmin),
        fmax=float(fmax),
        norm=None,
        dtype=np.float32,
    )

    empty = np.where(np.sum(mel_basis, axis=1) <= 0.0)[0]

    if len(empty) > 0:
        raise ValueError(
            f"Mel basis contains {len(empty)} empty filters. "
            f"Reduce n_mels, increase n_fft, or increase fs/fmax. "
            f"Current: fs={fs}, n_fft={n_fft}, n_mels={n_mels}, fmax={fmax}."
        )

    return np.ascontiguousarray(mel_basis, dtype=np.float32)


def make_mel_basis(
    fs,
    n_fft=1024,
    n_mels=128,
    fmin=0.0,
    fmax=None,
):
    """
    Cached mel basis with shape (n_mels, n_fft//2 + 1).

    The returned array must be treated as read-only.
    """
    if fmax is None:
        fmax = fs / 2.0

    return _make_mel_basis_cached(
        int(fs),
        int(n_fft),
        int(n_mels),
        float(fmin),
        float(fmax),
    )


def compute_hopiv_logmel_from_hoa_segment(
    hoa_segment,
    fs,
    hop_length=320,
    frame_length=640,
    n_fft=1024,
    n_mels=128,
    eps=1e-8,
):
    """
    Computes mel-binned HO-PIV and per-HOA-channel log-mel spectrograms.

    Input:
        hoa_segment:
            Shape (Q, samples), where Q = (N + 1)^2.

    Output:
        X:
            Shape (50, n_mels, 2 * (Q - 1) + Q)

            channels/features:
                [0 : Q-1]                 -> active HO-PIV, mel-binned
                [Q-1 : 2*(Q-1)]           -> reactive HO-PIV, mel-binned
                [2*(Q-1) : 2*(Q-1)+Q]     -> log-mel per HOA channel

    Notes:
        HO-PIV itself is still computed using the original STFT bins and the
        original HO-PIV energy normalization E_N(k,l). After that, the
        frequency-bin HO-PIV feature is projected onto the mel axis so it can
        be concatenated with log-mel on the same second dimension.
    """
    hoa_segment = np.asarray(hoa_segment, dtype=np.float32)

    if hoa_segment.ndim != 2:
        raise ValueError(
            f"Expected hoa_segment with shape (channels, samples), "
            f"got {hoa_segment.shape}."
        )

    num_channels = hoa_segment.shape[0]
    hoa_order = infer_hoa_order(num_channels)

    A = stft_hoa_fixed_num_frames(
        hoa_signal=hoa_segment,
        num_frames=50,
        hop_length=hop_length,
        frame_length=frame_length,
        n_fft=n_fft,
    )

    # A shape: (T, F, Q)
    a0 = A[:, :, 0:1]
    a_rest = A[:, :, 1:]

    I_hat = -a0 * np.conj(a_rest)

    I_active = np.real(I_hat).astype(np.float32)
    I_reactive = np.imag(I_hat).astype(np.float32)

    # Energy normalization:
    # E_N = sum_n 1/(2n+1) * sum_m |a_nm|^2
    E = np.zeros(
        A.shape[:2] + (1,),
        dtype=np.float32,
    )

    for n in range(hoa_order + 1):
        start_ch = n * n
        end_ch = (n + 1) * (n + 1)

        E += (
            np.sum(
                np.abs(A[:, :, start_ch:end_ch]) ** 2,
                axis=-1,
                keepdims=True,
            )
            / float(2 * n + 1)
        ).astype(np.float32)

    ho_piv_freq = -np.concatenate(
        [I_active, I_reactive],
        axis=-1,
    ) / np.maximum(E, eps)

    # ho_piv_freq shape: (T, F, 2*(Q-1))

    mel_basis = make_mel_basis(
        fs=fs,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=0.0,
        fmax=fs / 2.0,
    )

    # Mel-bin HO-PIV:
    # mel_basis:   (M, F)
    # ho_piv_freq: (T, F, C)
    # output:      (T, M, C)
    #
    # Because HO-PIV can be signed, we normalize the mel filters by their
    # row sums before applying them to avoid scale inflation.
    mel_basis_norm = mel_basis / np.maximum(
        np.sum(mel_basis, axis=1, keepdims=True),
        eps,
    )

    ho_piv_mel = np.einsum(
        "mf,tfc->tmc",
        mel_basis_norm,
        ho_piv_freq,
        optimize=True,
    ).astype(np.float32)

    # Per-HOA-channel log-mel power:
    # power:       (T, F, Q)
    # mel_basis:   (M, F)
    # mel_power:   (T, M, Q)
    power = (np.abs(A) ** 2).astype(np.float32)

    mel_power = np.einsum(
        "mf,tfq->tmq",
        mel_basis,
        power,
        optimize=True,
    ).astype(np.float32)

    logmel = np.log(np.maximum(mel_power, eps)).astype(np.float32)

    X = np.concatenate(
        [ho_piv_mel, logmel],
        axis=-1,
    ).astype(np.float32)

    return X


def make_accdoa_for_segment(
    speaker_activity_masks,
    speaker_directions,
    segment_start_sample,
    num_frames=50,
    hop_length=320,
    frame_length=640,
):
    """
    Vectorized frame activity calculation using cumulative sums.
    """
    directions = np.asarray(speaker_directions, dtype=np.float32)

    if directions.shape != (N_SPEAKERS, 3):
        raise ValueError(
            f"Expected speaker_directions shape {(N_SPEAKERS, 3)}, "
            f"got {directions.shape}."
        )

    frame_starts = (
        int(segment_start_sample)
        + np.arange(num_frames, dtype=np.int64) * int(hop_length)
    )
    frame_ends = frame_starts + int(frame_length)

    activity = np.zeros((num_frames, N_SPEAKERS), dtype=bool)

    for speaker_idx, mask in enumerate(speaker_activity_masks):
        mask = np.asarray(mask, dtype=np.uint8)
        cumulative = np.empty(mask.size + 1, dtype=np.int64)
        cumulative[0] = 0
        np.cumsum(mask, dtype=np.int64, out=cumulative[1:])

        starts = np.clip(frame_starts, 0, mask.size)
        ends = np.clip(frame_ends, 0, mask.size)

        active_samples = cumulative[ends] - cumulative[starts]
        frame_lengths = ends - starts

        activity[:, speaker_idx] = (
            active_samples > 0 #>= 0.5 * frame_lengths
        )


    return (
        activity[:, :, None] * directions[None, :, :]
    ).astype(np.float32, copy=False)


def save_hopiv_samples_for_mixture(
    ambisonics,
    speaker_activity_masks,
    speaker_directions,
    speaker_metadata,
    room,
    position,
    fs,
    output_wav_name,
    pt_batch_name,
    hop_length=320,
    frame_length=640,
    n_fft=1024,
    n_mels=128,
    overlap_frame_counts=None,
):
    # One-second segment stride, but enough analysis context for all STFT frames.
    # With 50 frames, hop=320, and frame_length=640, the feature window spans
    # (50 - 1) * 320 + 640 = 16320 samples.  Using only 16000 samples would
    # zero-pad the second half of the final STFT frame while ACCDOA labels still
    # read real activity from the following samples, causing feature/label
    # misalignment.
    segment_stride_samples = int(fs)
    num_frames = 50
    analysis_len_samples = (num_frames - 1) * hop_length + frame_length
    min_keep_samples = 10 * hop_length

    total_len = ambisonics.shape[1]
    sample_id = make_sample_id(room, position)

    sample_records = []
    csv_rows = []

    start = 0
    segment_idx = 0

    while start < total_len:
        remaining = total_len - start

        # Keep the original one-second segmentation policy for deciding whether
        # to retain a final partial example, but provide enough right context for
        # all 50 analysis frames.  Any unavailable context at the very end of the
        # recording is zero-padded consistently for both features and labels.
        if remaining < segment_stride_samples and remaining < min_keep_samples:
            break

        available_analysis_samples = min(remaining, analysis_len_samples)
        padded = available_analysis_samples < analysis_len_samples

        segment = np.zeros(
            (ambisonics.shape[0], analysis_len_samples),
            dtype=np.float32,
        )
        segment[:, :available_analysis_samples] = ambisonics[
            :, start:start + available_analysis_samples
        ]

        X = compute_hopiv_logmel_from_hoa_segment(
            hoa_segment=segment,
            fs=fs,
            hop_length=hop_length,
            frame_length=frame_length,
            n_fft=n_fft,
            n_mels=n_mels,
        )


        accdoa = make_accdoa_for_segment(
            speaker_activity_masks=speaker_activity_masks,
            speaker_directions=speaker_directions,
            segment_start_sample=start,
            num_frames=num_frames,
            hop_length=hop_length,
            frame_length=frame_length,
        )

        # Count how many sources are active in each frame (0, 1, 2, or 3).
        # A source is active when its ACCDOA vector has non-zero magnitude.
        if overlap_frame_counts is not None:
            active_per_frame = np.sum(
                np.linalg.norm(accdoa, axis=-1) > 0.5,
                axis=1,
            )

            for overlap_count in range(N_SPEAKERS + 1):
                overlap_frame_counts[overlap_count] += int(
                    np.sum(active_per_frame == overlap_count)
                )

        base_name = f"{sample_id}_segment{segment_idx:04d}"

        # Store only the tensors needed for training.
        # Metadata remains available in the CSV output.
        sample_records.append(
            (
                torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(accdoa, dtype=np.float32)),
            )
        )

        for frame_idx in range(num_frames):
            frame_start_sample = start + frame_idx * hop_length
            frame_start_s = frame_start_sample / fs

            for slot in range(N_SPEAKERS):
                meta = speaker_metadata[slot]
                rir_meta = meta["rir_metadata"]

                active = int(np.linalg.norm(accdoa[frame_idx, slot, :]) > 0.5)

                accdoa_x = float(accdoa[frame_idx, slot, 0])
                accdoa_y = float(accdoa[frame_idx, slot, 1])
                accdoa_z = float(accdoa[frame_idx, slot, 2])

                az_deg = rir_meta.get("az", None)
                el_deg = rir_meta.get("el", None)
                mic_orientation_az_deg = rir_meta.get("mic_orientation_az_deg", None)

                csv_rows.append(
                    {
                        "sample_id": sample_id,
                        "segment_idx": int(segment_idx),
                        "base_name": base_name,

                        "segment_start_sample": int(start),
                        "segment_start_s": float(start / fs),

                        "frame_idx": int(frame_idx),
                        "frame_start_sample": int(frame_start_sample),
                        "frame_start_s": float(frame_start_s),

                        "slot": int(slot),
                        "speech_path": meta["speech_path"],

                        "active": int(active),

                        # Frame-level ACCDOA target (zero while inactive).
                        "accdoa_x": accdoa_x,
                        "accdoa_y": accdoa_y,
                        "accdoa_z": accdoa_z,

                        # Static source direction copied from RIR metadata.
                        "source_accdoa_x": rir_meta.get("accdoa_x", None),
                        "source_accdoa_y": rir_meta.get("accdoa_y", None),
                        "source_accdoa_z": rir_meta.get("accdoa_z", None),

                        "az_deg": None if az_deg is None else round(float(az_deg), 3),
                        "el_deg": None if el_deg is None else round(float(el_deg), 3),
                        "mic_orientation_az_deg": (
                            None
                            if mic_orientation_az_deg is None
                            else round(float(mic_orientation_az_deg), 3)
                        ),

                        "pause_start_s": float(meta.get("pause_start_s", 0.0)),
                        "pause_middle_s": float(meta.get("pause_middle_s", 0.0)),
                        "pause_end_s": float(meta.get("pause_end_s", 0.0)),

                        "rir_delay_samples": int(meta.get("rir_delay_samples", 0)),
                        "rir_delay_s": float(meta.get("rir_delay_s", 0.0)),
                        "rt60_s": float(meta.get("rt60_s", 0.0)),
                        "vad_extension_samples": int(meta.get("vad_extension_samples", 0)),
                        "vad_extension_s": float(meta.get("vad_extension_s", 0.0)),

                        "Name": rir_meta.get("Name", meta["rir_file"]),
                        "source_id_raw": rir_meta.get("source_id_raw", None),
                        "source_index_offset": rir_meta.get("source_index_offset", None),
                        "RT": rir_meta.get("RT", None),
                        "c50": rir_meta.get("c50", None),
                        "export_mode": rir_meta.get("export_mode", None),
                        "ism_order": rir_meta.get("ism_order", None),
                        "rt60_125Hz": rir_meta.get("rt60_125Hz", None),
                        "rt60_250Hz": rir_meta.get("rt60_250Hz", None),
                        "rt60_500Hz": rir_meta.get("rt60_500Hz", None),
                        "rt60_1000Hz": rir_meta.get("rt60_1000Hz", None),
                        "rt60_2000Hz": rir_meta.get("rt60_2000Hz", None),
                        "rt60_4000Hz": rir_meta.get("rt60_4000Hz", None),
                        "rt60_8000Hz": rir_meta.get("rt60_8000Hz", None),
                        "rt60_16000Hz": rir_meta.get("rt60_16000Hz", None),

                        "x_room": rir_meta.get("x_room", None),
                        "y_room": rir_meta.get("y_room", None),
                        "z_room": rir_meta.get("z_room", None),

                        "x_source": rir_meta.get("x_source", None),
                        "y_source": rir_meta.get("y_source", None),
                        "z_source": rir_meta.get("z_source", None),

                        "x_mic": rir_meta.get("x_mic", None),
                        "y_mic": rir_meta.get("y_mic", None),
                        "z_mic": rir_meta.get("z_mic", None),

                        "status": rir_meta.get("status", None),
                    }
                )

        segment_idx += 1
        start += segment_stride_samples

    return sample_records, csv_rows


def save_room_batch_pt(pt_batch_folder, batch_idx, room_records):
    """
    Save one batch as a tuple of two dense tensors:

        X:      (N, 50, n_mels, feature_channels)
        accdoa: (N, 50, 3, 3)

    No room/sample metadata is stored in the .pt file.
    """
    if len(room_records) == 0:
        return None

    os.makedirs(pt_batch_folder, exist_ok=True)

    samples = []
    for room_record in room_records:
        samples.extend(room_record["samples"])

    if not samples:
        return None

    X = torch.stack([sample[0] for sample in samples], dim=0).contiguous()
    accdoa = torch.stack([sample[1] for sample in samples], dim=0).contiguous()

    batch_name = f"rooms_batch{batch_idx:04d}.pt"
    batch_path = os.path.join(pt_batch_folder, batch_name)

    torch.save((X, accdoa), batch_path)

    return batch_name


# =========================
# One room/position synthesis helper
# =========================

def synthesize_one_position(
    group,
    room,
    position,
    rir_folder,
    speech_files,
    fs,
    rng,
    hop_length,
    vad_frame_length,
    max_pause_seconds,
    correct_vad_for_rir,
    rir_delay_threshold_db,
    rt60_extension_fraction,
    order,
    apply_relative_sir=True,
    sir_min_db=0.0,
    sir_max_db=10.0,
    add_babble=True,
    num_babble_speakers=50,
    babble_snr_db=20.0,
):
    selected_rirs = []
    for speaker_idx in range(N_SPEAKERS):
        matches = group[group["s"] == speaker_idx]
        selected_rirs.append(None if len(matches) == 0 else matches.iloc[0])

    real_source_indices = [i for i, row in enumerate(selected_rirs) if row is not None]
    if not real_source_indices:
        print(f"Skipping R{room}_P{position}: no valid sources found")
        return None

    selected_speech_files = [None] * N_SPEAKERS
    chosen = rng.choice(speech_files, size=len(real_source_indices), replace=False)
    for idx, path in zip(real_source_indices, chosen):
        selected_speech_files[idx] = path

    convolved_tracks = []
    dry_track_lengths = []
    speaker_activity_masks = []
    speaker_directions = []
    speaker_metadata = []

    for speaker_idx in range(N_SPEAKERS):
        rir_row = selected_rirs[speaker_idx]
        speech_path = selected_speech_files[speaker_idx]

        if rir_row is None:
            convolved_tracks.append(None)
            speaker_activity_masks.append(None)
            dry_track_lengths.append(0)
            speaker_directions.append(np.zeros(3, dtype=np.float32))
            speaker_metadata.append({
                "speech_path": None,
                "speech_file": None,
                "phn_path": None,
                "phn_file": None,
                "rir_file": None,
                "pause_start_s": 0.0,
                "pause_middle_s": 0.0,
                "pause_end_s": 0.0,
                "rir_delay_samples": 0,
                "rir_delay_s": 0.0,
                "rt60_s": 0.0,
                "vad_extension_samples": 0,
                "vad_extension_s": 0.0,
                "missing_source": True,
                "rir_metadata": {
                    "Name": None, "RT": None, "rt60": None, "c50": None,
                    "x_room": None, "y_room": None, "z_room": None,
                    "x_source": None, "y_source": None, "z_source": None,
                    "x_mic": None, "y_mic": None, "z_mic": None,
                    "accdoa": [0.0, 0.0, 0.0],
                    "az": None, "el": None,
                    "mic_orientation_az_deg": None,
                    "status": "Missing",
                },
            })
            continue

        # The room pickle stores the RIR directly as a float array with
        # shape (channels, samples), so no WAV lookup or quantization is needed.
        if "_rir" not in rir_row.index:
            raise KeyError(
                f"PKL entry {rir_row.get('Name', '<unnamed>')} is missing '_rir'."
            )

        rir = np.asarray(rir_row["_rir"], dtype=np.float32)
        if rir.ndim != 2:
            raise ValueError(
                f"Expected PKL RIR with shape (channels, samples), got {rir.shape}."
            )

        rir_sample_rate = int(rir_row.get("_sample_rate", fs))
        if rir_sample_rate != int(fs):
            rir = librosa.resample(
                rir,
                orig_sr=rir_sample_rate,
                target_sr=int(fs),
                axis=1,
            ).astype(np.float32, copy=False)

        speech, speech_vad_mask, phn_path = load_timit_speech_and_activity(speech_path, fs=fs)
        track, sample_mask, pauses_s = make_repeated_speech_track(
            speech=speech,
            speech_vad_mask=speech_vad_mask,
            fs=fs,
            rng=rng,
            max_pause_seconds=max_pause_seconds,
        )
        rir = truncate_hoa_to_order(
            rir,
            target_order=order,
            signal_name=f"RIR {rir_row['Name']}",
        )

        rir_delay_samples = 0
        rir_delay_s = 0.0
        rt60_s = get_rt60_seconds_from_rir_row(rir_row)
        vad_extension_samples = 0
        vad_extension_s = 0.0
        if correct_vad_for_rir:
            rir_delay_samples = estimate_rir_delay_samples(rir, threshold_db=rir_delay_threshold_db)
            rir_delay_s = rir_delay_samples / float(fs)
            vad_extension_s = float(rt60_s) * float(rt60_extension_fraction)
            vad_extension_samples = int(round(vad_extension_s * fs))
            sample_mask = delay_and_extend_activity_mask(
                sample_mask, rir_delay_samples, vad_extension_samples, target_len=len(track)
            )

        metadata_accdoa = get_accdoa_from_rir_row(rir_row)
        metadata_az_deg, metadata_el_deg = get_az_el_deg_from_rir_row(rir_row)
        mic_orientation_az_deg = optional_float_from_rir_row(
            rir_row, "mic_orientation_az_deg"
        )

        dry_track_lengths.append(len(track))
        speaker_activity_masks.append(sample_mask)
        speaker_directions.append(metadata_accdoa)
        raw_rir_metadata = public_metadata_from_rir_row(rir_row)
        normalized_rir_metadata = {
            **raw_rir_metadata,
            # Canonical aliases used by the rest of this script.
            "Name": rir_row.get("Name", None),
            "RT": float(rt60_s),
            "rt60": float(rt60_s),
            "accdoa": metadata_accdoa.tolist(),
            "accdoa_x": float(metadata_accdoa[0]),
            "accdoa_y": float(metadata_accdoa[1]),
            "accdoa_z": float(metadata_accdoa[2]),
            "az": float(metadata_az_deg),
            "el": float(metadata_el_deg),
            "az_deg": float(metadata_az_deg),
            "el_deg": float(metadata_el_deg),
            "mic_orientation_az_deg": mic_orientation_az_deg,
            "status": rir_row.get("status", None),
        }

        speech_path_text = os.fspath(speech_path)
        if "DARPA_TIMIT/" in speech_path_text:
            portable_speech_path = (
                "DARPA_TIMIT/" + speech_path_text.split("DARPA_TIMIT/", 1)[-1]
            )
        else:
            portable_speech_path = speech_path_text

        speaker_metadata.append({
            "speech_path": portable_speech_path,
            "speech_file": os.path.basename(speech_path_text),
            "phn_path": phn_path,
            "phn_file": os.path.basename(phn_path),
            "rir_file": rir_row["Name"],
            **pauses_s,
            "rir_delay_samples": int(rir_delay_samples),
            "rir_delay_s": float(rir_delay_s),
            "rt60_s": float(rt60_s),
            "vad_extension_samples": int(vad_extension_samples),
            "vad_extension_s": float(vad_extension_s),
            "missing_source": False,
            "rir_metadata": normalized_rir_metadata,
        })
        conv = convolve_mono_with_multichannel_rir(track, rir, keep_original_length=True)
        convolved_tracks.append(conv)

    real_tracks = [x for x in convolved_tracks if x is not None]
    num_channels = check_same_hoa_channel_count(real_tracks)
    hoa_order = infer_hoa_order(num_channels)
    max_len = max(length for length in dry_track_lengths if length > 0)

    for i in range(N_SPEAKERS):
        if convolved_tracks[i] is None:
            convolved_tracks[i] = np.zeros((num_channels, max_len), dtype=np.float32)
            speaker_activity_masks[i] = np.zeros(max_len, dtype=bool)
        else:
            convolved_tracks[i] = pad_multichannel_to_length(convolved_tracks[i], max_len)
            speaker_activity_masks[i] = pad_bool_mask_to_length(speaker_activity_masks[i], max_len)

    speaker_directions = np.asarray(speaker_directions, dtype=np.float32)

    sir_normalization = []
    if apply_relative_sir:
        convolved_tracks, sir_normalization = normalize_tracks_to_relative_sir(
            tracks=convolved_tracks,
            activity_masks=speaker_activity_masks,
            rng=rng,
            sir_min_db=sir_min_db,
            sir_max_db=sir_max_db,
            # Use the first actually present source. This also supports
            # incomplete positions where S0 is absent but S1/S2 are available.
            reference_index=real_source_indices[0],
            power_channel=0,
        )
        for meta, info in zip(speaker_metadata, sir_normalization):
            meta.update({
                "sir_reference_speaker": int(info["reference_index"]),
                "target_sir_db": info["target_sir_db"],
                "achieved_sir_db": info["achieved_sir_db"],
                "sir_gain": float(info["gain"]),
                "active_power_before": float(info["active_power_before"]),
                "active_power_after": float(info["active_power_after"]),
            })

    # Sum the clean reverberant speaker tracks first. Before generating babble,
    # extend the clean voice mixture to an exact whole number of seconds. This
    # ensures that the final partial one-second segment contains continuing
    # babble over the samples that were zero-padded in the clean voice mixture.
    ambisonics = np.sum(convolved_tracks, axis=0).astype(np.float32)

    samples_per_second = int(fs)
    if samples_per_second <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")

    original_mix_len = int(ambisonics.shape[1])
    remainder = original_mix_len % samples_per_second
    voice_padding_samples = (samples_per_second - remainder) % samples_per_second
    padded_mix_len = original_mix_len + voice_padding_samples

    if voice_padding_samples > 0:
        ambisonics = pad_multichannel_to_length(ambisonics, padded_mix_len)
        convolved_tracks = [
            pad_multichannel_to_length(track, padded_mix_len)
            for track in convolved_tracks
        ]
        speaker_activity_masks = [
            pad_bool_mask_to_length(mask, padded_mix_len)
            for mask in speaker_activity_masks
        ]

    # Keep max_len consistent with the arrays returned by this function.
    max_len = padded_mix_len

    babble_noise = np.zeros_like(ambisonics)
    babble_info = {
        "enabled": False,
        "original_voice_mix_samples": original_mix_len,
        "voice_padding_samples": voice_padding_samples,
        "padded_voice_mix_samples": padded_mix_len,
    }
    if add_babble:
        active_any = np.any(np.stack(speaker_activity_masks, axis=0), axis=0)
        rt60_by_band_s = average_rt60_bands_from_rir_rows(selected_rirs)
        if not rt60_by_band_s:
            # Older metadata without rt60_*Hz columns remains usable.
            loaded_rt60 = [
                float(meta.get("rt60_s", 0.0))
                for meta in speaker_metadata
                if not meta.get("missing_source", False)
                and np.isfinite(float(meta.get("rt60_s", 0.0)))
                and float(meta.get("rt60_s", 0.0)) > 0.0
            ]
            fallback_rt60 = float(np.mean(loaded_rt60)) if loaded_rt60 else 0.5
            rt60_by_band_s = {1000: fallback_rt60}
        ambisonics, babble_noise, babble_info = add_diffuse_babble_noise(
            clean_ambisonics=ambisonics,
            speech_activity_mask=active_any,
            speech_files=speech_files,
            rt60_by_band_s=rt60_by_band_s,
            fs=fs,
            rng=rng,
            num_babble_speakers=num_babble_speakers,
            snr_db=babble_snr_db,
        )
        babble_info.update({
            "original_voice_mix_samples": original_mix_len,
            "voice_padding_samples": voice_padding_samples,
            "padded_voice_mix_samples": padded_mix_len,
        })

    ambisonics = normalize_peak(ambisonics, peak_value=0.9)

    return {
        "ambisonics": ambisonics,
        "speaker_activity_masks": speaker_activity_masks,
        "speaker_directions": speaker_directions,
        "speaker_metadata": speaker_metadata,
        "convolved_tracks": convolved_tracks,
        "num_channels": num_channels,
        "hoa_order": hoa_order,
        "max_len": max_len,
        "sir_normalization": sir_normalization,
        "babble_noise": babble_noise,
        "babble_info": babble_info,
    }


# =========================
# Main dataset generation
# =========================


def build_room_index(pkl_folder, nrooms=None, include_incomplete=False):
    """Index every pickle file, regardless of its filename."""

    pkl_folder = os.path.abspath(os.path.expanduser(pkl_folder))

    if not os.path.isdir(pkl_folder):
        raise NotADirectoryError(
            f"PKL folder does not exist: {pkl_folder}"
        )

    pkl_files = sorted(
        path
        for path in glob.glob(os.path.join(pkl_folder, "*.pkl"))
        if os.path.isfile(path)
    )

    # Optional compatibility with the existing flag.
    if not include_incomplete:
        pkl_files = [
            path for path in pkl_files
            if not os.path.basename(path).lower().endswith("_incomplete.pkl")
        ]

    if not pkl_files:
        raise FileNotFoundError(
            f"No .pkl files found in {pkl_folder}"
        )

    # Limit the file list before loading any pickle data.
    if nrooms is not None:
        pkl_files = pkl_files[:max(0, int(nrooms))]

    room_index = {}

    for mapped_room, pkl_path in enumerate(pkl_files):
        room_index[mapped_room] = {
            "mapped_room": mapped_room,

            # This is only a fallback. load_one_room_groups() will use
            # room_entry["room_idx"] when that field exists.
            "original_room": mapped_room,

            "pkl_path": pkl_path,
            "filename": os.path.basename(pkl_path),
        }

    return list(room_index), room_index


def _first_valid_sample_rate(*containers, default=16000):
    """Return the first positive integer sample rate found in the containers."""
    for container in containers:
        if not isinstance(container, Mapping):
            continue
        for key in ("sample_rate", "samplerate", "fs", "sr"):
            value = _finite_float(container.get(key, None))
            if value is not None and value > 0:
                return int(round(value))
    return int(default)


def normalize_rir_channel_layout(rir, metadata, entry_name):
    """Return an RIR with canonical shape ``(HOA channels, samples)``.

    The HOA order is inferred exclusively from the RIR channel dimension:
    valid channel counts are perfect squares ``1, 4, 9, 16, ...`` because an
    order-N HOA signal contains ``(N + 1) ** 2`` ACN channels.

    The supplied pickle normally stores ``(channels, samples)``.  This helper
    also safely accepts ``(samples, channels)`` and transposes it.
    """
    del metadata  # Kept in the signature for call-site compatibility.

    rir = np.asarray(rir, dtype=np.float32)
    if rir.ndim != 2:
        raise ValueError(
            f"{entry_name}: expected a two-dimensional RIR array, got {rir.shape}."
        )
    if 0 in rir.shape:
        raise ValueError(f"{entry_name}: RIR contains an empty dimension: {rir.shape}.")

    def is_valid_hoa_channel_count(value):
        value = int(value)
        root = int(round(np.sqrt(value)))
        return root >= 1 and root * root == value

    first_is_hoa = is_valid_hoa_channel_count(rir.shape[0])
    second_is_hoa = is_valid_hoa_channel_count(rir.shape[1])

    if first_is_hoa and not second_is_hoa:
        canonical = rir
    elif second_is_hoa and not first_is_hoa:
        canonical = rir.T
    elif first_is_hoa and second_is_hoa:
        # Both dimensions being perfect squares is technically ambiguous.
        # HOA channel counts are normally much smaller than RIR sample counts,
        # so prefer the smaller dimension.  If equal, retain channels-first.
        if rir.shape[0] <= rir.shape[1]:
            canonical = rir
        else:
            canonical = rir.T
    else:
        raise ValueError(
            f"{entry_name}: neither RIR dimension is a valid HOA channel count "
            f"(1, 4, 9, 16, 25, ...): shape={rir.shape}. "
        )

    # Final validation and explicit inference of the actual Ambisonics order.
    infer_hoa_order(canonical.shape[0])
    return np.ascontiguousarray(canonical, dtype=np.float32)


def _parse_position_and_source(position_id, pkl_path):
    position_text = str(position_id)
    parsed = re.fullmatch(r"(?P<p>\d+)_S(?P<s>\d+)", position_text)
    if parsed is None:
        parsed = re.search(
            r"(?:^|_)P(?P<p>\d+)_S(?P<s>\d+)(?:_|$)",
            position_text,
        )
    if parsed is None:
        raise ValueError(
            f"{pkl_path}: cannot parse receiver/source from "
            f"position_id={position_text!r}; expected '<P>_S<S>' or "
            "a name containing '_P<P>_S<S>'."
        )
    return position_text, int(parsed.group("p")), int(parsed.group("s"))


def load_one_room_groups(room_descriptor):
    """Load one nested room pickle and convert it to per-position DataFrames.

    Expected pickle structure::

        {
            "room_idx": 0,
            "sample_rate": 16000,          # optional
            "rirs": {
                "7_S2": {
                    "metadata": {
                        "accdoa_x": ...,
                        "accdoa_y": ...,
                        "accdoa_z": ...,
                        "az_deg": ...,
                        "el_deg": ...,
                        ...
                    },
                    "rir": np.ndarray,
                }
            }
        }

    RIR entries are flattened only inside the temporary DataFrame; the original
    arrays remain under the internal ``_rir`` column.
    """
    mapped_room = int(room_descriptor["mapped_room"])
    fallback_room = int(room_descriptor["original_room"])
    pkl_path = room_descriptor["pkl_path"]

    with open(pkl_path, "rb") as handle:
        room_entry = pickle.load(handle)

    if not isinstance(room_entry, dict):
        raise TypeError(f"{pkl_path} must contain a dictionary.")
    if "rirs" not in room_entry or not isinstance(room_entry["rirs"], dict):
        raise KeyError(f"{pkl_path} is missing the 'rirs' dictionary.")

    original_room = int(room_entry.get("room_idx", fallback_room))
    room_sample_rate = _first_valid_sample_rate(room_entry, default=16000)
    rirs = room_entry["rirs"]

    parsed_entries = []
    raw_source_ids = set()
    for position_id, rir_entry in rirs.items():
        if not isinstance(rir_entry, dict):
            raise TypeError(
                f"{pkl_path}: RIR entry {position_id!r} must be a dictionary."
            )
        if "rir" not in rir_entry:
            raise KeyError(f"{pkl_path}: RIR entry {position_id!r} has no 'rir'.")

        position_text, position, raw_source = _parse_position_and_source(
            position_id, pkl_path
        )
        raw_source_ids.add(raw_source)
        parsed_entries.append(
            (position_text, position, raw_source, rir_entry)
        )

    # Native files normally use S0..S2. Also accept S1..S3: the presence of S3
    # unambiguously indicates one-based indexing. An incomplete {S1,S2} set stays
    # zero-based because it can validly mean a missing S0.
    source_index_offset = (
        1 if 0 not in raw_source_ids and N_SPEAKERS in raw_source_ids else 0
    )

    position_rows = {}
    occupied_slots = set()
    for position_text, position, raw_source, rir_entry in parsed_entries:
        speaker = raw_source - source_index_offset
        if not 0 <= speaker < N_SPEAKERS:
            convention = (
                f"S1..S{N_SPEAKERS}" if source_index_offset else f"S0..S{N_SPEAKERS - 1}"
            )
            raise ValueError(
                f"{pkl_path}: source ID S{raw_source} maps to slot {speaker}, "
                f"outside 0..{N_SPEAKERS - 1}. Detected convention: {convention}."
            )

        slot_key = (position, speaker)
        if slot_key in occupied_slots:
            raise ValueError(
                f"{pkl_path}: duplicate RIR for position {position}, source slot {speaker}."
            )
        occupied_slots.add(slot_key)

        metadata_obj = rir_entry.get("metadata") or {}
        if not isinstance(metadata_obj, Mapping):
            raise TypeError(
                f"{pkl_path}: metadata for {position_text!r} must be a dictionary."
            )
        metadata = dict(metadata_obj)

        name = metadata.get(
            "Name", f"R{original_room:04d}_P{position}_S{raw_source}_IR"
        )
        status = metadata.get("status", "Success")
        if str(status).strip().lower() != "success":
            continue

        rir = normalize_rir_channel_layout(
            rir_entry["rir"], metadata=metadata, entry_name=f"{pkl_path}:{position_text}"
        )
        entry_sample_rate = _first_valid_sample_rate(
            rir_entry, metadata, room_entry, default=room_sample_rate
        )

        row = {
            **metadata,
            "Name": name,
            "status": status,
            "r": mapped_room,
            "original_room_idx": original_room,
            "p": position,
            "s": speaker,
            "source_id_raw": raw_source,
            "source_index_offset": source_index_offset,
            "_rir": rir,
            "_sample_rate": entry_sample_rate,
            "_pkl_path": pkl_path,
            "_position_id": position_text,
        }
        position_rows.setdefault(position, []).append(row)

    room_groups = []
    for position, rows in sorted(position_rows.items()):
        group = pd.DataFrame(rows).sort_values("s").reset_index(drop=True)
        group["rt60_normalized"] = group.apply(
            get_rt60_seconds_from_rir_row, axis=1
        ).astype(float)
        room_groups.append((position, group))

    if not room_groups:
        print(f"Warning: {os.path.basename(pkl_path)} contains no valid RIR groups.")
    return room_groups

def make_position_rng(seed, room, position):
    # Stable regardless of task count, task assignment, or loop ordering.
    seed_sequence = np.random.SeedSequence(
        [int(seed), int(room), int(position)]
    )
    return np.random.default_rng(seed_sequence)


def parse_split_ratios(split_text):
    """Parse TRAIN/VAL/TEST ratios and normalize them to sum to one."""
    if isinstance(split_text, str):
        parts = [part.strip() for part in split_text.split("/")]
    else:
        parts = list(split_text)
    if len(parts) != 3:
        raise ValueError("split must contain exactly three values: train/val/test")
    ratios = np.asarray([float(x) for x in parts], dtype=np.float64)
    if np.any(ratios < 0.0) or not np.isfinite(ratios).all():
        raise ValueError(f"Invalid split ratios: {parts}")
    total = float(ratios.sum())
    if total <= 0.0:
        raise ValueError("At least one split ratio must be positive.")
    return ratios / total


def split_rooms_deterministically(rooms, split_ratios, seed):
    """Assign sorted, contiguous rooms to train/val/test.

    The seed argument is retained for API compatibility, but room membership is
    no longer shuffled. For example, with rooms 0..9 and a 70/20/10 split:
    train=0..6, val=7..8, and test=9.
    """
    del seed  # Splitting is intentionally independent of the random seed.
    rooms = np.sort(np.asarray(list(rooms), dtype=np.int64))

    raw = split_ratios * len(rooms)
    counts = np.floor(raw).astype(int)
    remainder = len(rooms) - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1

    n_train, n_val, _ = counts.tolist()

    return {
        "train": rooms[:n_train].tolist(),
        "val": rooms[n_train:n_train + n_val].tolist(),
        "test": rooms[n_train + n_val:].tolist(),
    }


class RunningFeatureStats:
    """Numerically stable scalar statistics for HO-PIV and log-mel values."""
    def __init__(self, num_hoa_channels):
        self.hopiv_channels = 2 * (int(num_hoa_channels) - 1)
        self.stats = {
            "hopiv": {"count": 0, "mean": 0.0, "m2": 0.0},
            "logmel": {"count": 0, "mean": 0.0, "m2": 0.0},
        }

    @staticmethod
    def _update_one(state, values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        batch_count = int(values.size)
        batch_mean = float(values.mean())
        batch_m2 = float(np.sum((values - batch_mean) ** 2))
        old_count = int(state["count"])
        if old_count == 0:
            state.update(count=batch_count, mean=batch_mean, m2=batch_m2)
            return
        total = old_count + batch_count
        delta = batch_mean - float(state["mean"])
        state["mean"] += delta * batch_count / total
        state["m2"] += batch_m2 + delta * delta * old_count * batch_count / total
        state["count"] = total

    def update(self, X):
        X = np.asarray(X, dtype=np.float32)
        self._update_one(self.stats["hopiv"], X[..., :self.hopiv_channels])
        self._update_one(self.stats["logmel"], X[..., self.hopiv_channels:])

    def as_dict(self):
        result = {}
        for name, state in self.stats.items():
            count = int(state["count"])
            variance = float(state["m2"] / count) if count else 0.0
            result[name] = {
                "count": count,
                "mean": float(state["mean"]),
                "stdev": float(np.sqrt(max(variance, 0.0))),
            }
        return result


def generate_ambisonics_wavs(
    pkl_folder,
    speech_folders,
    output_folder,
    mode="train",
    split=(0.7, 0.2, 0.1),
    calculate_distribution=False,
    fs=16000,
    seed=0,
    hop_length=320,
    vad_frame_length=640,
    max_pause_seconds=0.5,
    n_fft=1024,
    n_mels=128,
    order=3,
    nrooms=None,
    write_wavs="all",
    rooms_per_pt=100,
    correct_vad_for_rir=True,
    rir_delay_threshold_db=-40.0,
    rt60_extension_fraction=1.0 / 6.0,
    plot_first_sample=False,
    first_sample_plot_max_seconds=None,
    apply_relative_sir=True,
    sir_min_db=0.0,
    sir_max_db=10.0,
    add_babble=True,
    num_babble_speakers=50,
    babble_snr_db=20.0,
):
    order = int(order)
    mode = str(mode).lower()
    if mode not in {"train", "test"}:
        raise ValueError(f"mode must be 'train' or 'test', got {mode!r}.")

    write_wavs = str(write_wavs).strip().lower()
    write_wav_aliases = {
        "none": "none",
        "no": "none",
        "no-wavs": "none",
        "false": "none",
        "all": "all",
        "all-wavs": "all",
        "true": "all",
        "testset": "testset",
        "test-set": "testset",
        "test-only": "testset",
        "only-testset": "testset",
    }
    try:
        write_wavs = write_wav_aliases[write_wavs]
    except KeyError as exc:
        raise ValueError(
            "write_wavs must be one of 'none', 'all', or 'testset', "
            f"got {write_wavs!r}."
        ) from exc

    if order < 0:
        raise ValueError(f"order must be non-negative, got {order}.")

    target_num_channels = (order + 1) ** 2
    all_rooms, room_index = build_room_index(pkl_folder, nrooms=nrooms)
    split_ratios = parse_split_ratios(split)
    if mode == "train":
        split_rooms = split_rooms_deterministically(all_rooms, split_ratios, seed)
    else:
        split_rooms = {"test": list(all_rooms)}

    speech_files_by_split = {}
    for split_name in split_rooms:
        speech_root = speech_folders["train" if split_name == "train" else "test"]
        files = sorted(
            path for path in glob.glob(os.path.join(speech_root, "**", "*.WAV.wav"), recursive=True)
            if os.path.isfile(path) and os.path.isfile(get_timit_phn_path(path))
        )
        if len(files) < N_SPEAKERS:
            raise ValueError(
                f"Need at least {N_SPEAKERS} TIMIT pairs for {split_name}, found {len(files)} in {speech_root}."
            )
        speech_files_by_split[split_name] = files

    _ = make_mel_basis(fs=fs, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=fs / 2.0)
    os.makedirs(output_folder, exist_ok=True)
    overlap_frame_counts = np.zeros(N_SPEAKERS + 1, dtype=np.int64)
    first_sample_plotted = False
    stats = RunningFeatureStats(target_num_channels) if calculate_distribution else None

    for split_name, split_room_list in split_rooms.items():
        split_output = os.path.join(output_folder, split_name)
        wav_folder = os.path.join(split_output, "wav")
        room_csv_folder = os.path.join(split_output, "csv")
        pt_batch_folder = os.path.join(split_output, "pt")
        os.makedirs(room_csv_folder, exist_ok=True)
        os.makedirs(pt_batch_folder, exist_ok=True)

        write_wavs_for_split = (
            write_wavs == "all"
            or (write_wavs == "testset" and split_name == "test")
        )
        if write_wavs_for_split:
            os.makedirs(wav_folder, exist_ok=True)

        all_batches = [
            split_room_list[i:i + rooms_per_pt]
            for i in range(0, len(split_room_list), rooms_per_pt)
        ]
        print(
            f"Mode={mode} split={split_name} | "
            f"rooms={len(split_room_list)} | batches={len(all_batches)}"
        )

        for batch_rooms in tqdm(all_batches, desc=f"Generate {split_name}"):
            batch_room_records = []
            batch_csv_rows = []
            batch_name = f"room_{int(batch_rooms[0]):04d}-{int(batch_rooms[-1]):04d}.pt"

            for room in tqdm(batch_rooms, desc=f"{split_name} rooms", unit="room", leave=False):
                room_sample_records = []
                room_csv_rows = []
                # Only this room's pickle is resident in memory.
                room_groups = load_one_room_groups(room_index[room])
                for position_idx, group in room_groups:
                    rng = make_position_rng(seed, room, position_idx)
                    synthesized = synthesize_one_position(
                        group=group, room=room, position=position_idx,
                        rir_folder=pkl_folder,
                        speech_files=speech_files_by_split[split_name], fs=fs, rng=rng,
                        hop_length=hop_length, vad_frame_length=vad_frame_length,
                        max_pause_seconds=max_pause_seconds,
                        correct_vad_for_rir=correct_vad_for_rir,
                        rir_delay_threshold_db=rir_delay_threshold_db,
                        rt60_extension_fraction=rt60_extension_fraction, order=order,
                        apply_relative_sir=apply_relative_sir, sir_min_db=sir_min_db,
                        sir_max_db=sir_max_db, add_babble=add_babble,
                        num_babble_speakers=num_babble_speakers,
                        babble_snr_db=babble_snr_db,
                    )
                    if synthesized is None:
                        continue

                    if plot_first_sample and not first_sample_plotted:
                        plot_first_realigned_multitrack_sample(
                            convolved_tracks=synthesized["convolved_tracks"],
                            speaker_activity_masks=synthesized["speaker_activity_masks"],
                            speaker_metadata=synthesized["speaker_metadata"], fs=fs,
                            output_path=os.path.join(split_output, "speaker_tracks.pdf"),
                            max_seconds=first_sample_plot_max_seconds,
                        )
                        first_sample_plotted = True

                    rir_name = next(meta["rir_file"] for meta in synthesized["speaker_metadata"] if meta["rir_file"] is not None)
                    rir_stem = re.sub(r"_S\d+(?:_IR)?$", "", os.path.splitext(os.path.basename(rir_name))[0])

                    # Ensure every generated WAV name contains a room identifier.
                    # Keep an existing Rxxxx token; otherwise prepend the current room.
                    if re.search(r"R\d{4}", rir_stem, flags=re.IGNORECASE) is None:
                        rir_stem = f"R{int(room):04d}_{rir_stem}"

                    output_wav_name = rir_stem + ".wav"
                    if write_wavs_for_split:
                        wavfile.write(
                            os.path.join(wav_folder, output_wav_name), fs,
                            synthesized["ambisonics"].T.astype(np.float32),
                        )

                    sample_records, csv_rows = save_hopiv_samples_for_mixture(
                        ambisonics=synthesized["ambisonics"],
                        speaker_activity_masks=synthesized["speaker_activity_masks"],
                        speaker_directions=synthesized["speaker_directions"],
                        speaker_metadata=synthesized["speaker_metadata"], room=room,
                        position=position_idx, fs=fs,
                        output_wav_name=output_wav_name, pt_batch_name=batch_name,
                        hop_length=hop_length, frame_length=vad_frame_length,
                        n_fft=n_fft, n_mels=n_mels,
                        overlap_frame_counts=overlap_frame_counts,
                    )
                    room_sample_records.extend(sample_records)
                    room_csv_rows.extend(csv_rows)

                    # In train mode, normalization statistics must only use train data.
                    if stats is not None and (mode == "test" or split_name == "train"):
                        for X_tensor, _ in sample_records:
                            stats.update(X_tensor.numpy())

                if room_sample_records:
                    batch_room_records.append({"room": int(room), "samples": room_sample_records})
                    batch_csv_rows.extend(room_csv_rows)

                # Drop DataFrames/RIR arrays from the just-processed pickle now.
                del room_groups

            samples = [sample for record in batch_room_records for sample in record["samples"]]
            if samples:
                X = torch.stack([sample[0] for sample in samples], dim=0).contiguous()
                accdoa = torch.stack([sample[1] for sample in samples], dim=0).contiguous()
                pt_path = os.path.join(pt_batch_folder, batch_name)
                csv_path = os.path.join(room_csv_folder, batch_name.replace(".pt", ".csv"))
                torch.save((X, accdoa), pt_path)
                pd.DataFrame(batch_csv_rows).to_csv(csv_path, index=False)
                print(f"Saved {split_name}/{batch_name}: X={tuple(X.shape)}, accdoa={tuple(accdoa.shape)}")

    if stats is not None:
        stats_payload = {
            "mode": mode,
            "scope": "train" if mode == "train" else "test",
            "hoa_order": order,
            "n_mels": n_mels,
            "statistics": stats.as_dict(),
        }
        stats_path = os.path.join(output_folder, "distribution.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_payload, f, indent=2)
        print(f"Saved feature distribution to: {stats_path}")

    total_overlap_frames = int(np.sum(overlap_frame_counts))
    print("\nSource-overlap distribution:")
    for overlap_count, frame_count in enumerate(overlap_frame_counts):
        percentage = 100.0 * int(frame_count) / total_overlap_frames if total_overlap_frames else 0.0
        print(f"  {overlap_count} active source(s): {int(frame_count)} frames ({percentage:.2f}%)")

def parse_args():
    parser = argparse.ArgumentParser(description="Single-machine HOA/HO-PIV dataset generation.")
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument(
        "--split", type=str, default="70/20/10",
        help="Room split as train/val/test weights (default: 70/20/10).",
    )
    parser.add_argument(
        "--calculate-distribution", action=argparse.BooleanOptionalAction, default=False,
        help="Calculate scalar mean/stdev independently for HO-PIV and log-mel. In train mode only train is used.",
    )
    parser.add_argument("--order", type=int, default=3, help="Target HOA order.")
    parser.add_argument("--base-dir", type=str, required=True, help="Directory containing room .pkl files.")
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--rooms-per-pt", type=int, default=1)
    parser.add_argument("--nrooms", type=int, default=None)
    parser.add_argument(
        "--write-wavs",
        choices=("none", "all", "testset"),
        default="all",
        help=(
            "WAV export policy: 'none' writes no WAVs, 'all' writes WAVs for "
            "every generated split, and 'testset' writes WAVs only for the test "
            "split. With --mode test, 'testset' is equivalent to 'all'."
        ),
    )
    parser.add_argument("--plot-first-sample", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    base_dir = os.path.abspath(os.path.expanduser(args.base_dir))
    output_folder = os.path.join(base_dir, args.output_folder)
    timit_root = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/DARPA_TIMIT/data"
    speech_folders = {
        "train": os.path.join(timit_root, "TRAIN"),
        "test": os.path.join(timit_root, "TEST"),
    }
    generate_ambisonics_wavs(
        pkl_folder=base_dir,
        speech_folders=speech_folders,
        output_folder=output_folder,
        mode=args.mode,
        split=args.split,
        calculate_distribution=args.calculate_distribution,
        fs=16000, seed=0, hop_length=320, vad_frame_length=640,
        max_pause_seconds=1.5,
        n_fft=1024, n_mels=128, order=args.order, nrooms=args.nrooms,
        write_wavs=args.write_wavs, rooms_per_pt=args.rooms_per_pt,
        correct_vad_for_rir=True, rir_delay_threshold_db=-40.0,
        rt60_extension_fraction=1.0 / 6.0,
        plot_first_sample=args.plot_first_sample,
    )