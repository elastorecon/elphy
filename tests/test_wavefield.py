#!/usr/bin/env python
"""Tests for :mod:`elphy.wavefield`.

These tests use synthetic phase data with a known ground truth so they
run without any external data dependency.  The synthetic model mirrors a
balanced bipolar MRE acquisition::

    phase_pos = +phi_motion + phi_background
    phase_neg = -phi_motion + phi_background

so that combining the two polarities must recover ``2 * phi_motion`` and
cancel the background exactly.
"""

import numpy as np
import pytest

from elphy import wavefield as wf


def _synthetic_acquisition(nx=16, ny=16, nz=4, n_offset=4, n_dir=3, n_freq=2, motion_amp=0.4):
    """Build a synthetic raw-phase array with known motion/background.

    Returns
    -------
    phase : ndarray
        Raw phase, layout ``[x, y, z, 2*n_offset, n_dir, n_freq]`` with
        interleaved even=positive / odd=negative polarities.
    motion : ndarray
        Ground-truth single-encoding motion phase,
        ``[x, y, z, n_offset, n_dir, n_freq]``.
    background : ndarray
        Ground-truth background phase (per voxel, broadcast over the
        encoding axis).
    """
    # Smooth spatial background (large, slowly varying).
    xx, yy, zz = np.meshgrid(
        np.linspace(0, 1, nx),
        np.linspace(0, 1, ny),
        np.linspace(0, 1, nz),
        indexing="ij",
    )
    background = 0.9 * (xx + 0.5 * yy - 0.3 * zz)  # radians, smooth
    # Motion phase: small travelling wave, distinct per offset/dir/freq.
    motion = np.zeros((nx, ny, nz, n_offset, n_dir, n_freq))
    for o in range(n_offset):
        for d in range(n_dir):
            for f in range(n_freq):
                k = 2 * np.pi * (1 + f)
                phase0 = 2 * np.pi * o / n_offset
                motion[..., o, d, f] = motion_amp * np.sin(k * xx + 0.5 * d + phase0)
    # Assemble interleaved polarities: even=pos, odd=neg.
    phase = np.zeros((nx, ny, nz, 2 * n_offset, n_dir, n_freq))
    bg = background[..., None, None]  # broadcast over dir, freq
    for o in range(n_offset):
        phase[..., 2 * o, :, :] = motion[..., o, :, :] + bg
        phase[..., 2 * o + 1, :, :] = -motion[..., o, :, :] + bg
    return phase, motion, background


def test_wrap_to_pi_range():
    x = np.array([-3 * np.pi, -np.pi, 0.0, np.pi, 3 * np.pi, 10.0])
    w = wf.wrap_to_pi(x)
    assert np.all(w > -np.pi - 1e-9)
    assert np.all(w <= np.pi + 1e-9)
    # wrapping is idempotent
    np.testing.assert_allclose(wf.wrap_to_pi(w), w, atol=1e-12)


def test_combine_cancels_background_and_doubles_motion():
    phase, motion, background = _synthetic_acquisition(motion_amp=0.3)
    combined = wf.combine_encodings(phase, encoding_axis=3, halve=False)
    # Expected: 2 * motion (wrapped). Motion here is small enough not to wrap.
    expected = wf.wrap_to_pi(2 * motion)
    assert combined.shape == motion.shape
    np.testing.assert_allclose(combined, expected, atol=1e-10)


def test_combine_halve_recovers_single_encoding_motion():
    phase, motion, _ = _synthetic_acquisition(motion_amp=0.2)
    combined = wf.combine_encodings(phase, encoding_axis=3, halve=True)
    np.testing.assert_allclose(combined, motion, atol=1e-10)


def test_combine_polarity_convention():
    phase, motion, _ = _synthetic_acquisition(motion_amp=0.2)
    even_pos = wf.combine_encodings(phase, halve=False, polarity="even_positive")
    odd_pos = wf.combine_encodings(phase, halve=False, polarity="odd_positive")
    # Swapping which polarity is "positive" flips the sign.
    np.testing.assert_allclose(even_pos, -odd_pos, atol=1e-10)


def test_combine_odd_axis_raises():
    bad = np.zeros((4, 4, 2, 3))  # encoding axis length 3 (odd)
    with pytest.raises(ValueError):
        wf.combine_encodings(bad, encoding_axis=3)


def test_phase_to_displacement_scalar_and_array():
    phase = np.linspace(-2, 2, 24).reshape(2, 3, 4)
    # scalar sensitivity
    disp = wf.phase_to_displacement(phase, 2.0)
    np.testing.assert_allclose(disp, phase / 2.0)
    # array sensitivity broadcasting over last axis
    k = np.array([1.0, 2.0, 4.0, 8.0])
    disp2 = wf.phase_to_displacement(phase, k)
    np.testing.assert_allclose(disp2, phase / k)


def test_meg_sensitivity_muthupillai_formula():
    # k = 2 * gamma * N * T * G / pi
    gamma, N, T, G = 2.675e8, 4.0, 0.02, 0.04
    k = wf.meg_sensitivity_muthupillai(N, T, G, gyromagnetic_ratio=gamma)
    expected = 2 * gamma * N * T * G / np.pi
    assert k == pytest.approx(expected)
    # displacement round-trip: phase = k * xi  ->  xi = phase / k
    xi = 1e-6  # 1 micron
    phase = k * xi
    np.testing.assert_allclose(wf.phase_to_displacement(phase, k), xi, rtol=1e-12)


def test_unwrap_recovers_ramp():
    # A phase ramp exceeding +-pi, wrapped, should unwrap back (up to a
    # global constant).
    nx = 32
    ramp = np.linspace(0, 6 * np.pi, nx)[:, None, None] * np.ones((nx, 4, 2))
    wrapped = wf.wrap_to_pi(ramp)
    unwrapped = wf.unwrap_field(wrapped[..., None, None])[..., 0, 0]
    # Remove global offset then compare.
    u = unwrapped - unwrapped[0]
    r = ramp - ramp[0]
    np.testing.assert_allclose(u, r, atol=1e-6)


def test_unwrap_mask_zeroes_outside():
    phase = wf.wrap_to_pi(np.linspace(0, 4 * np.pi, 8 * 8 * 2).reshape(8, 8, 2))
    mask = np.zeros((8, 8, 2), dtype=bool)
    mask[2:6, 2:6, :] = True
    out = wf.unwrap_field(phase[..., None], mask=mask)[..., 0]
    assert np.all(out[~mask] == 0.0)


def test_raw_phase_to_wavefield_pipeline():
    phase, motion, _ = _synthetic_acquisition(motion_amp=0.15)
    # No unwrap needed (small motion); no sensitivity -> stays in radians.
    field = wf.raw_phase_to_wavefield(phase, unwrap=False, halve=False)
    np.testing.assert_allclose(field, wf.wrap_to_pi(2 * motion), atol=1e-10)
    # With a scalar sensitivity, output scales by 1/k.
    field_disp = wf.raw_phase_to_wavefield(phase, unwrap=False, halve=False, meg_sensitivity=2.0)
    np.testing.assert_allclose(field_disp, field / 2.0, atol=1e-10)


def _synthetic_temporal_series(n_offset=8, amp=1.7, temporal_phase=0.6, nx=4, ny=4, nz=2):
    """Real motion field sampled across one vibration cycle.

    ``xi(t_k) = amp * cos(2*pi*k/n_offset + temporal_phase)`` for
    ``k = 0..n_offset-1`` along axis 3, so the first temporal harmonic
    must recover ``amp * exp(1j*temporal_phase)`` exactly.
    """
    k = np.arange(n_offset)
    series = amp * np.cos(2 * np.pi * k / n_offset + temporal_phase)
    # Broadcast the same series into every voxel (axes x,y,z) and one
    # direction/frequency, offset along axis 3.
    field = np.zeros((nx, ny, nz, n_offset, 1, 1))
    field[..., 0, 0] = series[None, None, None, :]
    return field, amp, temporal_phase


def test_temporal_harmonic_recovers_amplitude_and_phase():
    field, amp, tphase = _synthetic_temporal_series()
    wave = wf.temporal_harmonic(field, offset_axis=3, harmonic=1)
    # Offset axis removed.
    assert wave.shape == (4, 4, 2, 1, 1)
    expected = amp * np.exp(1j * tphase)
    np.testing.assert_allclose(wave[..., 0, 0], expected, atol=1e-10)


def test_temporal_harmonic_flip_conjugates():
    field, amp, tphase = _synthetic_temporal_series()
    n_offset = field.shape[3]
    wave = wf.temporal_harmonic(field, offset_axis=3, harmonic=1, flip=False)
    wave_flip = wf.temporal_harmonic(field, offset_axis=3, harmonic=1, flip=True)
    # Reversing a length-N real sequence maps X[1] -> exp(2j*pi/N)*conj(X[1]):
    # the magnitude (displacement amplitude) is preserved and the temporal
    # phase / propagation direction is reversed.
    np.testing.assert_allclose(np.abs(wave), np.abs(wave_flip), atol=1e-10)
    np.testing.assert_allclose(
        wave_flip, np.exp(2j * np.pi / n_offset) * np.conj(wave), atol=1e-10
    )


def test_temporal_harmonic_out_of_range_raises():
    field, _, _ = _synthetic_temporal_series(n_offset=4)
    with pytest.raises(ValueError):
        wf.temporal_harmonic(field, offset_axis=3, harmonic=4)


def test_pipeline_harmonic_extraction():
    # Build a raw acquisition whose combined motion varies sinusoidally
    # across temporal offsets, then run the full pipeline with harmonic=1.
    n_offset = 4
    amp, tphase = 0.2, 0.4
    k = np.arange(n_offset)
    motion = amp * np.cos(2 * np.pi * k / n_offset + tphase)
    nx = ny = 4
    nz = 2
    # Interleave +/- polarities: 2*n_offset encodings on axis 3.
    phase = np.zeros((nx, ny, nz, 2 * n_offset, 1, 1))
    background = 0.05
    for j in range(n_offset):
        phase[..., 2 * j, 0, 0] = motion[j] + background
        phase[..., 2 * j + 1, 0, 0] = -motion[j] + background
    wave = wf.raw_phase_to_wavefield(phase, unwrap=False, halve=True, harmonic=1)
    assert wave.shape == (nx, ny, nz, 1, 1)
    np.testing.assert_allclose(wave[..., 0, 0], amp * np.exp(1j * tphase), atol=1e-10)
