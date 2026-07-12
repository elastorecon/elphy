#!/usr/bin/env python
"""Tests for :mod:`elphy.operators`.

Analytic ground truth is used throughout so the tests need no external
data.  A transverse (shear) plane wave has zero divergence and a curl
set by the transverse gradient; a longitudinal (compressional) plane
wave has zero curl and a divergence set by the axial gradient.  The
band-pass filter is checked against pure sinusoids at known wavenumbers.
"""

import numpy as np
import pytest

from elphy import operators as op


def _grid(n=32, spacing=1.0):
    x = np.arange(n) * spacing
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return X, Y, Z


def _vector(ux, uy, uz):
    """Stack three (x,y,z) component arrays into an [x,y,z,3] field."""
    return np.stack([ux, uy, uz], axis=3)


def test_shear_wave_divergence_zero_curl_nonzero():
    # u = z_hat * sin(k x): transverse wave travelling in x.
    n, spacing = 40, 1.0
    k = 2 * np.pi / 20.0  # wavelength 20 voxels
    X, _, _ = _grid(n, spacing)
    uz = np.sin(k * X)
    field = _vector(np.zeros_like(uz), np.zeros_like(uz), uz)

    div = op.divergence(field, voxel_size=spacing)
    # Divergence of a transverse wave is identically zero.
    assert np.abs(div).max() < 1e-6

    crl = op.curl(field, voxel_size=spacing)
    # curl = (du_z/dy - du_y/dz, du_x/dz - du_z/dx, ...) = (0, -k cos(kx), 0)
    expected_y = -k * np.cos(k * X)
    inner = slice(2, -2)
    np.testing.assert_allclose(
        crl[inner, inner, inner, 1], expected_y[inner, inner, inner], atol=1.5e-2
    )
    assert np.abs(crl[inner, inner, inner, 0]).max() < 1e-6
    assert np.abs(crl[inner, inner, inner, 2]).max() < 1e-6


def test_compressional_wave_curl_zero_divergence_nonzero():
    # u = x_hat * sin(k x): longitudinal wave travelling in x.
    n, spacing = 40, 1.0
    k = 2 * np.pi / 20.0
    X, _, _ = _grid(n, spacing)
    ux = np.sin(k * X)
    field = _vector(ux, np.zeros_like(ux), np.zeros_like(ux))

    crl = op.curl(field, voxel_size=spacing)
    assert np.abs(crl).max() < 1e-6

    div = op.divergence(field, voxel_size=spacing)
    expected = k * np.cos(k * X)
    inner = slice(2, -2)
    np.testing.assert_allclose(
        div[inner, inner, inner], expected[inner, inner, inner], atol=1.5e-2
    )


def test_curl_handles_complex_and_trailing_freq_axis():
    # Complex field with a trailing frequency axis [x,y,z,3,nf].
    n, spacing = 24, 1.0
    k = 2 * np.pi / 8.0
    X, _, _ = _grid(n, spacing)
    uz = np.exp(1j * k * X)
    base = _vector(np.zeros_like(uz), np.zeros_like(uz), uz)  # [x,y,z,3]
    field = np.stack([base, 2 * base], axis=4)  # two "frequencies"
    crl = op.curl(field, voxel_size=spacing, direction_axis=3)
    assert crl.shape == field.shape
    assert np.iscomplexobj(crl)
    # Second frequency is 2x the first (linearity).
    np.testing.assert_allclose(crl[..., 1], 2 * crl[..., 0], atol=1e-9)


def test_curl_requires_length3_direction():
    field = np.zeros((8, 8, 8, 2))
    with pytest.raises(ValueError):
        op.curl(field, voxel_size=1.0)


def test_bandpass_passes_target_removes_out_of_band():
    # Two sinusoids: a low-k bulk trend and a high-k noise, plus a
    # mid-band "wave".  Band-pass should keep the middle one.
    n, spacing = 64, 1.0
    X, _, _ = _grid(n, spacing)
    k_low = 2 * np.pi * (1.0 / n)  # 1 cycle across FOV (bulk)
    k_mid = 2 * np.pi * (6.0 / n)  # 6 cycles (signal)
    k_high = 2 * np.pi * (24.0 / n)  # 24 cycles (noise)
    signal = np.sin(k_mid * X)
    field = np.sin(k_low * X) + signal + 0.5 * np.sin(k_high * X)

    # Cutoffs in cycles/voxel: pass ~6/64, block 1/64 and 24/64.
    filt = op.bandpass_filter(field, voxel_size=spacing, low_cutoff=3.0 / n, high_cutoff=12.0 / n)
    # Filtered field should track the mid-band signal much better than
    # the raw field does.
    inner = slice(4, -4)
    err_filt = np.std((filt - signal)[inner, inner, inner])
    err_raw = np.std((field - signal)[inner, inner, inner])
    assert err_filt < 0.25 * err_raw
    # Real input -> real output.
    assert not np.iscomplexobj(filt)


def test_bandpass_requires_a_cutoff_and_valid_order():
    field = np.zeros((8, 8, 8))
    with pytest.raises(ValueError):
        op.bandpass_filter(field, voxel_size=1.0)
    with pytest.raises(ValueError):
        op.bandpass_filter(field, voxel_size=1.0, low_cutoff=0.4, high_cutoff=0.1)


def test_bandpass_highpass_removes_dc():
    # A constant field is pure DC; a high-pass must annihilate it.
    field = np.full((16, 16, 16), 3.7)
    filt = op.bandpass_filter(field, voxel_size=1.0, low_cutoff=0.05)
    assert np.abs(filt).max() < 1e-9
