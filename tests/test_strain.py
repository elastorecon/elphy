"""Analytic-ground-truth tests for elphy.strain (octahedral-strain metrics)."""

import numpy as np
import pytest

import elphy


def _grid(n=12):
    ax = np.arange(n, dtype=float)
    return np.meshgrid(ax, ax, ax, indexing="ij")


def test_strain_tensor_constant_gradient_exact():
    # u = (c*x, 0, 0)  ->  e_xx = c, all else 0.
    X, Y, Z = _grid()
    c = 0.37
    field = np.zeros(X.shape + (3,))
    field[..., 0] = c * X
    e = elphy.strain_tensor(field, voxel_size=1.0)
    sl = (slice(1, -1),) * 3
    assert np.allclose(e["xx"][sl], c, atol=1e-9)
    for k in ("yy", "zz", "xy", "xz", "yz"):
        assert np.allclose(e[k][sl], 0.0, atol=1e-9)


def test_strain_tensor_voxel_size_scaling():
    # Same displacement, doubled voxel size -> half the gradient.
    X, Y, Z = _grid()
    field = np.zeros(X.shape + (3,))
    field[..., 2] = 0.5 * X  # u_z = 0.5 x
    e1 = elphy.strain_tensor(field, voxel_size=1.0)
    e2 = elphy.strain_tensor(field, voxel_size=2.0)
    sl = (slice(1, -1),) * 3
    assert np.allclose(e1["xz"][sl], 2 * e2["xz"][sl], atol=1e-9)


def test_oss_shear_quadrature_matches_analytic():
    # Two shear components in phase quadrature of equal magnitude A give a
    # time-*constant* OSS = 2/3*sqrt(6)*A, so the quadrature is exact.
    #   e_xz = A (real)      from u_z = 2A x
    #   e_xy = iA (imaginary) from u_y = 2iA x
    X, Y, Z = _grid()
    A = 0.4
    field = np.zeros(X.shape + (3,), dtype=complex)
    field[..., 2] = (2 * A) * X  # e_xz = A
    field[..., 1] = (2j * A) * X  # e_xy = iA
    oss = elphy.octahedral_shear_strain(field, voxel_size=1.0)
    expected = (2.0 / 3.0) * np.sqrt(6.0) * A
    sl = (slice(1, -1),) * 3
    assert np.allclose(oss[sl], expected, rtol=1e-6)


def test_oss_single_component_averages_correctly():
    # A single real shear component gives OSS(t) proportional to |cos t|;
    # a high-order average recovers the closed form (2/pi) factor.
    X, Y, Z = _grid()
    c = 0.8
    field = np.zeros(X.shape + (3,), dtype=complex)
    field[..., 2] = c * X  # e_xz = c/2
    oss = elphy.octahedral_shear_strain(field, voxel_size=1.0, n_gauss=201)
    exz = c / 2.0
    expected = (2.0 / 3.0) * np.sqrt(6.0) * exz * (2.0 / np.pi)
    sl = (slice(1, -1),) * 3
    assert np.allclose(oss[sl], expected, rtol=2e-3)


def test_oss_shear_orientation_invariant():
    # Pure shear in the x-z plane and in the x-y plane give identical OSS.
    X, Y, Z = _grid()
    c = 0.8
    f_xz = np.zeros(X.shape + (3,), dtype=complex)
    f_xz[..., 2] = c * X  # e_xz
    f_xy = np.zeros(X.shape + (3,), dtype=complex)
    f_xy[..., 1] = c * X  # e_xy
    o_xz = elphy.octahedral_shear_strain(f_xz, voxel_size=1.0)
    o_xy = elphy.octahedral_shear_strain(f_xy, voxel_size=1.0)
    sl = (slice(1, -1),) * 3
    assert np.allclose(o_xz[sl], o_xy[sl], rtol=1e-6)


def test_oss_zero_field_is_zero():
    field = np.zeros((10, 10, 10, 3), dtype=complex)
    oss = elphy.octahedral_shear_strain(field, voxel_size=1.0)
    assert np.allclose(oss, 0.0)


def test_ons_pure_dilation_matches_analytic():
    # u = (c*x, c*y, c*z) -> e_xx=e_yy=e_zz=c ; ONS = c ; time-avg |ONS| = (2/pi)c.
    X, Y, Z = _grid()
    c = 0.5
    field = np.zeros(X.shape + (3,), dtype=complex)
    field[..., 0] = c * X
    field[..., 1] = c * Y
    field[..., 2] = c * Z
    ons = elphy.octahedral_normal_strain(field, voxel_size=1.0)
    expected = (2.0 / np.pi) * c
    sl = (slice(1, -1),) * 3
    assert np.allclose(ons[sl], expected, rtol=1e-6)
    # A pure dilation has zero octahedral *shear* strain.
    oss = elphy.octahedral_shear_strain(field, voxel_size=1.0)
    assert np.allclose(oss[sl], 0.0, atol=1e-9)


def test_gauss_quadrature_converges():
    # 5-point vs 21-point half-period average agree to <1% on pure shear.
    X, Y, Z = _grid()
    field = np.zeros(X.shape + (3,), dtype=complex)
    # Smooth, non-degenerate field: two shear components of unequal magnitude
    # (never simultaneously zero) -> OSS(t) is smooth, quadrature converges.
    field[..., 2] = 0.4 * X  # e_xz = 0.2 (real)
    field[..., 1] = 0.5j * Y  # e_xy = 0.25i (imaginary)
    o5 = elphy.octahedral_shear_strain(field, voxel_size=1.0, n_gauss=5)
    o21 = elphy.octahedral_shear_strain(field, voxel_size=1.0, n_gauss=21)
    sl = (slice(1, -1),) * 3
    rel = np.abs(o5[sl] - o21[sl]) / np.abs(o21[sl])
    assert np.nanmax(rel) < 1e-2


def test_motion_noise_amplitude_recovers_sqrt2n_std():
    # Offsets = fitted sinusoid + known white noise -> N_amp = sqrt(2/N)*std(misfit).
    rng = np.random.default_rng(0)
    n = 8
    amp = np.full((6, 6, 6), 1.0 + 0.0j)
    k = np.arange(n)
    t = 2 * np.pi * k / n
    fit = np.real(amp[..., None] * np.exp(1j * t))
    noise = rng.normal(0, 0.1, fit.shape)
    offsets = np.moveaxis(fit + noise, -1, 3)  # offset axis = 3
    n_amp = elphy.motion_noise_amplitude(offsets, amp, offset_axis=3, harmonic=1)
    manual = np.sqrt(2.0 / n) * np.std(noise, axis=-1)
    assert np.allclose(n_amp, manual, rtol=1e-9)


def test_oss_snr_scales_inversely_with_noise():
    # Halving the motion noise doubles the OSS-SNR (both means are linear in noise).
    X, Y, Z = _grid()
    field = np.zeros(X.shape + (3,), dtype=complex)
    field[..., 2] = 0.8 * X
    mask = np.ones(X.shape, dtype=bool)
    noise1 = np.full(X.shape, 0.05)
    noise2 = 0.5 * noise1
    snr1 = elphy.oss_snr(field, noise1, voxel_size=1.0, mask=mask)
    snr2 = elphy.oss_snr(field, noise2, voxel_size=1.0, mask=mask)
    assert snr2 == pytest.approx(2 * snr1, rel=1e-6)


def test_oss_snr_distribution_shape_and_roi():
    X, Y, Z = _grid()
    field = np.zeros(X.shape + (3,), dtype=complex)
    field[..., 2] = 0.8 * X
    mask = np.zeros(X.shape, dtype=bool)
    mask[2:-2, 2:-2, 2:-2] = True
    snr, dist = elphy.oss_snr(
        field,
        np.full(X.shape, 0.05),
        voxel_size=1.0,
        mask=mask,
        return_distribution=True,
    )
    assert dist.shape == X.shape
    # Distribution is zero outside the eroded ROI.
    from scipy import ndimage

    roi = ndimage.binary_erosion(mask, iterations=1)
    assert np.allclose(dist[~roi], 0.0)
    assert np.isfinite(snr) and snr > 0


def test_oss_snr_per_frequency_axis():
    # A trailing frequency axis yields one SNR per frequency.
    X, Y, Z = _grid()
    field = np.zeros(X.shape + (3, 2), dtype=complex)
    field[..., 2, 0] = 0.8 * X
    field[..., 2, 1] = 1.6 * X  # twice the amplitude -> same SNR (noise scales too? no)
    mask = np.ones(X.shape, dtype=bool)
    noise = np.full(X.shape + (2,), 0.05)  # per-direction map broadcast? shape base
    # noise shaped like one strain map (spatial + freq): broadcast over directions.
    snr = elphy.oss_snr(field, noise, voxel_size=1.0, mask=mask)
    assert np.asarray(snr).shape == (2,)
    # Second frequency has twice the signal, same noise -> twice the SNR.
    assert snr[1] == pytest.approx(2 * snr[0], rel=1e-6)
