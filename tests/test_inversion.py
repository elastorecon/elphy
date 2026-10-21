"""Analytic tests for elphy.inversion (direct inversion and LFE).

Ground truth uses a shear plane wave u = A exp(i k x) in a homogeneous
medium: the Laplacian is -k^2 u, so the direct Helmholtz solve returns
G* = rho * omega^2 / k^2 = rho * f^2 * lambda^2, and LFE returns the local
spatial frequency 1/lambda (cycles/m), i.e. mu = rho (f/k)^2 = rho f^2 lambda^2.
"""

import numpy as np
import pytest

import elphy

RHO = 1000.0


def _shear_plane_wave(n=64, vox_mm=1.5, wavelength_mm=30.0, amp=1.0, polar=1, prop=0):
    """Complex shear plane wave on an n^3 grid, propagating along axis `prop`,
    polarized along direction `polar`."""
    coords = [np.arange(n) * vox_mm for _ in range(3)]
    grids = np.meshgrid(*coords, indexing="ij")
    k = 2 * np.pi / wavelength_mm  # rad/mm along propagation
    wave = amp * np.exp(1j * k * grids[prop])
    field = np.zeros((n, n, n, 3), dtype=complex)
    field[..., polar] = wave
    return field


def _expected_modulus(f_hz, wavelength_mm, rho=RHO):
    lam_m = wavelength_mm / 1000.0
    return rho * (f_hz**2) * (lam_m**2)


def _interior(vol, m=8):
    return vol[m:-m, m:-m, m:-m]


def test_direct_inversion_recovers_plane_wave_modulus():
    f = 50.0
    lam = 30.0
    field = _shear_plane_wave(n=64, vox_mm=1.5, wavelength_mm=lam)
    G = elphy.direct_inversion(field, voxel_size=1.5, frequency=f)
    expected = _expected_modulus(f, lam)  # 2250 Pa
    got = np.median(np.abs(_interior(G)))
    assert got == pytest.approx(expected, rel=0.02)


def test_curl_inversion_recovers_plane_wave_modulus():
    f = 50.0
    lam = 30.0
    field = _shear_plane_wave(n=64, vox_mm=1.5, wavelength_mm=lam)
    G = elphy.curl_based_inversion(field, voxel_size=1.5, frequency=f)
    expected = _expected_modulus(f, lam)
    got = np.median(np.abs(_interior(G)))
    assert got == pytest.approx(expected, rel=0.03)


def test_curl_inversion_rejects_longitudinal_wave():
    # A pure longitudinal (compressional) wave: u parallel to propagation.
    # Its curl is ~0, so curl-based inversion should not report the (wrong)
    # shear modulus that direct inversion would from the bulk motion.
    n, vox, lam = 64, 1.5, 30.0
    coords = [np.arange(n) * vox for _ in range(3)]
    X, Y, Z = np.meshgrid(*coords, indexing="ij")
    k = 2 * np.pi / lam
    longitudinal = np.zeros((n, n, n, 3), dtype=complex)
    longitudinal[..., 0] = np.exp(1j * k * X)  # u_x propagating in x
    q = elphy.curl(longitudinal, vox / 1000.0)
    assert np.max(np.abs(_interior(q))) < 1e-6


def test_modulus_scales_with_frequency_squared():
    lam = 30.0
    field = _shear_plane_wave(n=64, wavelength_mm=lam)
    g50 = np.median(np.abs(_interior(elphy.direct_inversion(field, 1.5, 50.0))))
    g30 = np.median(np.abs(_interior(elphy.direct_inversion(field, 1.5, 30.0))))
    # G* proportional to omega^2 -> ratio (50/30)^2.
    assert g50 / g30 == pytest.approx((50.0 / 30.0) ** 2, rel=1e-3)


def test_storage_loss_split_lossless_wave():
    # A propagating wave with no attenuation is (near) purely elastic:
    # G' dominates, G'' ~ 0.
    field = _shear_plane_wave(n=64, wavelength_mm=30.0)
    G = elphy.direct_inversion(field, 1.5, 50.0)
    Gi = _interior(G)
    assert np.median(Gi.real) == pytest.approx(2250.0, rel=0.02)
    assert np.median(np.abs(Gi.imag)) < 0.02 * np.median(Gi.real)


def test_shear_stiffness_is_abs():
    G = np.array([3.0 + 4.0j, -6.0 + 8.0j])
    assert np.allclose(elphy.shear_stiffness(G), [5.0, 10.0])


def test_smoothing_runs_and_stays_close():
    field = _shear_plane_wave(n=64, wavelength_mm=30.0)
    G = elphy.curl_based_inversion(field, 1.5, 50.0, smoothing=0.8)
    got = np.median(np.abs(_interior(G)))
    assert got == pytest.approx(2250.0, rel=0.05)


# ------------------------- LFE -------------------------


def _lfe_wave_2d(n=128, vox_mm=1.5, wavelength_mm=30.0):
    x = np.arange(n) * vox_mm
    X, Y = np.meshgrid(x, x, indexing="ij")
    k = 2 * np.pi / wavelength_mm
    return np.cos(k * X)  # real standing-oriented plane wave along x


def test_lfe_recovers_local_frequency():
    lam = 30.0  # mm
    img = _lfe_wave_2d(n=128, vox_mm=1.5, wavelength_mm=lam)
    k_map = elphy.local_frequency_estimation(img, voxel_size=1.5)
    expected = 1.0 / (lam / 1000.0)  # cycles/m = 33.33
    center = k_map[40:-40, 40:-40]
    assert np.median(center) == pytest.approx(expected, rel=0.15)


def test_lfe_higher_frequency_for_shorter_wavelength():
    img_long = _lfe_wave_2d(wavelength_mm=40.0)
    img_short = _lfe_wave_2d(wavelength_mm=20.0)
    k_long = np.median(elphy.local_frequency_estimation(img_long, 1.5)[40:-40, 40:-40])
    k_short = np.median(elphy.local_frequency_estimation(img_short, 1.5)[40:-40, 40:-40])
    assert k_short > k_long


def test_lfe_inversion_recovers_stiffness():
    f, lam = 50.0, 30.0
    img = _lfe_wave_2d(n=128, vox_mm=1.5, wavelength_mm=lam)
    mu = elphy.lfe_inversion(img, voxel_size=1.5, frequency=f)
    expected = RHO * (f * lam / 1000.0) ** 2  # 2250 Pa
    center = mu[40:-40, 40:-40]
    assert np.median(center) == pytest.approx(expected, rel=0.3)


def test_lfe_requires_square_slices():
    img = np.zeros((64, 48))
    with pytest.raises(ValueError, match="square"):
        elphy.local_frequency_estimation(img, 1.5)


def test_lfe_3d_slicewise():
    img = np.stack([_lfe_wave_2d(n=64, wavelength_mm=30.0) for _ in range(4)], axis=2)
    k = elphy.local_frequency_estimation(img, 1.5, spatial_axes=(0, 1), slice_axis=2)
    assert k.shape == (64, 64, 4)
    assert np.all(np.isfinite(k))
