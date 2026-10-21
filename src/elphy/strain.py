"""Octahedral-strain quality metrics for MRE displacement wavefields.

This module ports the octahedral shear strain (OSS) SNR of McGarry et al.,
*An octahedral shear strain-based measure of SNR for 3D MR elastography*
(Phys. Med. Biol. 56 (2011) N153), following the MATLAB reference
``oss_snr_filter.m`` / ``Strain_SNR.m`` (M. McGarry, Dartmouth College;
adapted for MRE by C. L. Johnson).  It operates on the complex displacement
wavefield produced by :func:`elphy.wavefield.temporal_harmonic` -- one
complex amplitude per voxel, direction, and frequency.

Definitions
-----------
For a displacement field ``u = (u_x, u_y, u_z)`` the small-strain tensor is

    e_ij = 1/2 (du_i/dx_j + du_j/dx_i).

The **octahedral shear strain** (deviatoric, Eq. 7 of McGarry 2011) is

    OSS = 2/3 * sqrt( (e_xx - e_yy)^2 + (e_xx - e_zz)^2 + (e_yy - e_zz)^2
                      + 6 (e_xy^2 + e_xz^2 + e_yz^2) ),

and the **octahedral normal strain** (dilatational, Eq. 6) is

    ONS = 1/3 (e_xx + e_yy + e_zz).

OSS is a non-linear function of the strains, so a complex harmonic strain
amplitude does not map to a single complex OSS amplitude.  Following the
paper, the time-averaged OSS over one cycle is obtained by numerically
integrating ``OSS(t)`` over half a period (OSS is periodic at twice the
drive frequency) with 5-point Gaussian quadrature -- a median error of
0.3% versus a 21-point reference.  ONS is linear and stays harmonic, so
its time-averaged absolute value is ``(2/pi) * |ONS_complex|`` (Eq. 10).

The **OSS-SNR** (Eq. 12) is the ROI mean of the time-averaged OSS divided
by the ROI mean of the time-averaged OSS of the motion *noise*, where the
noise OSS is obtained by propagating the per-voxel motion-noise amplitude
through the OSS equation (propagation of uncertainties with a central-
difference strain estimate).

Axis convention
---------------
The complex vector wavefield is laid out as ``field[x, y, z, direction,
frequency]``; ``direction`` indexes ``(u_x, u_y, u_z)`` and the optional
trailing ``frequency`` axis (and any further axes) is broadcast over
transparently.

Scope note
----------
The empirical calibration factors applied in the original
``oss_snr_filter.m`` (``* 1.35`` for 8 offsets, ``* 1.35 / sqrt(2.5)`` for
4 offsets) are site/sequence-specific corrections and are intentionally
*not* applied here; apply them downstream in site code if you need to match
a particular historical calibration.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "strain_tensor",
    "octahedral_shear_strain",
    "octahedral_normal_strain",
    "motion_noise_amplitude",
    "oss_snr",
]


def _spacing(voxel_size, ndim=3):
    spacing = np.atleast_1d(np.asarray(voxel_size, dtype=float))
    if spacing.size == 1:
        spacing = np.repeat(spacing, ndim)
    if spacing.size != ndim:
        raise ValueError(f"voxel_size must be a scalar or length-{ndim} sequence")
    if np.any(spacing <= 0):
        raise ValueError("voxel_size entries must be positive")
    return tuple(float(s) for s in spacing)


def _gauss_half_period(n_gauss):
    """Gauss-Legendre nodes/weights mapped to the half period ``[0, pi]``.

    Returns ``(xi, weight_factor)`` such that the cycle average of a
    half-period-integrable function ``f`` is ``sum(weight_factor * f(xi))``.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_gauss)
    a, b = 0.0, np.pi
    xi = 0.5 * (a + b) + 0.5 * (b - a) * nodes
    # Average over the full 2*pi cycle: (1/pi) * integral_0^pi.
    wfac = weights * ((b - a) / 2.0) / np.pi
    return xi, wfac


def _direction_components(field, direction_axis):
    if field.shape[direction_axis] != 3:
        raise ValueError(
            f"direction axis {direction_axis} must have length 3, "
            f"got {field.shape[direction_axis]}"
        )
    return [np.take(field, i, axis=direction_axis) for i in range(3)]


def strain_tensor(field, voxel_size, direction_axis=3, spatial_axes=(0, 1, 2)):
    r"""Complex small-strain tensor components of a vector wavefield.

    Computes the six unique components of the symmetric strain tensor
    ``e_ij = 1/2 (du_i/dx_j + du_j/dx_i)`` with central finite differences
    (``numpy.gradient``).  Input may be complex (harmonic amplitudes) or
    real; the output preserves the dtype.

    Parameters
    ----------
    field : array_like
        Vector wavefield with a length-3 ``direction_axis`` holding
        ``(u_x, u_y, u_z)`` and three spatial axes.
    voxel_size : float or sequence of float
        Spacing along the three spatial axes (scalar broadcasts).
    direction_axis : int, optional
        Axis holding the three vector components (default ``3``).
    spatial_axes : tuple of int, optional
        The three spatial axes ordered ``(x, y, z)`` (default ``(0, 1, 2)``).

    Returns
    -------
    dict
        Keys ``"xx", "yy", "zz", "xy", "xz", "yz"`` mapping to arrays with
        ``direction_axis`` removed (same shape as one vector component).
    """
    field = np.asarray(field)
    if len(spatial_axes) != 3:
        raise ValueError("strain_tensor requires exactly 3 spatial axes")
    ax, ay, az = (int(a) for a in spatial_axes)
    dx, dy, dz = _spacing(voxel_size, 3)
    ux, uy, uz = _direction_components(field, direction_axis)

    def gaxis(a):
        return a - 1 if a > direction_axis else a

    def grad(arr, a, spacing):
        return np.gradient(arr, spacing, axis=gaxis(a))

    exx = grad(ux, ax, dx)
    eyy = grad(uy, ay, dy)
    ezz = grad(uz, az, dz)
    exy = 0.5 * (grad(ux, ay, dy) + grad(uy, ax, dx))
    exz = 0.5 * (grad(ux, az, dz) + grad(uz, ax, dx))
    eyz = 0.5 * (grad(uy, az, dz) + grad(uz, ay, dy))
    return {"xx": exx, "yy": eyy, "zz": ezz, "xy": exy, "xz": exz, "yz": eyz}


def _oss_from_real_strains(e):
    """OSS (Eq. 7) from a dict of *real* strain components."""
    q = (
        (e["xx"] - e["yy"]) ** 2
        + (e["xx"] - e["zz"]) ** 2
        + (e["yy"] - e["zz"]) ** 2
        + 6.0 * (e["xy"] ** 2 + e["xz"] ** 2 + e["yz"] ** 2)
    )
    return (2.0 / 3.0) * np.sqrt(q)


def octahedral_shear_strain(
    field, voxel_size, direction_axis=3, spatial_axes=(0, 1, 2), n_gauss=5
):
    r"""Time-averaged octahedral shear strain of a complex wavefield.

    Evaluates Eq. 7 of McGarry et al. (2011) and averages it over one
    vibration cycle by 5-point Gaussian quadrature over the half period
    (Eq. 9).  OSS is the deviatoric strain invariant -- the maximum shear
    strain on any plane -- and is independent of coordinate orientation.

    Parameters
    ----------
    field : array_like
        Complex vector wavefield ``[..., direction, ...]`` (harmonic
        amplitudes).  A real field is treated as a zero-phase amplitude.
    voxel_size : float or sequence of float
        Spacing along the three spatial axes (scalar broadcasts).
    direction_axis, spatial_axes
        Axis layout, as in :func:`strain_tensor`.
    n_gauss : int, optional
        Number of Gauss-Legendre points for the half-period average
        (default ``5``, the paper's choice).

    Returns
    -------
    numpy.ndarray
        Real, non-negative time-averaged OSS with ``direction_axis``
        removed.
    """
    e = strain_tensor(field, voxel_size, direction_axis, spatial_axes)
    e = {k: np.asarray(v, dtype=complex) for k, v in e.items()}
    xi, wfac = _gauss_half_period(n_gauss)
    oss = np.zeros(e["xx"].shape, dtype=float)
    for x, w in zip(xi, wfac, strict=True):
        phase = np.exp(1j * x)
        er = {k: np.real(v * phase) for k, v in e.items()}
        oss = oss + w * _oss_from_real_strains(er)
    return oss


def octahedral_normal_strain(field, voxel_size, direction_axis=3, spatial_axes=(0, 1, 2)):
    r"""Time-averaged absolute octahedral normal strain (Eq. 6 / Eq. 10).

    The octahedral normal strain ``ONS = 1/3 (e_xx + e_yy + e_zz)`` is
    linear, so it stays harmonic; its time-averaged absolute value over one
    cycle is ``(2/pi) * |ONS_complex|`` (Eq. 10 of McGarry et al. 2011).

    Returns
    -------
    numpy.ndarray
        Real, non-negative time-averaged \|ONS\| with ``direction_axis``
        removed.
    """
    e = strain_tensor(field, voxel_size, direction_axis, spatial_axes)
    ons = (e["xx"] + e["yy"] + e["zz"]) / 3.0
    return (2.0 / np.pi) * np.abs(ons)


def motion_noise_amplitude(offsets, complex_amplitude, offset_axis=3, harmonic=1):
    r"""Per-voxel motion-amplitude noise from the temporal misfit.

    Estimates the noise in the fitted complex motion amplitude following
    Eqs. 2-3 of McGarry et al. (2011).  The measurement noise is the
    standard deviation of the misfit between the fitted fundamental
    sinusoid and the measured phase offsets,

        N_meas = std_t( d_i - Re(A * exp(i * omega * t_i)) ),

    and propagation through the DFT gives the amplitude noise

        N_amp = sqrt(2 / N) * N_meas,

    which is the same for the real part, imaginary part, and magnitude of
    the complex amplitude ``A``.

    Parameters
    ----------
    offsets : array_like
        Real per-offset motion field with the temporal-offset axis intact
        (e.g. the unwrapped, polarity-combined phase before harmonic
        extraction), shape ``[..., N, ...]`` along ``offset_axis``.
    complex_amplitude : array_like
        The fitted complex harmonic amplitude
        (:func:`elphy.wavefield.temporal_harmonic` output) with the offset
        axis already removed.
    offset_axis : int, optional
        Axis of ``offsets`` holding the ``N`` temporal offsets
        (default ``3``).
    harmonic : int, optional
        Harmonic index that ``complex_amplitude`` represents (default
        ``1``, the fundamental).

    Returns
    -------
    numpy.ndarray
        ``N_amp``, same shape as ``complex_amplitude``.
    """
    offsets = np.asarray(offsets, dtype=float)
    amp = np.asarray(complex_amplitude, dtype=complex)
    n = offsets.shape[offset_axis]
    k = np.arange(n)
    t = 2.0 * np.pi * harmonic * k / n
    # Reconstruct the fitted sinusoid at each offset: Re(A * exp(i * t_k)).
    amp_e = np.expand_dims(amp, offset_axis)
    shape = [1] * offsets.ndim
    shape[offset_axis] = n
    phase = np.exp(1j * t).reshape(shape)
    fit = np.real(amp_e * phase)
    misfit = offsets - fit
    n_meas = np.std(misfit, axis=offset_axis)
    return np.sqrt(2.0 / n) * n_meas


def oss_snr(
    field,
    motion_noise,
    voxel_size,
    mask=None,
    direction_axis=3,
    spatial_axes=(0, 1, 2),
    n_gauss=5,
    return_distribution=False,
):
    r"""Octahedral shear strain SNR of a complex displacement wavefield.

    Implements the OSS-SNR of McGarry et al. (2011), Eq. 12::

        OSS-SNR = mean_ROI( time-avg OSS ) / mean_ROI( time-avg OSS_noise )

    The signal OSS is :func:`octahedral_shear_strain`.  The noise OSS is
    obtained by propagating the per-direction motion-noise amplitude
    ``motion_noise`` through the OSS equation (propagation of
    uncertainties, central-difference strains), time-averaged over the
    half period with the same Gaussian quadrature.

    This corrects an indexing error in the original ``OSS_noise.m``
    reference (its ``z``-derivative terms read ``x``-neighbours); the
    strain derivatives here follow Eq. 5/7 of the paper.

    Parameters
    ----------
    field : array_like
        Complex vector wavefield ``[x, y, z, direction, (frequency)]``.
    motion_noise : array_like
        Per-voxel motion-amplitude noise ``N_amp`` (see
        :func:`motion_noise_amplitude`).  Either one map broadcast to all
        three directions, or a per-direction array with a length-3
        ``direction_axis`` matching ``field``.
    voxel_size : float or sequence of float
        Spacing along the three spatial axes (scalar broadcasts).  A single
        mean spacing is used for the propagation-of-errors scaling.
    mask : array_like of bool, optional
        ROI over the three spatial axes.  Voxels within one pixel of the
        mask edge are excluded (central differences are undefined there).
        If ``None``, the whole volume interior is used.
    direction_axis, spatial_axes, n_gauss
        As in :func:`octahedral_shear_strain`.
    return_distribution : bool, optional
        If ``True`` also return the per-voxel OSS-SNR map.

    Returns
    -------
    float or (float, numpy.ndarray)
        The scalar ROI OSS-SNR, and (if requested) the per-voxel SNR map
        (zero outside the valid ROI).  If ``field`` has a trailing
        frequency axis, the scalar is returned per frequency as an array.
    """
    field = np.asarray(field)
    e = strain_tensor(field, voxel_size, direction_axis, spatial_axes)
    e = {k: np.asarray(v, dtype=complex) for k, v in e.items()}
    base_shape = e["xx"].shape  # spatial (+ trailing freq), direction removed
    h = float(np.mean(_spacing(voxel_size, 3)))

    # Per-direction motion noise, aligned to the strain-map shape.
    mn = np.asarray(motion_noise, dtype=float)
    if mn.shape == base_shape:
        su = sv = sw = mn
    else:
        su, sv, sw = _direction_components(mn, direction_axis)

    xi, wfac = _gauss_half_period(n_gauss)
    oss = np.zeros(base_shape, dtype=float)
    oss_nse = np.zeros(base_shape, dtype=float)
    for x, w in zip(xi, wfac, strict=True):
        phase = np.exp(1j * x)
        er = {k: np.real(v * phase) for k, v in e.items()}
        so = _oss_from_real_strains(er)
        oss = oss + w * so
        # Propagation of errors: sigma_OSS^2 = sum_d (dOSS/d e_ij)^2 * var(e_ij)
        # via var(e_ij) from central differences of the noisy displacements.
        so_safe = np.where(so > 0, so, np.inf)
        # dOSS/d e_ij  (OSS = 2/3 sqrt(Q); dOSS/de = (1/(3 so)) dQ/de)
        dxx = (2.0 * er["xx"] - er["yy"] - er["zz"]) * 2.0 / (9.0 * so_safe)
        dyy = (2.0 * er["yy"] - er["xx"] - er["zz"]) * 2.0 / (9.0 * so_safe)
        dzz = (2.0 * er["zz"] - er["xx"] - er["yy"]) * 2.0 / (9.0 * so_safe)
        dxy = 12.0 * er["xy"] / (9.0 * so_safe)
        dxz = 12.0 * er["xz"] / (9.0 * so_safe)
        dyz = 12.0 * er["yz"] / (9.0 * so_safe)
        # Central-difference stencil variance coefficients:
        #  normal strain e_ii = (u(+1)-u(-1))/(2h): two nbrs -> 2*(1/2h)^2 = 1/(2h^2)
        #  shear  e_ij = 1/2(du_i/dj+du_j/di): each half-term (1/4h) over two nbrs
        cn = 1.0 / (2.0 * h**2)
        cs = 1.0 / (8.0 * h**2)
        var = (
            dxx**2 * su**2 * cn
            + dyy**2 * sv**2 * cn
            + dzz**2 * sw**2 * cn
            + dxy**2 * (su**2 + sv**2) * cs
            + dxz**2 * (su**2 + sw**2) * cs
            + dyz**2 * (sv**2 + sw**2) * cs
        )
        oss_nse = oss_nse + w * np.sqrt(var)

    # Build the valid ROI: mask interior (erode by one voxel along spatial axes).
    roi = np.ones([field.shape[a] for a in spatial_axes], dtype=bool)
    if mask is not None:
        roi = np.asarray(mask, dtype=bool)
    from scipy import ndimage

    roi = ndimage.binary_erosion(roi, iterations=1)
    # Broadcast the spatial ROI over any trailing axes of the strain maps.
    roi_full = roi.reshape(roi.shape + (1,) * (oss.ndim - roi.ndim))
    roi_full = np.broadcast_to(roi_full, oss.shape)

    with np.errstate(divide="ignore", invalid="ignore"):
        snr_map = np.where(oss_nse > 0, oss / oss_nse, 0.0)
    snr_map = np.where(roi_full, snr_map, 0.0)

    # Scalar SNR = mean(signal)/mean(noise) over the ROI, per trailing axis.
    extra = tuple(range(roi.ndim, oss.ndim))
    if extra:
        sig = np.where(roi_full, oss, np.nan)
        nse = np.where(roi_full, oss_nse, np.nan)
        snr = np.nanmean(sig, axis=tuple(range(roi.ndim))) / np.nanmean(
            nse, axis=tuple(range(roi.ndim))
        )
    else:
        snr = float(oss[roi].mean() / oss_nse[roi].mean())
    if return_distribution:
        return snr, snr_map
    return snr
