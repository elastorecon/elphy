r"""Direct material-property estimation from MRE displacement wavefields.

Two classic, locally homogeneous inversions that turn a complex harmonic
displacement field into a shear-modulus / stiffness map:

* :func:`curl_based_inversion` -- algebraic Helmholtz (direct) inversion.
  Under local homogeneity with the longitudinal wave removed, each
  component of the displacement (or of its curl) obeys the Helmholtz
  equation and the complex shear modulus is

  .. math:: G^* = -\rho\,\omega^2\, u / \nabla^2 u .

  Taking the curl :math:`q=\nabla\times u` first suppresses the
  compressional (longitudinal) part, so the estimate is not biased by
  bulk motion (Sinkus et al., 2005).  The three curl components are
  combined by a least-squares solve for the single complex :math:`G^*`
  (Oliphant, 2001; Sinkus et al., 2005).  :func:`direct_inversion` is the
  same solve applied straight to the displacement, for data where the
  longitudinal wave is already negligible.

* :func:`local_frequency_estimation` -- LFE via a bank of lognormal
  quadrature filters (Knutsson 1994; Manduca 1996, 2001).  It estimates
  the local spatial frequency (inverse wavelength) from the ratio of
  responses of adjacent octave-spaced filters; :func:`lfe_inversion`
  converts that to shear stiffness through :math:`\mu=\rho c^2` with
  :math:`c=f\lambda`.  LFE is markedly less noise-sensitive than the
  Laplacian-based direct inversion but assumes no attenuation, so it
  yields an effective elastic stiffness.

Both assume a linear, isotropic, locally homogeneous medium.  Density
defaults to 1000 kg/m^3 (soft tissue ~ water).  The modulus from the
Helmholtz solve is independent of the displacement *amplitude units*
(numerator and Laplacian scale together), but the spatial spacing must be
physical: ``voxel_size`` is taken in millimetres and converted to metres
internally, and ``frequency`` is the mechanical drive frequency in Hz.
"""

from __future__ import annotations

import numpy as np

from .operators import curl

__all__ = [
    "direct_inversion",
    "curl_based_inversion",
    "local_frequency_estimation",
    "lfe_inversion",
    "shear_stiffness",
]


def _laplacian(vol, spacing, spatial_axes):
    r"""Laplacian of a (complex) array via the compact 3-point stencil.

    Uses the second central difference
    :math:`(u_{i+1}-2u_i+u_{i-1})/h^2` per axis, which for a plane wave has
    a smaller discretization error than differencing the gradient twice.
    Boundary voxels use one-sided wrap (``np.roll``); callers exclude a
    boundary margin (the ROI erosion / interior slice).
    """
    lap = np.zeros_like(vol, dtype=np.complex128)
    for ax, h in zip(spatial_axes, spacing, strict=True):
        fwd = np.roll(vol, -1, axis=ax)
        bwd = np.roll(vol, 1, axis=ax)
        lap += (fwd - 2.0 * vol + bwd) / (h * h)
    return lap


def _smooth_complex(field, sigma, spatial_axes):
    from scipy.ndimage import gaussian_filter

    sig = [0.0] * field.ndim
    for ax in spatial_axes:
        sig[ax] = sigma
    real = gaussian_filter(field.real, sig)
    imag = gaussian_filter(field.imag, sig)
    return real + 1j * imag


def _helmholtz_least_squares(field, spacing, omega, density, spatial_axes, direction_axis):
    r"""Least-squares complex shear modulus from a vector Helmholtz field.

    Solves :math:`\min_{G} \sum_j |\nabla^2 u_j\, G + \rho\omega^2 u_j|^2`
    voxel-wise over the vector components ``j``:

    .. math:: G^* = -\rho\omega^2
        \frac{\sum_j \overline{\nabla^2 u_j}\, u_j}{\sum_j |\nabla^2 u_j|^2}.
    """
    ncomp = field.shape[direction_axis]
    num = None
    den = None
    for j in range(ncomp):
        u = np.take(field, j, axis=direction_axis)
        lap = _laplacian(u, spacing, spatial_axes)
        n = np.conj(lap) * u
        d = np.abs(lap) ** 2
        num = n if num is None else num + n
        den = d if den is None else den + d
    with np.errstate(divide="ignore", invalid="ignore"):
        g = -density * omega**2 * num / den
    g[~np.isfinite(g)] = 0.0
    return g


def direct_inversion(
    field,
    voxel_size,
    frequency,
    density=1000.0,
    spatial_axes=(0, 1, 2),
    direction_axis=3,
    smoothing=None,
):
    r"""Algebraic Helmholtz (direct) inversion of a displacement field.

    Estimates the complex shear modulus :math:`G^*=G'+iG''` voxel-wise
    from :math:`G^*=-\rho\omega^2 u/\nabla^2 u`, combining the vector
    components by a least-squares solve.  Use this when the longitudinal
    wave is already negligible; otherwise prefer
    :func:`curl_based_inversion`.

    Parameters
    ----------
    field : array_like
        Complex harmonic displacement, ``[x, y, z, direction]`` (a single
        frequency).  A trailing frequency axis is *not* handled here --
        select one frequency first.
    voxel_size : float or sequence of float
        Spatial spacing in millimetres (scalar broadcasts to 3 axes).
    frequency : float
        Mechanical drive frequency in Hz.
    density : float, optional
        Material density in kg/m^3 (default 1000).
    spatial_axes : tuple of int, optional
        The three spatial axes (default ``(0, 1, 2)``).
    direction_axis : int, optional
        Axis of the three motion components (default 3).
    smoothing : float, optional
        Gaussian smoothing sigma (voxels) applied to the field before
        differentiation, to temper the noise sensitivity of the Laplacian.

    Returns
    -------
    numpy.ndarray
        Complex shear-modulus map :math:`G^*` in pascals, over the spatial
        axes.  ``.real`` is the storage modulus :math:`G'`, ``.imag`` the
        loss modulus :math:`G''`, ``abs`` the magnitude :math:`|G^*|`.
    """
    field = np.asarray(field).astype(np.complex128)
    spacing_m = _spacing_metres(voxel_size)
    if smoothing:
        field = _smooth_complex(field, smoothing, spatial_axes)
    omega = 2.0 * np.pi * float(frequency)
    return _helmholtz_least_squares(field, spacing_m, omega, density, spatial_axes, direction_axis)


def curl_based_inversion(
    field,
    voxel_size,
    frequency,
    density=1000.0,
    spatial_axes=(0, 1, 2),
    direction_axis=3,
    smoothing=None,
):
    r"""Curl-based algebraic Helmholtz inversion (longitudinal-wave free).

    Takes the curl :math:`q=\nabla\times u` of the displacement to remove
    the compressional component, then applies the same least-squares
    Helmholtz solve to the three curl components.  This is the standard
    "direct inversion" of Sinkus et al. (2005).

    Parameters and returns are as for :func:`direct_inversion`; the
    smoothing (if given) is applied to the displacement *before* the curl.

    Notes
    -----
    The curl amplifies high-frequency noise, so modest ``smoothing`` (or an
    upstream band-pass, :func:`elphy.operators.bandpass_filter`) is usually
    warranted on real data.
    """
    field = np.asarray(field).astype(np.complex128)
    spacing_m = _spacing_metres(voxel_size)
    if smoothing:
        field = _smooth_complex(field, smoothing, spatial_axes)
    q = curl(field, spacing_m, spatial_axes=spatial_axes, direction_axis=direction_axis)
    omega = 2.0 * np.pi * float(frequency)
    return _helmholtz_least_squares(q, spacing_m, omega, density, spatial_axes, direction_axis)


def _spacing_metres(voxel_size):
    s = np.atleast_1d(np.asarray(voxel_size, dtype=float))
    if s.size == 1:
        s = np.repeat(s, 3)
    if s.size != 3 or np.any(s <= 0):
        raise ValueError("voxel_size must be a positive scalar or length-3 sequence (mm)")
    return s / 1000.0  # mm -> m


def shear_stiffness(modulus):
    r"""Shear stiffness :math:`\mu=|G^*|` from a complex modulus map (Pa)."""
    return np.abs(np.asarray(modulus))


# --------------------------------------------------------------------------
# Local frequency estimation (Knutsson 1994 / Manduca 1996, 2001)
# --------------------------------------------------------------------------


def _lognormal_filter_bank(matrix, fov_m, n_dirs=24, octave_fraction=2, val1=2048, val2=2):
    """Build the 2-D lognormal quadrature filter bank of Manduca (1996).

    Returns ``(Q, rho_cent)`` where ``Q`` has shape
    ``(matrix, matrix, n_banks, n_dirs)`` (fftshifted, DC at centre) and
    ``rho_cent`` are the bank centre spatial frequencies in cycles/m.
    Ported from the reference ``filt2d.m``.
    """
    k_inc = 1.0 / fov_m  # frequency-grid increment, cycles/m
    c = (matrix / 2.0) + 0.5
    iy, ix = np.mgrid[0:matrix, 0:matrix]
    r = np.sqrt((iy - c) ** 2 + (ix - c) ** 2)
    r_safe = np.where(r == 0, 1e-12, r)
    s_x = (ix - c) / r_safe
    s_y = (iy - c) / r_safe
    rho = r * k_inc
    rho_safe = np.where(rho == 0, 1e-12, rho)

    octs = np.log(val1 / val2) / np.log(2)
    n_banks = int(round(octs * octave_fraction)) + 1

    B = np.sqrt(8.0)
    C_B = (4.0 * octave_fraction) / (B**2 * np.log(2))

    rho_cent = np.zeros(n_banks)
    for ibb in range(0, n_banks, octave_fraction):
        if ibb == 0:
            rho_cent[ibb] = (1.0 / val1) * k_inc * (matrix / 2.0)
        else:
            rho_cent[ibb] = rho_cent[ibb - octave_fraction] * 2.0
            if octave_fraction == 2:
                rho_cent[ibb - 1] = (rho_cent[ibb] + rho_cent[ibb - 2]) / 2.0

    # radial lognormal component
    R = np.empty((matrix, matrix, n_banks))
    for b in range(n_banks):
        R[:, :, b] = np.exp(-C_B * (np.log(rho_safe / rho_cent[b]) ** 2))

    # directional cos^2 half-plane component
    theta = (2 * np.pi / n_dirs) * np.arange(n_dirs)
    D = np.empty((matrix, matrix, n_dirs))
    for d in range(n_dirs):
        s_dot_n = s_x * np.cos(theta[d]) + s_y * np.sin(theta[d])
        D[:, :, d] = np.where(s_dot_n >= 0, s_dot_n**2, 0.0)

    Q = R[:, :, :, None] * D[:, :, None, :]
    return Q, rho_cent


def _lfe_slice(image, Q, rho_cent):
    """Local spatial-frequency estimate (cycles/m) for one 2-D slice."""
    n_banks = Q.shape[2]
    n_dirs = Q.shape[3]
    X = np.fft.fftshift(np.fft.fft2(image))

    di = np.empty(image.shape + (n_banks, n_dirs))
    for b in range(n_banks):
        for d in range(n_dirs):
            filt = np.fft.ifft2(np.fft.ifftshift(Q[:, :, b, d] * X))
            di[:, :, b, d] = np.abs(filt)

    c = np.sum(di**2, axis=(0, 1))  # (n_banks, n_dirs) energies
    c1 = np.sum(c, axis=0)  # per-direction total
    denom_c1 = np.sum(c1)
    # direction-weighted per-bank image
    d2 = np.tensordot(di, c1, axes=([3], [0])) / denom_c1  # (ny, nx, n_banks)
    c2 = np.sum(c, axis=1)  # per-bank total

    num = np.zeros(image.shape)
    for b in range(n_banks - 1):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = d2[:, :, b + 1] / d2[:, :, b]
        ratio[~np.isfinite(ratio)] = 0.0
        num += ratio * np.sqrt(rho_cent[b] * rho_cent[b + 1]) * c2[b]
    phat = num / np.sum(c2)
    return phat


def local_frequency_estimation(
    image,
    voxel_size,
    spatial_axes=(0, 1),
    slice_axis=2,
    n_dirs=24,
    octave_fraction=2,
):
    r"""Local spatial frequency (inverse wavelength) by LFE.

    Estimates the local spatial frequency of a wave image using a bank of
    lognormal quadrature filters (Knutsson 1994; Manduca 1996), ported from
    the reference ``filt2d.m`` / ``lfe2d.m``.  Operates in-plane on square
    2-D slices; a 3-D volume is processed slice-by-slice along
    ``slice_axis``.

    Parameters
    ----------
    image : array_like
        Wave image.  If complex, the real part is used.  2-D ``[x, y]`` or
        3-D ``[x, y, z]`` with the two in-plane axes square.
    voxel_size : float or sequence of float
        In-plane spacing in millimetres (must be isotropic in-plane).
    spatial_axes : tuple of int, optional
        The two in-plane axes (default ``(0, 1)``).
    slice_axis : int, optional
        Axis to iterate over for a 3-D volume (default 2).
    n_dirs : int, optional
        Number of directional filters (default 24).
    octave_fraction : int, optional
        Filter density: 1 = whole octaves, 2 = half-octave spacing
        (default 2, as in the reference).

    Returns
    -------
    numpy.ndarray
        Local spatial frequency in cycles per metre (``1/wavelength``),
        same shape as ``image``.
    """
    image = np.asarray(image)
    if np.iscomplexobj(image):
        image = image.real
    image = image.astype(float)

    spacing = np.atleast_1d(np.asarray(voxel_size, dtype=float))
    if spacing.size == 1:
        spacing = np.repeat(spacing, 2)
    ax0, ax1 = spatial_axes
    n0, n1 = image.shape[ax0], image.shape[ax1]
    if n0 != n1:
        raise ValueError(f"LFE requires square in-plane slices, got {n0}x{n1}")
    if not np.isclose(spacing[0], spacing[1]):
        raise ValueError("LFE requires isotropic in-plane spacing")
    fov_m = n0 * spacing[0] / 1000.0

    Q, rho_cent = _lognormal_filter_bank(n0, fov_m, n_dirs=n_dirs, octave_fraction=octave_fraction)

    if image.ndim == 2:
        return _lfe_slice(image, Q, rho_cent)
    if image.ndim == 3:
        out = np.empty_like(image)
        nsl = image.shape[slice_axis]
        for k in range(nsl):
            sl = np.take(image, k, axis=slice_axis)
            phat = _lfe_slice(sl, Q, rho_cent)
            idx = [slice(None)] * 3
            idx[slice_axis] = k
            out[tuple(idx)] = phat
        return out
    raise ValueError("image must be 2-D or 3-D")


def lfe_inversion(
    image,
    voxel_size,
    frequency,
    density=1000.0,
    spatial_axes=(0, 1),
    slice_axis=2,
    n_dirs=24,
    octave_fraction=2,
):
    r"""Shear stiffness (Pa) by local frequency estimation.

    Runs :func:`local_frequency_estimation` and converts the local spatial
    frequency :math:`k` (cycles/m) to an effective elastic shear stiffness
    through the wave speed :math:`c=f/k` and :math:`\mu=\rho c^2`:

    .. math:: \mu = \rho\,(f/k)^2 .

    Parameters are as for :func:`local_frequency_estimation`, plus
    ``frequency`` (Hz) and ``density`` (kg/m^3, default 1000).

    Returns
    -------
    numpy.ndarray
        Effective shear stiffness in pascals, same shape as ``image``.
    """
    phat = local_frequency_estimation(
        image,
        voxel_size,
        spatial_axes=spatial_axes,
        slice_axis=slice_axis,
        n_dirs=n_dirs,
        octave_fraction=octave_fraction,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        c = float(frequency) / phat  # wave speed, m/s
        mu = density * c**2
    mu[~np.isfinite(mu)] = 0.0
    return mu
