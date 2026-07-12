# -*- coding: utf-8 -*-
"""Core MRE wavefield operations.

This module contains the general, scanner-agnostic operations that turn
motion-encoded *phase* data into a *displacement* wavefield ready for a
material-property inversion.  Everything here operates on plain
:class:`numpy.ndarray` objects with an explicit, documented axis
convention so the functions can be reused regardless of how the data was
acquired or loaded.

Canonical axis convention
--------------------------
The full raw MRE phase array is 6-D::

    phase[x, y, z, encoding, direction, frequency]

* ``x, y, z``      -- spatial voxel indices.
* ``encoding``     -- motion-encoding samples.  For a balanced bipolar
                      acquisition there are ``2 * n_offset`` samples: the
                      motion-encoding-gradient (MEG) is played with both
                      positive and negative polarity at each temporal
                      offset (phase of the vibration cycle).  By the
                      convention used here (and by the Feng et al. 2025
                      public dataset) the samples are interleaved as
                      ``[+off0, -off0, +off1, -off1, ...]`` -- i.e. the
                      *even* Python indices ``0, 2, 4, ...`` are the
                      positive encodings and the *odd* indices
                      ``1, 3, 5, ...`` are the negative encodings.
* ``direction``    -- motion-encoding direction (typically 3: x, y, z).
* ``frequency``    -- vibration frequency.

Not every array carries all six axes (e.g. a single-frequency,
single-direction acquisition).  The functions below take the relevant
axis index as an argument and default to this canonical layout, so
trailing singleton axes can simply be omitted.

Physical model
--------------
A balanced bipolar MEG encodes tissue motion into the image phase as::

    phase_pos = +phi_motion + phi_background
    phase_neg = -phi_motion + phi_background

Subtracting the two polarities cancels the (static) background phase and
*doubles* the motion-encoded phase::

    phase_pos - phase_neg = 2 * phi_motion

so the combined phase has both a clean background and higher SNR.  The
combined phase is then spatially unwrapped and, optionally, scaled to a
physical displacement using the MEG sensitivity (radians per unit
displacement); see :func:`phase_to_displacement`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "wrap_to_pi",
    "combine_encodings",
    "unwrap_field",
    "phase_to_displacement",
    "meg_sensitivity_muthupillai",
    "raw_phase_to_wavefield",
]


def wrap_to_pi(phase):
    """Wrap phase values to the interval ``(-pi, pi]``.

    Parameters
    ----------
    phase : array_like
        Phase in radians.

    Returns
    -------
    numpy.ndarray
        Phase wrapped to ``(-pi, pi]``.
    """
    phase = np.asarray(phase)
    return np.angle(np.exp(1j * phase))


def combine_encodings(phase, encoding_axis=3, halve=False, polarity="even_positive"):
    """Combine positive/negative MEG polarities to cancel background phase.

    For a balanced bipolar acquisition each temporal offset is sampled
    with a positive and a negative motion-encoding-gradient polarity.
    Subtracting the two cancels the static background phase and doubles
    the motion-encoded phase (see module docstring).

    Parameters
    ----------
    phase : array_like
        Phase in radians with a motion-encoding axis holding
        ``2 * n_offset`` interleaved samples.
    encoding_axis : int, optional
        Axis holding the interleaved polarity/offset samples
        (default ``3``, the canonical layout).
    halve : bool, optional
        If ``True`` divide the polarity difference by two to recover the
        single-encoding motion phase ``phi_motion``.  If ``False``
        (default) return the full doubled-SNR difference
        ``2 * phi_motion``, matching the convention of the Feng et al.
        2025 public dataset.
    polarity : {"even_positive", "odd_positive"}, optional
        Which interleaved samples are the positive encodings.
        ``"even_positive"`` (default) treats Python indices
        ``0, 2, 4, ...`` as positive -- equivalent to the MATLAB
        convention where the *odd-numbered* (1-based) samples are
        positive.  ``"odd_positive"`` swaps the roles.

    Returns
    -------
    numpy.ndarray
        Combined, background-corrected motion phase, wrapped to
        ``(-pi, pi]``.  The length of ``encoding_axis`` is halved
        (one value per temporal offset).

    Raises
    ------
    ValueError
        If the encoding axis does not have an even length.
    """
    phase = np.asarray(phase, dtype=float)
    n = phase.shape[encoding_axis]
    if n % 2 != 0:
        raise ValueError(
            "encoding axis length must be even (positive/negative "
            f"polarity pairs); got {n} along axis {encoding_axis}"
        )
    if polarity == "even_positive":
        pos = np.take(phase, np.arange(0, n, 2), axis=encoding_axis)
        neg = np.take(phase, np.arange(1, n, 2), axis=encoding_axis)
    elif polarity == "odd_positive":
        pos = np.take(phase, np.arange(1, n, 2), axis=encoding_axis)
        neg = np.take(phase, np.arange(0, n, 2), axis=encoding_axis)
    else:
        raise ValueError(
            f"polarity must be 'even_positive' or 'odd_positive', got {polarity!r}"
        )
    diff = wrap_to_pi(pos - neg)
    if halve:
        diff = diff / 2.0
    return diff


def unwrap_field(phase, mask=None, spatial_axes=(0, 1, 2)):
    """Spatially unwrap a (masked) phase field.

    Wraps :func:`skimage.restoration.unwrap_phase`, applying the
    unwrap over the spatial axes for each combination of the remaining
    (non-spatial) axes.  A boolean ``mask`` restricts unwrapping to the
    voxels of interest (e.g. the object), which greatly improves
    robustness in the presence of background noise.

    Parameters
    ----------
    phase : array_like
        Wrapped phase in radians.  May have arbitrary trailing axes
        (encoding/offset, direction, frequency) beyond the spatial ones.
    mask : ndarray of bool, optional
        Spatial mask with shape ``phase.shape[:3]`` (the spatial axes).
        Voxels where ``mask`` is ``False`` are excluded from the
        unwrap and returned as ``0``.  If ``None`` (default) the whole
        volume is unwrapped.
    spatial_axes : tuple of int, optional
        The spatial axes to unwrap over (default ``(0, 1, 2)``).  Must be
        the leading axes.

    Returns
    -------
    numpy.ndarray
        Unwrapped phase, same shape and dtype ``float`` as ``phase``.

    Notes
    -----
    ``skimage`` is imported lazily so that importing :mod:`elphy` does
    not require it unless unwrapping is actually used.
    """
    from skimage.restoration import unwrap_phase as _skimage_unwrap

    phase = np.asarray(phase, dtype=float)
    if spatial_axes != tuple(range(len(spatial_axes))):
        raise ValueError("spatial_axes must be the leading axes, e.g. (0, 1, 2)")
    nsp = len(spatial_axes)
    spatial_shape = phase.shape[:nsp]
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != spatial_shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match spatial shape {spatial_shape}"
            )

    trailing_shape = phase.shape[nsp:]
    out = np.zeros_like(phase)
    # Iterate over every combination of the trailing axes.
    for idx in np.ndindex(*trailing_shape) if trailing_shape else [()]:
        vol = phase[(Ellipsis,) + idx] if idx else phase
        if mask is None:
            out[(Ellipsis,) + idx] = _skimage_unwrap(vol)
        else:
            masked = np.ma.array(vol, mask=~mask)
            uw = _skimage_unwrap(masked)
            filled = np.where(mask, np.ma.getdata(uw), 0.0)
            out[(Ellipsis,) + idx] = filled
    return out


def phase_to_displacement(phase, meg_sensitivity):
    """Convert motion-encoded phase to physical displacement.

    The motion-encoded phase relates to the physical displacement
    amplitude ``xi`` through the MEG sensitivity ``k`` (radians per unit
    displacement)::

        phase = k * xi   ==>   xi = phase / k

    This replaces any hard-coded phase-to-displacement scale factor with
    an explicit, physically meaningful calibration.  The sensitivity can
    be computed from the sequence parameters (see
    :func:`meg_sensitivity_muthupillai`) or, for the Feng et al. 2025
    dataset, taken from the per-frequency values reported in the data
    descriptor.

    Parameters
    ----------
    phase : array_like
        Motion-encoded phase in radians.
    meg_sensitivity : float or array_like
        MEG sensitivity in radians per unit displacement.  A scalar
        applies to the whole array; an array must broadcast against
        ``phase`` (e.g. one value per frequency along the last axis).

    Returns
    -------
    numpy.ndarray
        Displacement in the length unit implied by ``meg_sensitivity``
        (e.g. micrometres if ``k`` is in rad/micron).
    """
    phase = np.asarray(phase, dtype=float)
    k = np.asarray(meg_sensitivity, dtype=float)
    return phase / k


def meg_sensitivity_muthupillai(n_cycles, meg_period_s, gradient_T_per_m,
                                gyromagnetic_ratio=267.513e6):
    """MEG phase sensitivity from sequence parameters (Muthupillai 1995).

    Encoding sensitivity ``k`` (radians of phase per metre of
    displacement amplitude) for a sinusoidal motion-encoding gradient,
    following Muthupillai et al., *Science* 269:1854 (1995)::

        k = (2 * gamma * N * T * G) / pi

    where ``gamma`` is the gyromagnetic ratio (rad/s/T), ``N`` the number
    of gradient cycles, ``T`` the period of one gradient cycle (s), and
    ``G`` the gradient amplitude (T/m).  Port of the MATLAB
    ``mre_disp.m`` reference implementation.

    Parameters
    ----------
    n_cycles : float
        Number of motion-encoding-gradient cycles (``N``).
    meg_period_s : float
        Period of one MEG cycle in seconds (``T``); the reciprocal of
        the mechanical vibration frequency for a matched encoding.
    gradient_T_per_m : float
        MEG amplitude in tesla per metre (``G``).
    gyromagnetic_ratio : float, optional
        Proton gyromagnetic ratio in rad/s/T
        (default ``2.675e8`` = ``42.58 MHz/T * 2*pi``).

    Returns
    -------
    float
        Sensitivity ``k`` in radians per metre of displacement.  Divide
        a phase (radians) by this to obtain displacement in metres.
    """
    return (2.0 * gyromagnetic_ratio * n_cycles * meg_period_s * gradient_T_per_m) / np.pi


def raw_phase_to_wavefield(phase, mask=None, encoding_axis=3, halve=False,
                           polarity="even_positive", meg_sensitivity=None,
                           unwrap=True):
    """Full raw-phase to displacement-wavefield pipeline.

    Convenience wrapper chaining the standard steps:

    1. :func:`combine_encodings` -- cancel background, combine polarities.
    2. :func:`unwrap_field` -- spatial phase unwrap (optional).
    3. :func:`phase_to_displacement` -- scale to physical displacement
       (only if ``meg_sensitivity`` is given).

    Parameters
    ----------
    phase : array_like
        Raw motion-encoded phase, canonical layout
        ``[x, y, z, encoding, direction, frequency]`` (radians).
    mask : ndarray of bool, optional
        Spatial object mask, shape ``phase.shape[:3]``.  Strongly
        recommended for in vivo data.
    encoding_axis, halve, polarity
        Passed to :func:`combine_encodings`.
    meg_sensitivity : float or array_like, optional
        If given, the unwrapped phase is converted to physical
        displacement via :func:`phase_to_displacement`.  If ``None``
        (default) the result is returned as unwrapped phase in radians
        (matching the Feng et al. 2025 ``U`` convention).
    unwrap : bool, optional
        Whether to spatially unwrap the combined phase (default
        ``True``).

    Returns
    -------
    numpy.ndarray
        The displacement wavefield: unwrapped phase in radians, or
        physical displacement if ``meg_sensitivity`` was supplied.  The
        encoding axis is halved (one value per temporal offset).
    """
    combined = combine_encodings(
        phase, encoding_axis=encoding_axis, halve=halve, polarity=polarity
    )
    field = unwrap_field(combined, mask=mask) if unwrap else combined
    if meg_sensitivity is not None:
        field = phase_to_displacement(field, meg_sensitivity)
    return field
