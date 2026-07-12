"""Wavefield operators for MRE displacement fields.

This module contains general, scanner-agnostic operations applied to a
complex MRE displacement wavefield *after* the temporal harmonic has been
extracted (see :func:`elphy.wavefield.temporal_harmonic`).  These are the
standard pre-inversion steps that condition the wavefield and separate the
shear component that carries the elastic information.

Two families of operator are provided:

* **k-space band-pass filtering** (:func:`bandpass_filter`) -- a radial
  Butterworth filter that suppresses spatial frequencies outside the band
  where the shear wave lives.  The high-pass edge removes slowly varying
  bulk/rigid-body motion and residual background; the low-pass edge
  removes high-spatial-frequency noise.  This ports and generalises the
  MATLAB ``mre_filter`` (C. L. Johnson, 2011), which applied a 4th-order
  radial Butterworth band-pass slice by slice.

* **vector differential operators** (:func:`curl`, :func:`divergence`).
  The measured displacement is a superposition of a shear (transverse,
  divergence-free) wave and a compressional (longitudinal, curl-free)
  wave.  Taking the curl removes the compressional part and any bulk
  motion, isolating the shear wavefield used by curl-based inversions;
  the divergence isolates the compressional part.

Axis convention
---------------
The complex vector wavefield is laid out as::

    field[x, y, z, direction, frequency]

* ``x, y, z``     -- spatial voxel indices.
* ``direction``   -- the three Cartesian components of the displacement
                     vector, ordered ``(u_x, u_y, u_z)``.
* ``frequency``   -- vibration frequency (optional trailing axis).

The differential operators require the three spatial axes and the
direction axis; extra trailing axes (e.g. frequency) are looped over
transparently.  The band-pass filter operates on whatever spatial axes
are named and passes all other axes through unchanged, so it works on a
scalar field, a vector field, or a full multi-frequency stack.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "bandpass_filter",
    "curl",
    "divergence",
]


def _normalise_spacing(voxel_size, ndim):
    """Return a length-``ndim`` tuple of positive float spacings."""
    spacing = np.atleast_1d(np.asarray(voxel_size, dtype=float))
    if spacing.size == 1:
        spacing = np.repeat(spacing, ndim)
    if spacing.size != ndim:
        raise ValueError(
            f"voxel_size must be a scalar or length-{ndim} sequence, got {spacing.size} values"
        )
    if np.any(spacing <= 0):
        raise ValueError("voxel_size entries must be positive")
    return tuple(float(s) for s in spacing)


def bandpass_filter(
    field,
    voxel_size,
    low_cutoff=None,
    high_cutoff=None,
    order=4,
    spatial_axes=(0, 1, 2),
):
    """Apply a radial Butterworth band-pass filter in k-space.

    The passband is the annulus ``low_cutoff < |k| < high_cutoff`` in
    spatial frequency, where ``|k|`` is the radial wavenumber in cycles
    per unit length (the reciprocal of ``voxel_size``'s unit).  The
    filter is the product of a Butterworth high-pass and low-pass::

        H_high(k) = 1 / (1 + (low_cutoff  / |k|)**order)
        H_low(k)  = 1 / (1 + (|k| / high_cutoff)**order)

    This matches the MATLAB ``mre_filter`` reference (default
    ``order=4``).  Either edge may be omitted to obtain a pure low- or
    high-pass filter.

    Parameters
    ----------
    field : array_like
        Real or complex field.  The transform is applied over
        ``spatial_axes``; all other axes are broadcast/looped over.
    voxel_size : float or sequence of float
        Voxel spacing along each spatial axis, in a consistent length
        unit (e.g. mm).  A scalar is broadcast to all spatial axes.
        Cutoffs must be given in cycles per that same unit.
    low_cutoff : float, optional
        Lower passband edge (high-pass cutoff) in cycles per unit
        length.  ``None`` (default) disables the high-pass, keeping the
        DC/low-frequency content.
    high_cutoff : float, optional
        Upper passband edge (low-pass cutoff) in cycles per unit length.
        ``None`` (default) disables the low-pass.
    order : float, optional
        Butterworth exponent controlling roll-off sharpness
        (default ``4``, matching the reference).
    spatial_axes : tuple of int, optional
        Axes over which to filter (default ``(0, 1, 2)``).  Use
        ``(0, 1)`` to reproduce the slice-by-slice 2-D behaviour of the
        original ``mre_filter``.

    Returns
    -------
    numpy.ndarray
        The filtered field, same shape as ``field``.  Complex input
        stays complex; a real input is returned real (the imaginary part
        introduced by the symmetric filter is negligible and discarded).

    Raises
    ------
    ValueError
        If both cutoffs are ``None``, if ``low_cutoff >= high_cutoff``,
        or if ``voxel_size`` has the wrong length.
    """
    if low_cutoff is None and high_cutoff is None:
        raise ValueError("at least one of low_cutoff, high_cutoff must be given")
    if low_cutoff is not None and high_cutoff is not None and low_cutoff >= high_cutoff:
        raise ValueError(f"low_cutoff ({low_cutoff}) must be < high_cutoff ({high_cutoff})")

    field = np.asarray(field)
    was_real = not np.iscomplexobj(field)
    spatial_axes = tuple(int(a) for a in spatial_axes)
    spacing = _normalise_spacing(voxel_size, len(spatial_axes))

    # Build the radial wavenumber grid over the spatial axes only,
    # broadcast against the full array shape.
    ksq = np.zeros([field.shape[a] for a in spatial_axes], dtype=float)
    grids = np.meshgrid(
        *[np.fft.fftfreq(field.shape[a], d=s) for a, s in zip(spatial_axes, spacing, strict=True)],
        indexing="ij",
    )
    for g in grids:
        ksq = ksq + g**2
    kmag = np.sqrt(ksq)

    transfer = np.ones_like(kmag)
    if low_cutoff is not None:
        # High-pass: 0 at DC (kmag == 0), -> 1 well above low_cutoff.
        with np.errstate(divide="ignore"):
            hp = 1.0 / (1.0 + (low_cutoff / kmag) ** order)
        hp[kmag == 0] = 0.0
        transfer = transfer * hp
    if high_cutoff is not None:
        lp = 1.0 / (1.0 + (kmag / high_cutoff) ** order)
        transfer = transfer * lp

    # Reshape the transfer function to broadcast against `field`.
    shape = [1] * field.ndim
    for a in spatial_axes:
        shape[a] = field.shape[a]
    transfer = transfer.reshape(shape)

    spectrum = np.fft.fftn(field, axes=spatial_axes)
    filtered = np.fft.ifftn(spectrum * transfer, axes=spatial_axes)
    return filtered.real if was_real else filtered


def curl(field, voxel_size, direction_axis=3, spatial_axes=(0, 1, 2)):
    r"""Curl of a 3-component vector wavefield.

    Computes :math:`\nabla \times \mathbf{u}` with central finite
    differences::

        (curl u)_x = du_z/dy - du_y/dz
        (curl u)_y = du_x/dz - du_z/dx
        (curl u)_z = du_y/dx - du_x/dy

    Taking the curl of the MRE displacement field removes the
    compressional (longitudinal, curl-free) component and any bulk
    rigid-body motion, isolating the shear wavefield.

    Parameters
    ----------
    field : array_like
        Real or complex vector field with a length-3 ``direction_axis``
        holding ``(u_x, u_y, u_z)`` and three spatial axes.
    voxel_size : float or sequence of float
        Spacing along the three spatial axes (scalar broadcasts).  Sets
        the physical gradient scale.
    direction_axis : int, optional
        Axis holding the three vector components (default ``3``).
    spatial_axes : tuple of int, optional
        The three spatial axes, ordered ``(x, y, z)``
        (default ``(0, 1, 2)``).

    Returns
    -------
    numpy.ndarray
        The curl, same shape as ``field`` (a 3-vector along
        ``direction_axis``), same dtype family as the input.

    Raises
    ------
    ValueError
        If ``direction_axis`` does not have length 3 or ``spatial_axes``
        is not length 3.
    """
    field = np.asarray(field)
    if len(spatial_axes) != 3:
        raise ValueError("curl requires exactly 3 spatial axes")
    if field.shape[direction_axis] != 3:
        raise ValueError(
            f"direction axis {direction_axis} must have length 3, "
            f"got {field.shape[direction_axis]}"
        )
    ax, ay, az = (int(a) for a in spatial_axes)
    dx, dy, dz = _normalise_spacing(voxel_size, 3)

    def comp(i):
        return np.take(field, i, axis=direction_axis)

    ux, uy, uz = comp(0), comp(1), comp(2)

    # Spatial-axis positions shift down by one after the take() removes
    # direction_axis if it precedes them; recompute gradient axes.
    def gaxis(a):
        return a - 1 if a > direction_axis else a

    def d(arr, a, spacing):
        return np.gradient(arr, spacing, axis=gaxis(a))

    cx = d(uz, ay, dy) - d(uy, az, dz)
    cy = d(ux, az, dz) - d(uz, ax, dx)
    cz = d(uy, ax, dx) - d(ux, ay, dy)
    return np.stack([cx, cy, cz], axis=direction_axis)


def divergence(field, voxel_size, direction_axis=3, spatial_axes=(0, 1, 2)):
    r"""Divergence of a 3-component vector wavefield.

    Computes :math:`\nabla \cdot \mathbf{u} = du_x/dx + du_y/dy +
    du_z/dz` with central finite differences.  The divergence isolates
    the compressional (longitudinal) component of the wavefield and is
    near zero for a purely shear (transverse) wave.

    Parameters
    ----------
    field : array_like
        Real or complex vector field, as for :func:`curl`.
    voxel_size : float or sequence of float
        Spacing along the three spatial axes (scalar broadcasts).
    direction_axis : int, optional
        Axis holding the three vector components (default ``3``).
    spatial_axes : tuple of int, optional
        The three spatial axes, ordered ``(x, y, z)``
        (default ``(0, 1, 2)``).

    Returns
    -------
    numpy.ndarray
        The scalar divergence with ``direction_axis`` removed.

    Raises
    ------
    ValueError
        If ``direction_axis`` does not have length 3 or ``spatial_axes``
        is not length 3.
    """
    field = np.asarray(field)
    if len(spatial_axes) != 3:
        raise ValueError("divergence requires exactly 3 spatial axes")
    if field.shape[direction_axis] != 3:
        raise ValueError(
            f"direction axis {direction_axis} must have length 3, "
            f"got {field.shape[direction_axis]}"
        )
    ax, ay, az = (int(a) for a in spatial_axes)
    dx, dy, dz = _normalise_spacing(voxel_size, 3)

    def comp(i):
        return np.take(field, i, axis=direction_axis)

    def gaxis(a):
        return a - 1 if a > direction_axis else a

    ux, uy, uz = comp(0), comp(1), comp(2)
    return (
        np.gradient(ux, dx, axis=gaxis(ax))
        + np.gradient(uy, dy, axis=gaxis(ay))
        + np.gradient(uz, dz, axis=gaxis(az))
    )
