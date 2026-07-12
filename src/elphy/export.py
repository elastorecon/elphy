"""Solver-agnostic export of MRE displacement wavefields for inversion.

The inversion step of MRE (recovering stiffness from the measured motion)
lives in separate solver codes, each with its own input convention.  This
module writes the complex displacement wavefield produced by
:mod:`elphy.wavefield` into a small, self-describing *inversion payload*
and provides writers for three concrete container formats:

``.mat``
    MATLAB container compatible with the historical ``mbmre`` inversion
    handoff: per-direction complex motion arrays (``Xmotion``, ``Ymotion``,
    ``Zmotion``), a ``mask``, an optional anatomical ``t2stack``, and an
    ``mreParams`` struct (FOV, matrix size, frequency, quality metric).
``.npz``
    NumPy archive holding the canonical
    ``[x, y, z, direction, frequency]`` wavefield plus its metadata, with
    :func:`read_inversion_npz` for a lossless round trip.
NIfTI
    One file per component (real/imaginary of each direction), with the
    voxel spacing written into the affine, for neuroimaging / BIDS tools.

All writers consume the same payload dict from
:func:`build_inversion_payload`, so a new solver format is one more writer
over an unchanged representation.

Axis / direction convention
----------------------------
The wavefield uses the canonical elphy layout
``field[x, y, z, direction, frequency]`` where ``direction`` indexes
``(u_x, u_y, u_z)`` at ``0, 1, 2``.  The ``.mat`` writer maps
``Xmotion = direction 0``, ``Ymotion = direction 1``,
``Zmotion = direction 2`` by default; pass ``direction_names`` to relabel
if a downstream solver expects a different naming or order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field as _dc_field

import numpy as np

__all__ = [
    "InversionPayload",
    "build_inversion_payload",
    "export_inversion_mat",
    "export_inversion_npz",
    "read_inversion_npz",
    "export_inversion_nifti",
    "export_wavefield",
]


@dataclass
class InversionPayload:
    """Canonical, solver-agnostic representation of an MRE wavefield.

    Attributes
    ----------
    wavefield : numpy.ndarray
        Complex displacement wavefield, layout
        ``[x, y, z, direction, frequency]`` (direction length 3).
    voxel_size_mm : numpy.ndarray
        Spatial spacing ``(dx, dy, dz)`` in millimetres.
    frequencies_hz : numpy.ndarray
        Vibration frequency (Hz) for each entry of the frequency axis.
    mask : numpy.ndarray or None
        Boolean ROI over the three spatial axes.
    magnitude : numpy.ndarray or None
        Anatomical / magnitude reference over the three spatial axes.
    quality : dict
        Optional scalar quality metrics (e.g. ``{"oss_snr": {...}}``).
    name : str
        Short label for the sample.
    metadata : dict
        Free-form provenance (units, source, pipeline parameters).
    """

    wavefield: np.ndarray
    voxel_size_mm: np.ndarray
    frequencies_hz: np.ndarray
    mask: np.ndarray | None = None
    magnitude: np.ndarray | None = None
    quality: dict = _dc_field(default_factory=dict)
    name: str = "elphy_wavefield"
    metadata: dict = _dc_field(default_factory=dict)


def _as_spacing(voxel_size):
    s = np.atleast_1d(np.asarray(voxel_size, dtype=float))
    if s.size == 1:
        s = np.repeat(s, 3)
    if s.size != 3:
        raise ValueError("voxel_size must be a scalar or length-3 sequence")
    if np.any(s <= 0):
        raise ValueError("voxel_size entries must be positive")
    return s


def build_inversion_payload(
    field,
    voxel_size,
    frequencies,
    mask=None,
    magnitude=None,
    quality=None,
    name="elphy_wavefield",
    metadata=None,
):
    """Assemble a canonical :class:`InversionPayload` from a wavefield.

    Parameters
    ----------
    field : array_like
        Complex displacement wavefield.  A 4-D array ``[x, y, z, direction]``
        is treated as a single frequency and promoted to
        ``[x, y, z, direction, 1]``; a 5-D array
        ``[x, y, z, direction, frequency]`` is used as-is.
    voxel_size : float or sequence of float
        Spatial spacing ``(dx, dy, dz)`` in millimetres (scalar broadcasts).
    frequencies : float or sequence of float
        Vibration frequency (Hz) for each frequency-axis entry.  A scalar is
        allowed only for a single-frequency field.
    mask : array_like of bool, optional
        ROI over the three spatial axes.
    magnitude : array_like, optional
        Anatomical / magnitude reference over the three spatial axes.
    quality : dict, optional
        Scalar quality metrics to carry along (e.g. from
        :func:`elphy.strain.oss_snr`).
    name : str, optional
        Short sample label.
    metadata : dict, optional
        Extra provenance fields.

    Returns
    -------
    InversionPayload
    """
    field = np.asarray(field)
    if field.ndim == 4:
        field = field[..., np.newaxis]
    elif field.ndim != 5:
        raise ValueError(
            "field must be [x,y,z,direction] or [x,y,z,direction,frequency]; "
            f"got ndim={field.ndim}"
        )
    if field.shape[3] != 3:
        raise ValueError(f"direction axis (axis 3) must have length 3, got {field.shape[3]}")
    field = field.astype(np.complex128, copy=False)

    spacing = _as_spacing(voxel_size)
    nfreq = field.shape[4]
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
    if freqs.size == 1 and nfreq > 1:
        raise ValueError(f"field has {nfreq} frequencies but a single frequency was given")
    if freqs.size != nfreq:
        raise ValueError(f"frequencies has {freqs.size} entries but field has {nfreq} frequencies")

    spatial = field.shape[:3]
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != spatial:
            raise ValueError(f"mask shape {mask.shape} != spatial shape {spatial}")
    if magnitude is not None:
        magnitude = np.asarray(magnitude, dtype=float)
        if magnitude.shape != spatial:
            raise ValueError(f"magnitude shape {magnitude.shape} != spatial shape {spatial}")

    meta = {
        "wavefield_layout": "[x, y, z, direction, frequency]",
        "direction_order": "0=u_x, 1=u_y, 2=u_z",
        "displacement_units": "micrometre"
        if metadata is None
        else metadata.get("displacement_units", "micrometre"),
        "voxel_units": "mm",
        "frequency_units": "Hz",
        "producer": "elphy.export",
    }
    if metadata:
        meta.update(metadata)

    return InversionPayload(
        wavefield=field,
        voxel_size_mm=spacing,
        frequencies_hz=freqs,
        mask=mask,
        magnitude=magnitude,
        quality=dict(quality or {}),
        name=str(name),
        metadata=meta,
    )


def _select_frequency(payload, frequency_index):
    """Return (wavefield_slice_4d, freq_hz) for one frequency index."""
    if payload.wavefield.shape[4] == 1:
        fi = 0
    elif frequency_index is None:
        raise ValueError("payload has multiple frequencies; pass frequency_index to select one")
    else:
        fi = int(frequency_index)
    return payload.wavefield[:, :, :, :, fi], float(payload.frequencies_hz[fi])


def export_inversion_mat(path, payload, frequency_index=None, direction_names=None):
    """Write a MATLAB ``.mat`` inversion handoff for one frequency.

    Produces the historical ``mbmre`` layout: complex per-direction motion
    arrays, a ``mask``, an optional anatomical ``t2stack``, and an
    ``mreParams`` struct with FOV, matrix size, frequency, and any scalar
    quality metrics.

    Parameters
    ----------
    path : str or path-like
        Output ``.mat`` path.
    payload : InversionPayload
    frequency_index : int, optional
        Which frequency-axis entry to write (required if the payload holds
        more than one frequency).
    direction_names : sequence of str, optional
        Names for the three motion arrays in direction order ``(x, y, z)``.
        Default ``("Xmotion", "Ymotion", "Zmotion")``.
    """
    from scipy.io import savemat

    wf, freq = _select_frequency(payload, frequency_index)
    names = direction_names or ("Xmotion", "Ymotion", "Zmotion")
    if len(names) != 3:
        raise ValueError("direction_names must have exactly 3 entries")

    ny, nx, nz = wf.shape[:3]
    dx, dy, dz = payload.voxel_size_mm
    mreParams = {
        "subj": payload.name,
        "FOVx": float(nx * dx),
        "FOVy": float(ny * dy),
        "FOVz": float(nz * dz),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "freq": freq,
        "voxel_size_mm": payload.voxel_size_mm.astype(float),
    }
    for key, val in payload.quality.items():
        if np.isscalar(val):
            mreParams[key] = float(val)

    out = {"mreParams": mreParams}
    for i, nm in enumerate(names):
        out[nm] = wf[:, :, :, i]
    if payload.mask is not None:
        out["mask"] = payload.mask.astype(np.float64)
    if payload.magnitude is not None:
        out["t2stack"] = payload.magnitude.astype(np.float64)
    savemat(str(path), out, do_compression=True)


def export_inversion_npz(path, payload):
    """Write the full canonical payload to a compressed ``.npz`` archive.

    Stores the complete ``[x, y, z, direction, frequency]`` wavefield and
    all metadata; :func:`read_inversion_npz` reconstructs the payload
    losslessly.
    """
    arrays = {
        "wavefield": payload.wavefield,
        "voxel_size_mm": payload.voxel_size_mm,
        "frequencies_hz": payload.frequencies_hz,
        "name": np.array(payload.name),
        "metadata_json": np.array(json.dumps(payload.metadata)),
        "quality_json": np.array(json.dumps(_jsonable(payload.quality))),
    }
    if payload.mask is not None:
        arrays["mask"] = payload.mask
    if payload.magnitude is not None:
        arrays["magnitude"] = payload.magnitude
    np.savez_compressed(str(path), **arrays)


def _jsonable(obj):
    """Recursively convert numpy scalars/arrays for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def read_inversion_npz(path):
    """Reconstruct an :class:`InversionPayload` written by
    :func:`export_inversion_npz`."""
    with np.load(str(path), allow_pickle=False) as z:
        return InversionPayload(
            wavefield=z["wavefield"],
            voxel_size_mm=z["voxel_size_mm"],
            frequencies_hz=z["frequencies_hz"],
            mask=z["mask"] if "mask" in z.files else None,
            magnitude=z["magnitude"] if "magnitude" in z.files else None,
            quality=json.loads(str(z["quality_json"])) if "quality_json" in z.files else {},
            name=str(z["name"]) if "name" in z.files else "elphy_wavefield",
            metadata=json.loads(str(z["metadata_json"])) if "metadata_json" in z.files else {},
        )


def _affine(spacing):
    aff = np.eye(4)
    aff[0, 0], aff[1, 1], aff[2, 2] = (float(s) for s in spacing)
    return aff


def export_inversion_nifti(directory, payload, frequency_index=None, prefix=None):
    """Write per-component NIfTI volumes for one frequency.

    Splits the complex wavefield into real and imaginary parts for each of
    the three directions and writes six ``.nii.gz`` volumes (plus the mask
    and magnitude if present), with the voxel spacing in each affine.
    Requires :mod:`nibabel`.

    Returns
    -------
    list of str
        The paths written.
    """
    import os

    import nibabel as nib

    wf, freq = _select_frequency(payload, frequency_index)
    pre = prefix or payload.name
    aff = _affine(payload.voxel_size_mm)
    os.makedirs(directory, exist_ok=True)
    written = []
    comp = ("x", "y", "z")
    for i, c in enumerate(comp):
        for part, arr in (("real", wf[:, :, :, i].real), ("imag", wf[:, :, :, i].imag)):
            p = os.path.join(directory, f"{pre}_u{c}_{part}.nii.gz")
            nib.save(nib.Nifti1Image(np.asarray(arr, dtype=np.float32), aff), p)
            written.append(p)
    if payload.mask is not None:
        p = os.path.join(directory, f"{pre}_mask.nii.gz")
        nib.save(nib.Nifti1Image(payload.mask.astype(np.uint8), aff), p)
        written.append(p)
    if payload.magnitude is not None:
        p = os.path.join(directory, f"{pre}_magnitude.nii.gz")
        nib.save(nib.Nifti1Image(np.asarray(payload.magnitude, dtype=np.float32), aff), p)
        written.append(p)
    return written


def export_wavefield(path, payload, frequency_index=None, **kwargs):
    """Dispatch to a writer by the extension of ``path``.

    ``.mat`` -> :func:`export_inversion_mat`, ``.npz`` ->
    :func:`export_inversion_npz`, ``.nii``/``.nii.gz`` (or a directory) ->
    :func:`export_inversion_nifti`.
    """
    p = str(path)
    low = p.lower()
    if low.endswith(".mat"):
        return export_inversion_mat(p, payload, frequency_index=frequency_index, **kwargs)
    if low.endswith(".npz"):
        return export_inversion_npz(p, payload)
    if low.endswith((".nii", ".nii.gz")) or "." not in p.split("/")[-1]:
        return export_inversion_nifti(p, payload, frequency_index=frequency_index, **kwargs)
    raise ValueError(f"cannot infer export format from path: {path}")
