"""Input/output helpers for MRE data.

Loaders that bring reconstructed MRE data into the canonical in-memory
layout used throughout :mod:`elphy` (see :mod:`elphy.wavefield`).

Currently supported:

* MATLAB ``.mat`` files following the layout of the public MRE dataset
  of Feng et al., *Scientific Data* 12:1680 (2025)
  (https://doi.org/10.1038/s41597-025-05968-9), Science Data Bank
  https://www.scidb.cn/en/s/eYFnie .  Both the classic (v7, scipy) and
  HDF5-based (v7.3) ``.mat`` formats are handled transparently.
* The historical ``mbmre`` inversion handoff (``mre_toolbox``'s
  ``proc_mbmre_*.m``): see :func:`load_mbmre_for_inversion`, the
  counterpart to :func:`elphy.export.export_inversion_mat`.
* The NLI (nonlinear inversion) material-property reconstruction handoff
  (``mre_toolbox``'s ``ReconProps.mat``): see
  :func:`load_nli_reconprops`, the counterpart to
  :func:`elphy.export.export_nli_nifti`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "FENG2025_FREQUENCIES_HZ",
    "FENG2025_MEG_SENSITIVITY_RAD_PER_MM",
    "load_mat_array",
    "load_mbmre_for_inversion",
    "load_nli_reconprops",
    "MREWavefield",
    "load_scidb_feng2025",
]

#: Vibration frequencies (Hz) of the Feng et al. 2025 dataset, in the
#: order of the frequency axis of the ``.mat`` arrays.
FENG2025_FREQUENCIES_HZ = (30.0, 40.0, 50.0, 60.0)

#: Motion-encoding efficiencies (rad/mm) reported per vibration
#: frequency in the Feng et al. 2025 data descriptor, keyed by
#: frequency in Hz.  Use these as the ``meg_sensitivity`` when converting
#: the dataset's phase to physical displacement.
FENG2025_MEG_SENSITIVITY_RAD_PER_MM = {
    30.0: 1.30e5,
    40.0: 1.70e5,
    50.0: 1.36e5,
    60.0: 1.14e5,
}


def load_mat_array(path, var=None):
    """Load a single array from a MATLAB ``.mat`` file.

    Transparently handles both the classic ``.mat`` format (loaded via
    :func:`scipy.io.loadmat`) and the HDF5-based v7.3 format (loaded via
    :mod:`h5py`, with the MATLAB column-major axis order transposed back
    to match the scipy convention).

    Parameters
    ----------
    path : str or path-like
        Path to the ``.mat`` file.
    var : str, optional
        Name of the variable to load.  If ``None`` (default) the single
        non-metadata variable in the file is returned; a ``ValueError``
        is raised if the file contains more than one candidate.

    Returns
    -------
    numpy.ndarray
        The loaded array.

    Raises
    ------
    ValueError
        If ``var`` is ``None`` and the file does not contain exactly one
        candidate variable, or if ``var`` is not found.
    """
    import scipy.io as sio

    try:
        mat = sio.loadmat(path)
        keys = [k for k in mat if not k.startswith("__")]
        if var is None:
            if len(keys) != 1:
                raise ValueError(
                    f"expected exactly one variable in {path}, found {keys}; "
                    "pass var= to disambiguate"
                )
            var = keys[0]
        if var not in mat:
            raise ValueError(f"variable {var!r} not found in {path}; have {keys}")
        return np.asarray(mat[var])
    except NotImplementedError:
        # v7.3 (HDF5) .mat file
        import h5py

        with h5py.File(path, "r") as f:
            keys = [k for k in f.keys() if not k.startswith("#")]
            if var is None:
                if len(keys) != 1:
                    raise ValueError(
                        f"expected exactly one variable in {path}, found {keys}; "
                        "pass var= to disambiguate"
                    ) from None
                var = keys[0]
            if var not in f:
                raise ValueError(f"variable {var!r} not found in {path}; have {keys}") from None
            # h5py returns MATLAB arrays with reversed axis order.
            return np.asarray(f[var]).T


def _mbmre_axes_to_nifti(arr):
    """Apply the ``mre_toolbox`` MATLAB-array -> NIfTI axis convention.

    ``proc_mbmre_lr_1.m`` documents this exactly: "the data is changed
    into NIFTI via ``flipdim(flipdim(permute(image,[2 1 3 ...]),1),2)``".
    For a 3-D spatial array (the case here - ``t2stack``, ``mask``, and
    each direction's motion field are all ``[ny, nx, nz]``) that is: swap
    the first two axes, then flip the (new) first two axes.
    """
    arr = np.swapaxes(np.asarray(arr), 0, 1)
    return np.flip(arr, axis=(0, 1))


def load_mbmre_for_inversion(path, direction_names=("Xmotion", "Ymotion", "Zmotion")):
    """Load a historical ``mbmre_for_inversion.mat`` handoff.

    Reads the ``.mat`` file written by ``mre_toolbox``'s
    ``proc_mbmre_2.m`` (variables ``mreParams``, ``mask``, ``t2stack``,
    and per-direction complex motion arrays) and returns the equivalent
    :class:`elphy.export.InversionPayload`, ready for
    :func:`elphy.export.export_inversion_nifti` or any other writer in
    :mod:`elphy.export`.

    The arrays in the ``.mat`` file are in MATLAB's raw ``[ny, nx, nz]``
    acquisition order, *not* the flipped/permuted order used when
    ``mre_toolbox`` itself writes NIfTI (see ``t2stack.nii``,
    ``maskx.nii``). This loader applies that same axis convention (via
    :func:`_mbmre_axes_to_nifti`) to every spatial array - ``t2stack``,
    ``mask``, and each of ``Xmotion``/``Ymotion``/``Zmotion`` - so the
    returned payload, and anything derived from it (e.g. displacement
    NIfTIs registered against DTI vectors), lands in the exact same voxel
    orientation as the existing ``t2stack.nii``/``maskx.nii``.

    Parameters
    ----------
    path : str or path-like
        Path to the ``mbmre_for_inversion.mat`` file.
    direction_names : sequence of str, optional
        Names of the three complex motion arrays in the ``.mat`` file, in
        ``(x, y, z)`` order. Default ``("Xmotion", "Ymotion", "Zmotion")``,
        matching ``proc_mbmre_2.m``.

    Returns
    -------
    elphy.export.InversionPayload
    """
    import scipy.io as sio

    from .export import build_inversion_payload

    mat = sio.loadmat(str(path))
    mp = mat["mreParams"][0, 0]

    def _scalar(name):
        return mp[name].item()

    t2stack = _mbmre_axes_to_nifti(mat["t2stack"])
    mask = _mbmre_axes_to_nifti(mat["mask"]) != 0
    motion = [_mbmre_axes_to_nifti(mat[name]) for name in direction_names]
    field = np.stack(motion, axis=-1)  # [x, y, z, direction]

    # Derived from the raw [ny, nx, nz] array shape (not mreParams.nx/ny,
    # which mre_toolbox sets identically to ny - a latent bug in the
    # MATLAB source that happens to be harmless only when nx == ny).
    ny_raw, nx_raw, nz_raw = mat["t2stack"].shape
    dy = _scalar("FOVy") / ny_raw
    dx = _scalar("FOVx") / nx_raw
    dz = _scalar("FOVz") / nz_raw

    quality = {}
    if "oss_snr" in mp.dtype.names:
        quality["oss_snr"] = _scalar("oss_snr")

    return build_inversion_payload(
        field,
        voxel_size=(dx, dy, dz),
        frequencies=_scalar("freq"),
        mask=mask,
        magnitude=t2stack,
        quality=quality,
        name=str(_scalar("subj")),
        metadata={"source": "mbmre_toolbox", "source_path": str(path)},
    )


def _mat_struct_to_dict(struct_arr):
    """Flatten a scipy ``loadmat`` 1x1 struct array into a plain dict.

    Scalar fields become Python scalars (``str`` for MATLAB char arrays);
    non-scalar fields become nested lists, so the result is JSON-safe.
    """
    s = struct_arr[0, 0]
    out = {}
    for field_name in s.dtype.names:
        value = np.asarray(s[field_name]).squeeze()
        out[field_name] = value.item() if value.ndim == 0 else value.tolist()
    return out


def _find_nli_interp_data(path):
    """Locate the ``*.InterpData.mat`` sibling of an NLI ``ReconProps.mat``.

    ``InterpData.mat`` lives next to the mesh-building files, one or two
    directories above the ``inv_cascade`` directory that holds
    ``ReconProps.mat`` (its filename is derived from the mesh basename, not
    the recon basename, so it can't be located by string substitution). Both
    the immediate parent directory and its parent are checked, to handle
    either a direct ``inv_cascade/*.ReconProps.mat`` path or the convenience
    symlink ``mre_toolbox`` leaves one level up.

    Returns
    -------
    str or None
        The matched path, or ``None`` if zero or more than one
        ``*.InterpData.mat`` file was found in either directory.
    """
    import glob
    import os

    d = os.path.dirname(os.path.abspath(str(path)))
    for _ in range(2):
        hits = glob.glob(os.path.join(d, "*.InterpData.mat"))
        if len(hits) == 1:
            return hits[0]
        if hits:
            return None
        d = os.path.dirname(d)
    return None


def load_nli_reconprops(path):
    """Load an NLI (nonlinear inversion) ``*.ReconProps.mat`` handoff.

    Reads the ``.mat`` file written at the end of an NLI material-property
    reconstruction cascade (``mre_toolbox``'s voxelmesh pipeline): per-voxel
    real/imaginary shear modulus, bulk modulus, and density, a damping-ratio
    map, a Rayleigh-composition classification, a structural ``MagIm``
    reference, and reconstruction metadata (``invparam``, ``meshinputs``,
    ``recondir``, ``meshres``, ``meshind``). Returns the equivalent
    :class:`elphy.export.MaterialPropertyMap`, ready for
    :func:`elphy.export.export_nli_nifti`.

    The property maps (``DR``, ``RC``, ``RealShear``/``ImagShear``,
    ``RealBulk``/``ImagBulk``, ``RealDens``/``ImagDens``) and ``MagIm`` are,
    like ``mbmre_for_inversion.mat``'s arrays, in MATLAB's raw ``[ny, nx,
    nz]`` acquisition order rather than NIfTI order, and share that same
    raw order with the mesh-building input (confirmed by their near-perfect
    spatial overlap with the raw, untransformed ``mbmre_for_inversion.mat``
    mask). This loader therefore applies the identical axis convention as
    :func:`load_mbmre_for_inversion` (via :func:`_mbmre_axes_to_nifti`) to
    every spatial array, landing the property maps in the same voxel
    orientation as the existing ``maskx.nii``/``t2stack.nii`` and any
    externally-registered reference (e.g. a T2/FLAIR resampled into the MRE
    grid).

    The voxel spacing of the property maps is taken from ``meshres``'s
    first row - the finest ("voxelmesh") mesh resolution, which matches the
    original MRE acquisition's voxel size (confirmed against the paired
    ``mbmre_for_inversion.mat``'s ``mreParams`` FOV/matrix size for this
    sample).

    ``MagIm`` (inside ``ReconProps.mat``) commonly sits on a smaller,
    cropped grid (the estimation domain) with no origin offset available to
    register it against the property maps. This loader therefore looks for
    a ``*.InterpData.mat`` file next to the mesh-building inputs (one or two
    directories above ``ReconProps.mat``; see :func:`_find_nli_interp_data`)
    and, if exactly one is found, uses its ``MagIm_int`` array instead - it
    sits on the exact same full grid as the property maps (confirmed via
    its ``maskint``, which is bit-identical to the raw ``mbmre_for_inversion
    .mat`` mask), so no separate spacing or translation is needed. If no
    unique ``InterpData.mat`` is found, falls back to ``ReconProps.mat``'s
    own (cropped) ``MagIm`` with a warning.

    Parameters
    ----------
    path : str or path-like
        Path to the ``*.ReconProps.mat`` file.

    Returns
    -------
    elphy.export.MaterialPropertyMap
    """
    import os
    import warnings

    import scipy.io as sio

    from .export import MaterialPropertyMap, _as_spacing

    mat = sio.loadmat(str(path))

    def _prop(name):
        return _mbmre_axes_to_nifti(np.asarray(mat[name], dtype=np.float64))

    meshres_m = np.asarray(mat["meshres"], dtype=float)
    voxel_size_mm = _as_spacing(meshres_m[0] * 1000.0)

    shear = _prop("RealShear") + 1j * _prop("ImagShear")
    bulk = _prop("RealBulk") + 1j * _prop("ImagBulk")
    density = _prop("RealDens") + 1j * _prop("ImagDens")
    damping_ratio = _prop("DR")
    rc = _prop("RC")

    interp_data_path = _find_nli_interp_data(path)
    if interp_data_path is not None:
        interp = sio.loadmat(interp_data_path)
        magnitude = _mbmre_axes_to_nifti(np.asarray(interp["MagIm_int"], dtype=np.float64))
        magnitude_voxel_size_mm = voxel_size_mm.copy()
    else:
        warnings.warn(
            f"no unique *.InterpData.mat found near {path!r}; falling back to "
            "ReconProps.mat's own (cropped, unregistered) MagIm as the magnitude reference",
            stacklevel=2,
        )
        magnitude = _mbmre_axes_to_nifti(np.asarray(mat["MagIm"], dtype=np.float64))
        magnitude_voxel_size_mm = voxel_size_mm.copy()

    metadata = {
        "source": "nli_reconprops",
        "source_path": str(path),
        "interp_data_path": interp_data_path,
        "recondir": str(np.asarray(mat["recondir"]).ravel()[0]),
        "meshres_m": meshres_m.tolist(),
        "meshind": np.asarray(mat["meshind"]).tolist(),
        "invparam": _mat_struct_to_dict(mat["invparam"]),
        "meshinputs": _mat_struct_to_dict(mat["meshinputs"]),
    }

    return MaterialPropertyMap(
        shear_modulus_pa=shear,
        bulk_modulus_pa=bulk,
        density_kg_m3=density,
        damping_ratio=damping_ratio,
        rayleigh_composition=rc,
        voxel_size_mm=voxel_size_mm,
        magnitude=magnitude,
        magnitude_voxel_size_mm=magnitude_voxel_size_mm,
        name=os.path.splitext(os.path.basename(str(path)))[0],
        metadata=metadata,
    )


@dataclass
class MREWavefield:
    """Container for a loaded MRE acquisition.

    Attributes
    ----------
    phase_raw : numpy.ndarray or None
        Raw motion-encoded phase, layout
        ``[x, y, z, encoding, direction, frequency]`` (radians), with
        interleaved positive/negative polarity samples along the
        encoding axis.
    mag : numpy.ndarray or None
        Magnitude images, same layout as ``phase_raw``.
    displacement : numpy.ndarray or None
        Provided (pre-processed) displacement/wavefield, layout
        ``[x, y, z, offset, direction, frequency]``.
    modulus : numpy.ndarray or None
        Complex shear modulus map ``[x, y, z]`` in Pa (real = storage,
        imag = loss), when available.
    frequencies_hz : tuple of float or None
        Vibration frequency (Hz) for each entry of the frequency axis.
    meg_sensitivity_rad_per_mm : dict or None
        MEG sensitivity (rad/mm) keyed by frequency (Hz).
    name : str
        Optional label (e.g. ``"phantom"``, ``"brain"``, ``"liver"``).
    """

    phase_raw: np.ndarray | None = None
    mag: np.ndarray | None = None
    displacement: np.ndarray | None = None
    modulus: np.ndarray | None = None
    frequencies_hz: tuple | None = None
    meg_sensitivity_rad_per_mm: dict | None = None
    name: str = ""

    @property
    def shape(self):
        """Spatial+encoding shape of the raw phase array (or None)."""
        return None if self.phase_raw is None else self.phase_raw.shape

    def sensitivity_array(self):
        """MEG sensitivity broadcastable against the frequency axis.

        Returns
        -------
        numpy.ndarray
            Shape ``(1, 1, 1, 1, 1, n_freq)`` array of sensitivities
            (rad/mm), aligned with the frequency axis of ``phase_raw``,
            for direct use as ``meg_sensitivity`` in
            :func:`elphy.wavefield.phase_to_displacement`.

        Raises
        ------
        ValueError
            If frequency/sensitivity metadata is missing.
        """
        if not self.frequencies_hz or not self.meg_sensitivity_rad_per_mm:
            raise ValueError("frequency / sensitivity metadata not available")
        vals = np.array(
            [self.meg_sensitivity_rad_per_mm[f] for f in self.frequencies_hz],
            dtype=float,
        )
        return vals.reshape(1, 1, 1, 1, 1, -1)


def load_scidb_feng2025(directory, name="", load=("PhaseRaw", "Mag", "U", "G")):
    """Load one sample of the Feng et al. 2025 public MRE dataset.

    The dataset stores each sample (``phantom``, ``brain``, ``liver``) as
    a directory of ``.mat`` files named ``<Kind>_<sample>.mat`` where
    ``<Kind>`` is one of ``PhaseRaw``, ``Mag``, ``U``, ``G``.

    Parameters
    ----------
    directory : str or path-like
        Path to the sample directory (e.g. ``.../V4/phantom``).
    name : str, optional
        Sample label.  If empty, inferred from the directory basename.
    load : sequence of str, optional
        Which arrays to load (default all four).

    Returns
    -------
    MREWavefield
        Populated container with dataset frequency/sensitivity metadata
        attached.
    """
    import glob
    import os

    directory = os.fspath(directory)
    if not name:
        name = os.path.basename(os.path.normpath(directory))

    def _find(kind):
        # Prefer <Kind>_<name>.mat, else any <Kind>_*.mat in the dir.
        cand = os.path.join(directory, f"{kind}_{name}.mat")
        if os.path.exists(cand):
            return cand
        hits = sorted(glob.glob(os.path.join(directory, f"{kind}_*.mat")))
        return hits[0] if hits else None

    kwargs = {}
    kind_to_attr = {
        "PhaseRaw": "phase_raw",
        "Mag": "mag",
        "U": "displacement",
        "G": "modulus",
    }
    for kind in load:
        path = _find(kind)
        if path is not None:
            kwargs[kind_to_attr[kind]] = load_mat_array(path)

    return MREWavefield(
        frequencies_hz=FENG2025_FREQUENCIES_HZ,
        meg_sensitivity_rad_per_mm=dict(FENG2025_MEG_SENSITIVITY_RAD_PER_MM),
        name=name,
        **kwargs,
    )
