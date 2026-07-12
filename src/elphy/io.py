"""Input/output helpers for MRE data.

Loaders that bring reconstructed MRE data into the canonical in-memory
layout used throughout :mod:`elphy` (see :mod:`elphy.wavefield`).

Currently supported:

* MATLAB ``.mat`` files following the layout of the public MRE dataset
  of Feng et al., *Scientific Data* 12:1680 (2025)
  (https://doi.org/10.1038/s41597-025-05968-9), Science Data Bank
  https://www.scidb.cn/en/s/eYFnie .  Both the classic (v7, scipy) and
  HDF5-based (v7.3) ``.mat`` formats are handled transparently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "FENG2025_FREQUENCIES_HZ",
    "FENG2025_MEG_SENSITIVITY_RAD_PER_MM",
    "load_mat_array",
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
