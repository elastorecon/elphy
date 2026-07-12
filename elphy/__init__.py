# -*- coding: utf-8 -*-

"""Top-level package for MR Elastography (elphy).

``elphy`` provides the general, scanner-agnostic building blocks for
magnetic resonance elastography (MRE) processing: loading reconstructed
images, combining motion-encoding polarities, phase unwrapping, and
converting motion-encoded phase into a complex displacement wavefield for
downstream material-property inversion.
"""

__author__ = """Aaron T Anderson"""
__email__ = 'aandrsn3@illinois.edu'
__version__ = '0.1.0'

# Core wavefield operations (numpy-only; always importable).
from .wavefield import (
    wrap_to_pi,
    combine_encodings,
    unwrap_field,
    phase_to_displacement,
    meg_sensitivity_muthupillai,
    raw_phase_to_wavefield,
)

# I/O helpers (scipy / h5py imported lazily inside the functions).
from .io import (
    load_mat_array,
    load_scidb_feng2025,
    MREWavefield,
    FENG2025_FREQUENCIES_HZ,
    FENG2025_MEG_SENSITIVITY_RAD_PER_MM,
)

__all__ = [
    "wrap_to_pi",
    "combine_encodings",
    "unwrap_field",
    "phase_to_displacement",
    "meg_sensitivity_muthupillai",
    "raw_phase_to_wavefield",
    "load_mat_array",
    "load_scidb_feng2025",
    "MREWavefield",
    "FENG2025_FREQUENCIES_HZ",
    "FENG2025_MEG_SENSITIVITY_RAD_PER_MM",
]


