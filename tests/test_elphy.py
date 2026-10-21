#!/usr/bin/env python

"""Package-level tests for `elphy`.

Physics/algorithm tests live in ``test_wavefield.py``; this module just
checks that the public API is importable and wired up correctly.
"""

import elphy


def test_version():
    assert isinstance(elphy.__version__, str)


def test_public_api_exposed():
    """The core wavefield + io names are importable from the top level."""
    for name in [
        "wrap_to_pi",
        "combine_encodings",
        "unwrap_field",
        "phase_to_displacement",
        "meg_sensitivity_muthupillai",
        "raw_phase_to_wavefield",
        "load_mat_array",
        "load_mbmre_for_inversion",
        "load_nli_reconprops",
        "load_scidb_feng2025",
        "MREWavefield",
        "MaterialPropertyMap",
        "export_nli_nifti",
        "FENG2025_FREQUENCIES_HZ",
        "FENG2025_MEG_SENSITIVITY_RAD_PER_MM",
    ]:
        assert hasattr(elphy, name), f"elphy.{name} is missing"


def test_feng2025_metadata_consistent():
    """Frequency tuple and sensitivity table agree."""
    freqs = elphy.FENG2025_FREQUENCIES_HZ
    sens = elphy.FENG2025_MEG_SENSITIVITY_RAD_PER_MM
    assert set(freqs) == set(sens)
    assert len(freqs) == 4
