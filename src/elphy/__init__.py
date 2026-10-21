"""Top-level package for MR Elastography (elphy).

``elphy`` provides the general, scanner-agnostic building blocks for
magnetic resonance elastography (MRE) processing: loading reconstructed
images, combining motion-encoding polarities, phase unwrapping, and
converting motion-encoded phase into a complex displacement wavefield for
downstream material-property inversion.
"""

__author__ = """Aaron T Anderson"""
__email__ = "aandrsn3@illinois.edu"
__version__ = "0.1.0"

# Core wavefield operations (numpy-only; always importable).
# I/O helpers (scipy / h5py imported lazily inside the functions).
from .export import (
    InversionPayload,
    MaterialPropertyMap,
    build_inversion_payload,
    export_inversion_mat,
    export_inversion_nifti,
    export_inversion_npz,
    export_nli_nifti,
    export_wavefield,
    read_inversion_npz,
)
from .inversion import (
    curl_based_inversion,
    direct_inversion,
    lfe_inversion,
    local_frequency_estimation,
    shear_stiffness,
)
from .io import (
    FENG2025_FREQUENCIES_HZ,
    FENG2025_MEG_SENSITIVITY_RAD_PER_MM,
    MREWavefield,
    load_mat_array,
    load_mbmre_for_inversion,
    load_nli_reconprops,
    load_scidb_feng2025,
)
from .operators import (
    bandpass_filter,
    curl,
    divergence,
)
from .strain import (
    motion_noise_amplitude,
    octahedral_normal_strain,
    octahedral_shear_strain,
    oss_snr,
    strain_tensor,
)
from .wavefield import (
    combine_encodings,
    meg_sensitivity_muthupillai,
    phase_to_displacement,
    raw_phase_to_wavefield,
    temporal_harmonic,
    unwrap_field,
    wrap_to_pi,
)

__all__ = [
    "wrap_to_pi",
    "combine_encodings",
    "unwrap_field",
    "phase_to_displacement",
    "meg_sensitivity_muthupillai",
    "temporal_harmonic",
    "raw_phase_to_wavefield",
    "bandpass_filter",
    "curl",
    "divergence",
    "strain_tensor",
    "octahedral_shear_strain",
    "octahedral_normal_strain",
    "motion_noise_amplitude",
    "oss_snr",
    "direct_inversion",
    "curl_based_inversion",
    "local_frequency_estimation",
    "lfe_inversion",
    "shear_stiffness",
    "InversionPayload",
    "build_inversion_payload",
    "export_inversion_mat",
    "export_inversion_npz",
    "read_inversion_npz",
    "export_inversion_nifti",
    "export_wavefield",
    "MaterialPropertyMap",
    "export_nli_nifti",
    "load_mat_array",
    "load_mbmre_for_inversion",
    "load_nli_reconprops",
    "load_scidb_feng2025",
    "MREWavefield",
    "FENG2025_FREQUENCIES_HZ",
    "FENG2025_MEG_SENSITIVITY_RAD_PER_MM",
]
