"""Tests for elphy.io."""

import warnings

import numpy as np
import pytest

import elphy
from elphy.io import _mbmre_axes_to_nifti


def _write_mbmre_mat(path, ny=6, nx=5, nz=4):
    """Write a synthetic mbmre_for_inversion.mat matching mre_toolbox's layout."""
    from scipy.io import savemat

    rng = np.random.default_rng(0)
    t2stack = rng.standard_normal((ny, nx, nz))
    mask = (rng.standard_normal((ny, nx, nz)) > 0).astype(np.uint8)
    motion = {
        name: rng.standard_normal((ny, nx, nz)) + 1j * rng.standard_normal((ny, nx, nz))
        for name in ("Xmotion", "Ymotion", "Zmotion")
    }
    mre_params = np.zeros(
        (1, 1),
        dtype=[
            ("subj", "O"),
            ("FOVx", "O"),
            ("FOVy", "O"),
            ("FOVz", "O"),
            ("nx", "O"),
            ("ny", "O"),
            ("nz", "O"),
            ("freq", "O"),
            ("oss_snr", "O"),
        ],
    )
    mre_params[0, 0]["subj"] = "mbmre_for_inversion"
    mre_params[0, 0]["FOVx"] = float(nx * 2)
    mre_params[0, 0]["FOVy"] = float(ny * 2)
    mre_params[0, 0]["FOVz"] = float(nz * 2)
    mre_params[0, 0]["nx"] = ny  # mre_toolbox sets nx = ny (source bug, mirrored here)
    mre_params[0, 0]["ny"] = ny
    mre_params[0, 0]["nz"] = nz
    mre_params[0, 0]["freq"] = 50.0
    mre_params[0, 0]["oss_snr"] = 21.8

    savemat(
        str(path),
        {
            "t2stack": t2stack,
            "mask": mask,
            "mreParams": mre_params,
            **motion,
        },
    )
    return t2stack, mask, motion


def test_mbmre_axes_to_nifti_matches_documented_recipe():
    """_mbmre_axes_to_nifti must equal flipdim(flipdim(permute(x,[2 1 3]),1),2)."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 4, 5))
    expected = np.flip(np.flip(np.transpose(x, (1, 0, 2)), axis=0), axis=1)
    assert np.array_equal(_mbmre_axes_to_nifti(x), expected)


def test_load_mbmre_for_inversion_roundtrip(tmp_path):
    ny, nx, nz = 6, 5, 4
    mat_path = tmp_path / "mbmre_for_inversion.mat"
    t2stack, mask, motion = _write_mbmre_mat(mat_path, ny, nx, nz)

    payload = elphy.load_mbmre_for_inversion(mat_path)

    # Spatial shape is [nx, ny, nz] after the axis swap.
    assert payload.magnitude.shape == (nx, ny, nz)
    assert payload.mask.shape == (nx, ny, nz)
    assert payload.wavefield.shape == (nx, ny, nz, 3, 1)

    assert np.array_equal(payload.magnitude, _mbmre_axes_to_nifti(t2stack))
    assert np.array_equal(payload.mask, _mbmre_axes_to_nifti(mask) != 0)
    for i, name in enumerate(("Xmotion", "Ymotion", "Zmotion")):
        assert np.array_equal(payload.wavefield[:, :, :, i, 0], _mbmre_axes_to_nifti(motion[name]))

    assert payload.voxel_size_mm.tolist() == [2.0, 2.0, 2.0]
    assert payload.frequencies_hz.tolist() == [50.0]
    assert payload.quality["oss_snr"] == pytest.approx(21.8)
    assert payload.name == "mbmre_for_inversion"


def _write_nli_reconprops_mat(path, ny=6, nx=5, nz=4, ny_mag=4, nx_mag=3, nz_mag=4):
    """Write a synthetic *.ReconProps.mat matching mre_toolbox's NLI layout."""
    from scipy.io import savemat

    rng = np.random.default_rng(2)
    props = {
        name: rng.standard_normal((ny, nx, nz))
        for name in (
            "DR",
            "RC",
            "RealShear",
            "ImagShear",
            "RealBulk",
            "ImagBulk",
            "RealDens",
            "ImagDens",
        )
    }
    mag_im = rng.standard_normal((ny_mag, nx_mag, nz_mag))

    invparam = np.zeros(
        (1, 1),
        dtype=[("version_number", "O"), ("freq_rads_inv", "O"), ("basistype", "O")],
    )
    invparam[0, 0]["version_number"] = 7.5
    invparam[0, 0]["freq_rads_inv"] = 314.159265
    invparam[0, 0]["basistype"] = np.array([[1, 1, 1]])

    meshinputs = np.zeros((1, 1), dtype=[("muest", "O"), ("rhoest", "O")])
    meshinputs[0, 0]["muest"] = 3300
    meshinputs[0, 0]["rhoest"] = 1000

    savemat(
        str(path),
        {
            **props,
            "MagIm": mag_im,
            "meshres": np.array([[1.6e-3] * 3, [8e-3] * 3, [16e-3] * 3]),
            "meshind": np.array([[1, 2, 2], [1, 2, 2]], dtype=np.uint8),
            "recondir": np.array(["/fake/recon/dir"]),
            "invparam": invparam,
            "meshinputs": meshinputs,
        },
    )
    return props, mag_im


def _write_interp_data_mat(path, ny=6, nx=5, nz=4, seed=3):
    """Write a synthetic *.InterpData.mat with a full-grid MagIm_int."""
    from scipy.io import savemat

    rng = np.random.default_rng(seed)
    magim_int = rng.standard_normal((ny, nx, nz))
    maskint = (rng.standard_normal((ny, nx, nz)) > 0).astype(np.uint8)
    savemat(str(path), {"MagIm_int": magim_int, "maskint": maskint})
    return magim_int


def test_load_nli_reconprops_roundtrip(tmp_path):
    """No sibling *.InterpData.mat here, so this also covers the fallback path."""
    ny, nx, nz = 6, 5, 4
    mat_path = tmp_path / "MRE-Zone_G3300.v7.5.inv.iso.incomp.visc..0500.ReconProps.mat"
    props, mag_im = _write_nli_reconprops_mat(mat_path, ny, nx, nz)

    with pytest.warns(UserWarning, match="InterpData"):
        payload = elphy.load_nli_reconprops(mat_path)

    assert payload.shear_modulus_pa.shape == (nx, ny, nz)
    assert np.array_equal(
        payload.shear_modulus_pa,
        _mbmre_axes_to_nifti(props["RealShear"]) + 1j * _mbmre_axes_to_nifti(props["ImagShear"]),
    )
    assert np.array_equal(
        payload.bulk_modulus_pa,
        _mbmre_axes_to_nifti(props["RealBulk"]) + 1j * _mbmre_axes_to_nifti(props["ImagBulk"]),
    )
    assert np.array_equal(
        payload.density_kg_m3,
        _mbmre_axes_to_nifti(props["RealDens"]) + 1j * _mbmre_axes_to_nifti(props["ImagDens"]),
    )
    assert np.array_equal(payload.damping_ratio, _mbmre_axes_to_nifti(props["DR"]))
    assert np.array_equal(payload.rayleigh_composition, _mbmre_axes_to_nifti(props["RC"]))
    assert np.array_equal(payload.magnitude, _mbmre_axes_to_nifti(mag_im))

    assert payload.voxel_size_mm.tolist() == [1.6, 1.6, 1.6]
    assert payload.magnitude_voxel_size_mm.tolist() == [1.6, 1.6, 1.6]
    assert payload.metadata["invparam"]["version_number"] == pytest.approx(7.5)
    assert payload.metadata["invparam"]["basistype"] == [1, 1, 1]
    assert payload.metadata["meshinputs"]["muest"] == 3300
    assert payload.metadata["recondir"] == "/fake/recon/dir"
    assert payload.metadata["interp_data_path"] is None


def test_load_nli_reconprops_finds_interp_data(tmp_path):
    """A sibling *.InterpData.mat's full-grid MagIm_int replaces the cropped MagIm."""
    ny, nx, nz = 6, 5, 4
    mesh_dir = tmp_path / "MREv7mesh_voxelmesh"
    inv_cascade = mesh_dir / "inv_cascade"
    inv_cascade.mkdir(parents=True)

    mat_path = inv_cascade / "MRE-Zone_G3300.v7.5.inv.iso.incomp.visc..0500.ReconProps.mat"
    _write_nli_reconprops_mat(mat_path, ny, nx, nz)

    interp_path = mesh_dir / "MREv7mesh_voxelmesh.InterpData.mat"
    magim_int = _write_interp_data_mat(interp_path, ny, nx, nz)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        payload = elphy.load_nli_reconprops(mat_path)

    assert payload.magnitude.shape == (nx, ny, nz)
    assert np.array_equal(payload.magnitude, _mbmre_axes_to_nifti(magim_int))
    assert payload.magnitude_voxel_size_mm.tolist() == [1.6, 1.6, 1.6]
    assert payload.metadata["interp_data_path"] == str(interp_path)


def test_load_nli_reconprops_falls_back_on_ambiguous_interp_data(tmp_path):
    """More than one *.InterpData.mat match is treated the same as none."""
    mat_path = tmp_path / "MRE-Zone_G3300.v7.5.inv.iso.incomp.visc..0500.ReconProps.mat"
    _, mag_im = _write_nli_reconprops_mat(mat_path)
    _write_interp_data_mat(tmp_path / "a.InterpData.mat")
    _write_interp_data_mat(tmp_path / "b.InterpData.mat")

    with pytest.warns(UserWarning, match="InterpData"):
        payload = elphy.load_nli_reconprops(mat_path)

    assert np.array_equal(payload.magnitude, _mbmre_axes_to_nifti(mag_im))
    assert payload.metadata["interp_data_path"] is None


def test_load_nli_reconprops_exports_nifti(tmp_path):
    """The loaded payload should feed straight into export_nli_nifti."""
    nib = pytest.importorskip("nibabel")

    mat_path = tmp_path / "ReconProps.mat"
    _write_nli_reconprops_mat(mat_path)
    with pytest.warns(UserWarning, match="InterpData"):
        payload = elphy.load_nli_reconprops(mat_path)

    out_dir = tmp_path / "nifti"
    written = elphy.export_nli_nifti(out_dir, payload, prefix="sub-TEST")
    assert len(written) == 10  # 8 property maps + magnitude + json sidecar

    nii_written = [p for p in written if p.endswith(".nii.gz")]
    assert len(nii_written) == 9
    for p in nii_written:
        img = nib.load(p)
        assert img.shape[:3] in (payload.damping_ratio.shape, payload.magnitude.shape)

    json_path = tmp_path / "nifti" / "sub-TEST.json"
    assert str(json_path) in written
    import json

    with open(json_path) as f:
        meta = json.load(f)
    assert meta["invparam"]["version_number"] == pytest.approx(7.5)


def test_load_mbmre_for_inversion_exports_nifti(tmp_path):
    """The loaded payload should feed straight into export_inversion_nifti."""
    nib = pytest.importorskip("nibabel")

    mat_path = tmp_path / "mbmre_for_inversion.mat"
    _write_mbmre_mat(mat_path)
    payload = elphy.load_mbmre_for_inversion(mat_path)

    out_dir = tmp_path / "nifti"
    written = elphy.export_inversion_nifti(out_dir, payload, prefix="sub-TEST")
    assert len(written) == 8  # 3 directions x (real, imag) + mask + magnitude
    for p in written:
        img = nib.load(p)
        assert img.shape[:3] == payload.magnitude.shape
