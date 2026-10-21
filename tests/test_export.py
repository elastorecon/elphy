"""Tests for elphy.export (solver-agnostic inversion handoff)."""

import numpy as np
import pytest

import elphy


def _payload(nfreq=2):
    rng = np.random.default_rng(0)
    shape = (6, 5, 4, 3, nfreq)
    field = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    mask = np.zeros((6, 5, 4), dtype=bool)
    mask[1:-1, 1:-1, 1:-1] = True
    mag = rng.standard_normal((6, 5, 4))
    freqs = [50.0, 60.0][:nfreq]
    return elphy.build_inversion_payload(
        field,
        voxel_size=(3.0, 3.0, 3.0),
        frequencies=freqs,
        mask=mask,
        magnitude=mag,
        quality={"oss_snr": 12.3},
        name="unit",
    ), field


def test_build_promotes_4d_to_single_frequency():
    field = np.zeros((4, 4, 4, 3), dtype=complex)
    p = elphy.build_inversion_payload(field, 3.0, frequencies=50.0)
    assert p.wavefield.shape == (4, 4, 4, 3, 1)
    assert p.frequencies_hz.tolist() == [50.0]
    assert p.voxel_size_mm.tolist() == [3.0, 3.0, 3.0]


def test_build_rejects_bad_direction_length():
    field = np.zeros((4, 4, 4, 2, 1), dtype=complex)
    with pytest.raises(ValueError, match="direction"):
        elphy.build_inversion_payload(field, 3.0, frequencies=50.0)


def test_build_rejects_frequency_mismatch():
    field = np.zeros((4, 4, 4, 3, 2), dtype=complex)
    with pytest.raises(ValueError, match="frequenc"):
        elphy.build_inversion_payload(field, 3.0, frequencies=[50.0])


def test_npz_roundtrip_lossless(tmp_path):
    p, field = _payload()
    f = tmp_path / "wf.npz"
    elphy.export_inversion_npz(f, p)
    q = elphy.read_inversion_npz(f)
    assert np.allclose(q.wavefield, p.wavefield)
    assert np.array_equal(q.mask, p.mask)
    assert np.allclose(q.magnitude, p.magnitude)
    assert np.allclose(q.frequencies_hz, p.frequencies_hz)
    assert q.quality["oss_snr"] == pytest.approx(12.3)
    assert q.name == "unit"
    assert q.metadata["wavefield_layout"] == "[x, y, z, direction, frequency]"


def test_mat_export_direction_mapping(tmp_path):
    from scipy.io import loadmat

    p, field = _payload()
    f = tmp_path / "inv.mat"
    elphy.export_inversion_mat(f, p, frequency_index=0)
    m = loadmat(str(f))
    # Xmotion=dir0, Ymotion=dir1, Zmotion=dir2 at frequency index 0.
    assert np.allclose(m["Xmotion"], field[:, :, :, 0, 0])
    assert np.allclose(m["Ymotion"], field[:, :, :, 1, 0])
    assert np.allclose(m["Zmotion"], field[:, :, :, 2, 0])
    assert "mask" in m and "t2stack" in m


def test_mat_params_struct(tmp_path):
    from scipy.io import loadmat

    p, _ = _payload()
    f = tmp_path / "inv.mat"
    elphy.export_inversion_mat(f, p, frequency_index=1)
    m = loadmat(str(f), squeeze_me=True, struct_as_record=False)
    mp = m["mreParams"]
    assert mp.freq == 60.0
    assert mp.nx == 5 and mp.ny == 6 and mp.nz == 4
    assert mp.FOVx == pytest.approx(5 * 3.0)
    assert mp.oss_snr == pytest.approx(12.3)


def test_multifreq_mat_requires_index(tmp_path):
    p, _ = _payload()
    with pytest.raises(ValueError, match="frequency_index"):
        elphy.export_inversion_mat(tmp_path / "x.mat", p)


def test_custom_direction_names(tmp_path):
    from scipy.io import loadmat

    p, field = _payload()
    f = tmp_path / "inv.mat"
    elphy.export_inversion_mat(f, p, frequency_index=0, direction_names=("Ux", "Uy", "Uz"))
    m = loadmat(str(f))
    assert np.allclose(m["Ux"], field[:, :, :, 0, 0])
    assert "Xmotion" not in m


def test_nifti_export_components(tmp_path):
    nib = pytest.importorskip("nibabel")
    p, field = _payload()
    paths = elphy.export_inversion_nifti(tmp_path / "nii", p, frequency_index=0)
    # 3 directions x (real, imag) + mask + magnitude = 8 files.
    assert len(paths) == 8
    img = nib.load([q for q in paths if q.endswith("ux_real.nii.gz")][0])
    assert np.allclose(img.get_fdata(), field[:, :, :, 0, 0].real, atol=1e-5)
    # Voxel spacing lands in the affine.
    assert np.allclose(np.abs(np.diag(img.affine))[:3], [3.0, 3.0, 3.0])


def test_export_wavefield_dispatch(tmp_path):
    p, _ = _payload(nfreq=1)
    elphy.export_wavefield(tmp_path / "a.npz", p)
    assert (tmp_path / "a.npz").exists()
    elphy.export_wavefield(tmp_path / "b.mat", p)
    assert (tmp_path / "b.mat").exists()
    with pytest.raises(ValueError, match="cannot infer"):
        elphy.export_wavefield(tmp_path / "c.foo", p)
