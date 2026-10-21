"""Tests for elphy.cli."""

import pytest
from click.testing import CliRunner

from elphy.cli import main

from .test_io import _write_mbmre_mat, _write_nli_reconprops_mat


def test_mbmre2nifti_default_output_dir(tmp_path):
    pytest.importorskip("nibabel")

    mat_path = tmp_path / "mbmre_for_inversion.mat"
    _write_mbmre_mat(mat_path)

    runner = CliRunner()
    result = runner.invoke(
        main, ["mbmre2nifti", str(mat_path), "--prefix", "sub-TEST_ses-A_run-1"]
    )

    assert result.exit_code == 0, result.output
    out_dir = tmp_path / "mbmre_nifti"
    written = sorted(p.name for p in out_dir.iterdir())
    assert written == sorted(
        f"sub-TEST_ses-A_run-1_{suffix}"
        for suffix in (
            "ux_real.nii.gz",
            "ux_imag.nii.gz",
            "uy_real.nii.gz",
            "uy_imag.nii.gz",
            "uz_real.nii.gz",
            "uz_imag.nii.gz",
            "mask.nii.gz",
            "magnitude.nii.gz",
        )
    )
    assert all(str(out_dir / name) in result.output for name in written)


def test_mbmre2nifti_explicit_output_dir(tmp_path):
    pytest.importorskip("nibabel")

    mat_path = tmp_path / "mbmre_for_inversion.mat"
    _write_mbmre_mat(mat_path)
    out_dir = tmp_path / "custom_out"

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["mbmre2nifti", str(mat_path), "--prefix", "sub-TEST_ses-A_run-1", "-o", str(out_dir)],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "sub-TEST_ses-A_run-1_magnitude.nii.gz").exists()


def test_mbmre2nifti_requires_prefix(tmp_path):
    mat_path = tmp_path / "mbmre_for_inversion.mat"
    _write_mbmre_mat(mat_path)

    runner = CliRunner()
    result = runner.invoke(main, ["mbmre2nifti", str(mat_path)])
    assert result.exit_code != 0
    assert "--prefix" in result.output


def test_nli2nifti_default_output_dir(tmp_path):
    pytest.importorskip("nibabel")

    mat_path = tmp_path / "ReconProps.mat"
    _write_nli_reconprops_mat(mat_path)

    runner = CliRunner()
    result = runner.invoke(main, ["nli2nifti", str(mat_path), "--prefix", "sub-TEST_ses-A_run-1"])

    assert result.exit_code == 0, result.output
    out_dir = tmp_path / "nli_nifti"
    written = sorted(p.name for p in out_dir.iterdir())
    expected = sorted(
        f"sub-TEST_ses-A_run-1_{suffix}.nii.gz"
        for suffix in (
            "realshear",
            "imagshear",
            "realbulk",
            "imagbulk",
            "realdens",
            "imagdens",
            "dampingratio",
            "rc",
            "magnitude",
        )
    ) + ["sub-TEST_ses-A_run-1.json"]
    assert written == sorted(expected)
    assert all(str(out_dir / name) in result.output for name in written)


def test_nli2nifti_requires_prefix(tmp_path):
    mat_path = tmp_path / "ReconProps.mat"
    _write_nli_reconprops_mat(mat_path)

    runner = CliRunner()
    result = runner.invoke(main, ["nli2nifti", str(mat_path)])
    assert result.exit_code != 0
    assert "--prefix" in result.output
