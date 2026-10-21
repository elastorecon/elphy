"""Console script for elphy."""

import os
import sys

import click


@click.group()
@click.version_option()
def main():
    """elphy: MR elastography processing toolkit."""


@main.command("mbmre2nifti")
@click.argument("mat_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--prefix",
    required=True,
    help="Filename prefix for the written NIfTI files, e.g. sub-DEV3_ses-A_run-1.",
)
@click.option(
    "-o",
    "--out",
    "out_dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory. Default: an 'mbmre_nifti' directory next to MAT_PATH.",
)
def mbmre2nifti(mat_path, prefix, out_dir):
    """Convert a historical mbmre_for_inversion.mat handoff to NIfTI.

    Writes MAGNITUDE, MASK, and per-direction real/imag motion NIfTI
    volumes (see :func:`elphy.export.export_inversion_nifti`), in the
    same voxel orientation as mre_toolbox's own t2stack.nii/maskx.nii
    (see :func:`elphy.io.load_mbmre_for_inversion`).
    """
    from .export import export_inversion_nifti
    from .io import load_mbmre_for_inversion

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(mat_path)), "mbmre_nifti")

    payload = load_mbmre_for_inversion(mat_path)
    written = export_inversion_nifti(out_dir, payload, prefix=prefix)
    for p in written:
        click.echo(p)


@main.command("nli2nifti")
@click.argument("mat_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--prefix",
    required=True,
    help="Filename prefix for the written NIfTI/JSON files, e.g. sub-DEV3_ses-A_run-1.",
)
@click.option(
    "-o",
    "--out",
    "out_dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory. Default: an 'nli_nifti' directory next to MAT_PATH.",
)
def nli2nifti(mat_path, prefix, out_dir):
    """Convert an NLI *.ReconProps.mat material-property handoff to NIfTI.

    Writes real/imaginary shear modulus, bulk modulus, and density, a
    damping-ratio map, a Rayleigh-composition classification, and the
    structural MagIm reference as NIfTI volumes (see
    :func:`elphy.export.export_nli_nifti`), in the same voxel orientation
    as mre_toolbox's own maskx.nii/t2stack.nii (see
    :func:`elphy.io.load_nli_reconprops`), plus a JSON sidecar with the
    reconstruction parameters.
    """
    from .export import export_nli_nifti
    from .io import load_nli_reconprops

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(mat_path)), "nli_nifti")

    payload = load_nli_reconprops(mat_path)
    written = export_nli_nifti(out_dir, payload, prefix=prefix)
    for p in written:
        click.echo(p)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
