=======================
elphy — MR Elastography
=======================


.. image:: https://github.com/Elastography/elphy/actions/workflows/ci.yml/badge.svg
        :target: https://github.com/Elastography/elphy/actions/workflows/ci.yml
        :alt: CI Status

.. image:: https://img.shields.io/pypi/v/elphy.svg
        :target: https://pypi.python.org/pypi/elphy
        :alt: PyPI Version

.. image:: https://readthedocs.org/projects/elphy/badge/?version=latest
        :target: https://elphy.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/badge/code%20style-ruff-000000.svg
        :target: https://github.com/astral-sh/ruff
        :alt: Code style: ruff


General, scanner-agnostic tools for magnetic resonance elastography (MRE):
load reconstructed images, build the complex displacement wavefield, and
prepare data for material-property (inversion) analysis.

``elphy`` is the public core of an MRE processing stack — the steps that
are common to all MRE regardless of scanner, acquisition, or downstream
inversion solver. Site- and solver-specific code layers on top of it.


* Free software: MIT license
* Documentation: https://elphy.readthedocs.io


Features
--------

* **Data loading** — read reconstructed MRE ``.mat`` files (both classic
  and HDF5/v7.3 formats), with a loader for the Feng et al. 2025 public
  MRE dataset (phantom, brain, liver) and its acquisition metadata.
* **Polarity combination** — combine positive/negative motion-encoding
  gradient polarities to cancel the static background phase and double
  the motion-encoded SNR.
* **Phase unwrapping** — masked spatial unwrapping over arbitrary
  encoding/direction/frequency axes.
* **Phase → displacement** — convert motion-encoded phase to physical
  displacement via the Muthupillai (1995) MEG sensitivity, replacing
  hard-coded scale factors with explicit sequence physics.
* **Pipeline** — a single ``raw_phase_to_wavefield`` call chains the
  steps from raw phase to a displacement wavefield ready for inversion.

A canonical 6-D array layout ``[x, y, z, encoding, direction, frequency]``
is used throughout.


Quick start
-----------

.. code-block:: python

    import elphy

    # Load one sample of the Feng et al. 2025 public dataset.
    data = elphy.load_scidb_feng2025(".../V4/phantom")

    # Object mask from the magnitude images.
    mag = data.mag[:, :, :, 0, 0, 2]          # dir 0, 50 Hz
    mask = mag > 0.1 * mag.max()

    # Raw phase -> displacement wavefield.
    wavefield = elphy.raw_phase_to_wavefield(data.phase_raw, mask=mask)


Installation
------------

.. code-block:: console

    git clone https://github.com/Elastography/elphy.git
    cd elphy
    python -m venv .venv
    .venv/bin/pip install -e ".[dev]"    # editable, with dev tools


Roadmap
-------

* Temporal-harmonic (first-harmonic) extraction into complex motion
  fields.
* Wavefield operators (curl/divergence, k-space filters) and quality
  metrics (octahedral-shear-strain SNR).
* Solver-agnostic export for downstream inversion codes.
* Alignment of derivatives with the BIDS standard, following the
  qMRI-BIDS extension (BEP001) precedent, toward a BIDS extension
  proposal for MRE.


Credits
-------

This package was originally scaffolded with Cookiecutter_ and the
`audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
