=======
History
=======

Unreleased
----------

* Add wavefield core (polarity combination, phase unwrapping, Muthupillai
  phase-to-displacement) and MRE data I/O, including a loader for the
  Feng et al. 2025 public dataset.
* Add temporal first-harmonic extraction (``temporal_harmonic``) and an
  optional ``harmonic`` step in ``raw_phase_to_wavefield``.
* Add an ``elphy.operators`` module with opt-in wavefield operators:
  curl, divergence, and a radial Butterworth k-space band-pass filter.
* Add an ``elphy.strain`` module with the octahedral shear strain SNR of
  McGarry et al. (2011): strain tensor, time-averaged OSS/ONS, motion-noise
  amplitude, and ``oss_snr`` (scalar and per-voxel distribution).
* Add an ``elphy.export`` module: a solver-agnostic inversion payload with
  writers for MATLAB ``.mat`` (the ``mbmre`` handoff), NumPy ``.npz``
  (lossless round trip via ``read_inversion_npz``), and per-component
  NIfTI.
* Modernize packaging: PEP 621 ``pyproject.toml``, ``src/`` layout,
  GitHub Actions CI (Python 3.10-3.14), and ``ruff`` for linting and
  formatting.

0.1.0 (2018-09-27)
------------------

* First release on PyPI.
