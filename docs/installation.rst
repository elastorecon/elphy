.. highlight:: shell

============
Installation
============


Requirements
------------

elphy requires Python 3.10 or newer. Its runtime dependencies (NumPy, SciPy,
h5py, nibabel, scikit-image, and Click) are installed automatically.


Stable release
--------------

.. note::

   The ``elphy`` release currently on PyPI (0.1.0) predates the current code
   base. Until a new release is published, install from source (below).

To install the latest release from PyPI, run:

.. code-block:: console

    $ pip install elphy

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: https://docs.python-guide.org/starting/installation/


From sources
------------

The sources for MR Elastography can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone https://github.com/Elastography/elphy.git

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/Elastography/elphy/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ pip install .

For development (tests, linting, and building these docs), install in
editable mode with the ``dev`` extra:

.. code-block:: console

    $ pip install -e ".[dev]"


.. _Github repo: https://github.com/Elastography/elphy
.. _tarball: https://github.com/Elastography/elphy/tarball/master
