.. tapify documentation master file, created by
   sphinx-quickstart on Thu Sep  9 16:18:42 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

======
tapify
======

``tapify`` is a Python package that implements a suite of multitaper spectral estimation techniques for analyzing time series data. It supports analysis of both evenly and unevenly sampled time series data.

The multitaper statistic was first proposed by Thomson (1982) as a non-parametric estimator of the spectrum of a time series. It is attractive because it tackles the problems of bias and consistency, which makes it an improvement over the classical periodogram for evenly sampled data and the Lomb-Scargle periodogram for uneven sampling. In basic statistical terms, this estimator allows us to confidently look at the properties of a time series in the frequency or Fourier domain.

Installation instructions
=========================

Dependencies
------------

``tapify`` requires the use of 
 - `numpy <https://numpy.org/>`__,
 - `scipy <https://scipy.org/>`__,
 - `matplotlib <https://matplotlib.org/>`__ (for plotting),
 - `finufft <https://finufft.readthedocs.io/en/latest/>`__ or `nfft <https://github.com/jakevdp/nfft>`__ (for uneven sampling).

Installation
------------

``tapify`` is currently not yet available on PyPI, but it can be
installed by downloading the source code or cloning the GitHub
repository and running the standard::

       python setup.py install

command or its usual variants (``python setup.py install --user``,
``python setup.py install --prefix=/PATH/TO/INSTALL/DIRECTORY``,
etc.).

For more information, please open an Issue on the GitHub page.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   intro.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
