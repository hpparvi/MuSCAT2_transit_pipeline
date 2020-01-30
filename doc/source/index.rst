.. MuSCAT2_pipeline documentation master file, created by
   sphinx-quickstart on Fri Jan 17 19:58:12 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MuSCAT2 data analysis pipeline
==============================

MuSCAT2 transit analysis pipeline consists of a set of Python scripts and classes that aim to make the analysis of
MuSCAT2 photometry easy and painless. The pipeline covers the reduction of generic (non-transit) photometry, transit
analysis, and more specific TESS follow up analysis. The pipeline is mainly aimed to be used from inside a Jupyter
notebook, but it can also be used from inside a Python script.

The pipeline is divided into two Python packages: `muscat2ph` for photometry and `muscat2ta` for light curve reduction
and transit analysis.


.. toctree::
   :maxdepth: 3
   :caption: Contents:

   overview
   photometry
   transit_analysis


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
