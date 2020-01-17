Installation
============

Prerequisites
-------------

The pipeline requires a set of Python packages that can be easily installed using `conda` (or `pip`)

.. code-block:: bash

    conda install numpy scipy tqdm traitlets pandas xarray matplotlib seaborn numba uncertainties
    conda install -c conda-forge astropy pyopencl emcee corner
    conda install -c astropy astroquery photutils

The pipeline relies also on `PyTransit` and `LDTk`, and these are best to be cloned from GitHub for easy updating.
First, you should go to a directory you store software source code (or create one if you don't have one already), and
clone the two repositories from GitHub inside this directory

.. code-block:: bash

    git clone https://github.com/hpparvi/PyTransit.git
    cd PyTransit
    git checkout dev
    python setup.py install

.. code-block:: bash

    git clone https://github.com/hpparvi/ldtk.git
    cd ldtk
    git checkout dev
    python setup.py install

After cloning and installation, you should keep an eye on code updates on these two packages, and pull the changes and
reinstall when a package is updated

.. code-block:: bash

    cd pytransit
    git pull
    python setup.py install

MuSCAT2 analysis pipeline
-------------------------

The pipeline can be installed after all the required packages have been installed

.. code-block:: bash

    git clone https://github.com/hpparvi/MuSCAT2_transit_pipeline.git
    cd MuSCAT2_transit_pipeline
    python setup.py install
    python setup.py develop
