MuSCAT2 light curve reduction and transit analysis
==================================================

The MuSCAT2 transit analysis package (`muscat2ta`) does quite a bit more than just transit analysis. The package offers
tools for light curve reduction and detrending (whether the light curve contains a transit or not)

Pre-analysis steps
------------------

#. Execute `m2init <target_name>` to create an analysis directory `<target_name>` with the default directory structure.
#. Copy the photometry from each night to `<target_name>/photometry/<yyyymmdd>` subdirectories.
#. Move into the analysis directory and execute `m2nbtemplate <target_name> <night>` to create a template notebook.
#. Open the template notebook in Jupyter.

Transit analysis
----------------

Single transit
**************

The template notebook begins with a cell initialising the main `TransitAnalysis` class

.. code-block:: python

    ta = TransitAnalysis(target, night, tid=TID, cids=CIDS)

where `target` is the target name, `night` is the observing night, `tid` is the target ID, and `cids` is a list of IDs
of reference stars to be included into the reference star optimisation (marked in the photometry reference frames).
The `TransitAnalysis` class has options to tailor the analysis, but this information should be enough for the basic use
cases.


Multiple epochs
***************


TFOP analysis
*************

#. After the generic pre-analysis steps, copy one .fits file with its corresponding .wcs file from the MuSCAT2 NAS directory
   photometry_org to the photometry directory (this step will be removed in the future)
#. Follow the template notebook

Observations without a (known) transit
**************************************

First, if we're not expecting to see a transit in the light curve, it is useful to let `TransitAnalysis` know that. This
can be done by setting the `with_transit` argument to `False`.

.. code-block:: python

    ta = TransitAnalysis(target, night, tid=TID, cids=CIDS, with_transit=False)

Tuning the analysis
-------------------

OpenCL
******

The pipeline can use PyTransit's OpenCL transit model for transit modelling, which can significantly accelerate the
analysis. This can be done by initialising `TransitAnalysis` with the `with_opencl` argument set to `True`.

.. code-block:: python

    ta = TransitAnalysis(target, night, tid=TID, cids=CIDS, with_opencl=True)

Trimming the light curves
*************************

The light curves can be trimmed from the beginning and end by setting the `mjd_start` and `mjd_end` in the `TransitAnalysis`
initialisation. This may be necessary if a part of the light curve is corrupted, or has strong systematics due to large
airmass or similar.

Restricting apertures used in the optimisation
**********************************************

The apreture ranges used in the optimisation can be constrained by the `aperture_lims` argument in `TransitAnalysis`
initialisation.