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

.. code-block:: python

    from muscat2ta..m2mnlpf import M2MultiNightLPF

    lpf = M2MultiNightLPF(name, use_opencl=USE_OPENCL)


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