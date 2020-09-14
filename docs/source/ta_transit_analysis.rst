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

Secondary eclipse modelling
***************************

TBD