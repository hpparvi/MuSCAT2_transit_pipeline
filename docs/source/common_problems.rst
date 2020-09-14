FAQ
---------------

OpenCL
******

The pipeline can use PyTransit's OpenCL transit model for transit modelling, which can significantly accelerate the
analysis. This can be done by initialising `TransitAnalysis` with the `with_opencl` argument set to `True`.

.. code-block:: python

    ta = TransitAnalysis(target, night, tid=TID, cids=CIDS, with_opencl=True)

Restricting apertures used in the optimisation
**********************************************

The apreture ranges used in the optimisation can be constrained by the `aperture_lims` argument in `TransitAnalysis`
initialisation.