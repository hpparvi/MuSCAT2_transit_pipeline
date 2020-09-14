Clearing the field
------------------

In cases where the candidate signal is considered too shallow to be securely confirmed with MuSCAT2, we can opt to let
the target star saturate in order to get good photometry from the faint nearby stars to rule out blended EBs. In these
cases we can't do transit fitting, but the main task will be carried out by the ``TFOPAnalysis.plot_possible_blends()``
method.

Initialisation
**************

Since the main target is saturated, ``TFOPAnalysis`` needs to be initialized with a flag letting it know that the target
saturation is not an issue, and blend plotting routine needs some extra information to be able to show the expected
transit depths. First, the ``TFOPAnalysis`` needs to be initialized as

.. code-block:: python

    ta = TFOPAnalysis(..., clear_field_only=True)

Plotting possible blends
************************

Because the target is saturated, the method for plotting possible blends needs to be given the target-comparison flux ratio
as an argument. The flux ratio can be calculated as

.. code-block:: python

    fr = ta.flux_ratio(PBI, S1, S2)

where ``PBI`` is the passband index, ``S1`` is the target star index, and ``S2`` is the index of the star chosen as the comparison
star in the possible blend plotting. Now, the possible blends are plotted as

.. code-block:: python

    ta.plot_possible_blends(CID, AID, CAID, c_flux_factor=fr)

where ``c_flux_factor`` is the flux ratio between the target star and the comparison star. If the target is not saturated
in one passband, this can be calculated with ``TFOPAnalysis.flux_ratio`` method. If the target is saturated in all
observations, the ratio can be calculated, for example, based on GAIA catalog values.

Wrapping up the analysis
************************
