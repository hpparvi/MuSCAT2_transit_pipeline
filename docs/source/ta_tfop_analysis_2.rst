Clearing the field
------------------

In cases where the candidate signal is considered too shallow to be securely confirmed with MuSCAT2, we can opt to let
the target star saturate in order to get good photometry from the faint nearby stars to rule out blended EBs. In these
cases we carry out all the TFOP analysis steps except the transit fitting, but we will first need to tell the ``TFOPAnalysis``
class that we're only interested in clearing the field, and we may need to help the blend-plotting method a bit if it cannot
automatically compute the flux ratios between the target star and the possible blends.

Initialisation
**************

Since the main target is saturated, ``TFOPAnalysis`` needs to be initialized with a flag that tells it that the target
has been allowed to saturate and that we're only interested in clearing the field. This can be done by setting the
``clear_field_only`` argument to ``True``

.. code-block:: python

    ta = TFOPAnalysis(..., clear_field_only=True)


So, for example

.. code-block:: python

    ta = TFOPAnalysis('toi01557.01', '200819', 0, [2,3,4], clear_field_only=True)`

Plotting possible blends
************************

The blend-plotting method uses the flux ratio between the target and a possible blend to plot the expected signal depth.
The method needs to be given the flux-ratio between the target and the comparison star if the target is completely saturated
in some passbands. If the target has some good photometry in some passband, the flux ratio can be calculated as

.. code-block:: python

    fr = ta.flux_ratio(PBI, S1, S2)

where ``PBI`` is the passband index, ``S1`` is the target star index, and ``S2`` is the index of the star chosen as the comparison
star in the possible blend plotting. Now, the possible blends can be plotted as

.. code-block:: python

    ta.plot_possible_blends(CID, AID, CAID, c_flux_factor=fr)

where ``c_flux_factor`` is the flux ratio between the target star and the comparison star. If the target is saturated in all
observations, the ratio can be calculated, for example, based on GAIA catalog values.

Wrapping up the analysis
************************
