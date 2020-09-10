TFOP analysis tutorial
======================

First, make sure you've done :ref:`the initialisation steps<pre analysis>`. Now, you should have a
Jupyter notebook with most of the steps pre-filled and ready to run.

Our TFOP analyses fall generally into two categories: a) full analysis including transit modelling, and
b) clearing the field where we do not try to model the transit. The full analysis is carried out when
the transit is expected to be deep enough for us to observe it with sufficient S/N to detect chromatic
depth variations. In this case the observations may have been done defocussed, and the target star
is not saturated.
The field clearing is done when the transit is too shallow for us but we can still expect to catch any
blended EBs. In this case, observations are carried out in as good focus as possible, and we often allow
the target star to saturate to increase the S/N of the faint nearby stars.

The first step in a TFOP analysis is to figure out if you're doing a transit modelling analysis or
field clearing. For this, consult the MuSCAT2 observation database and the per-night observing notes
sent to the MuSCAT2 mailing list.

Full analysis
-------------

The full analysis (with transit fitting) carries out the steps required for ExoFOP submission and
creates reduced light curves in fits-format that can be used in subsequent analyses (or shared with
collaborators).

The TFOP analysis is done with the ``muscat2ta.tfopanalysis.TFOPAnalysis`` class that contains all the necessary methods for the TFOP data reduction and analysis.

`TFOPAnalysis`` is initialised giving

 - ``target``: target name exactly as in the `MuSCAT2` catalog
 - ``date``: Observing date in the format ``yymmdd``
 - ``tid``: target ID. This should be 0 if we have an astrometric solution for the field, but can be something
            else if astrometry has failed for any reason. Please check the reference frames against the field
            shown for the target in the MuSCAT2 observation database.
 - ``cids``: list of IDs of the useful comparison stars to include into the comparison star optimisation.
             *Note:* The final set of comparison stars will be a subset of these stars. The analysis is relatively
             robust against bad comparison stars, but each comparison star adds free parameters to the optimisation,
             so it's best to constrain this to a small number of good comparison stars.

and has a set of optional arguments that can be used to fine-tune the analysis

 - ``passbands ['g','r','i','z_s']``: passbands to include into the analysis.
 - ``aperture_limits [(0, inf)]``: a (min, max) tuple of aperture ID limits for the aperture optimisation.
 - ``use_opencl [False]``: use OpenCL (some parts of the computations will be done with the GPU) if ``True``.
 - ``with_transit [True]``: include a transit model into the light curve model if ``True``.
 - ``contamination [None]``: allow the flux to be contaminated by an unresolved star if ``True``.
 - ``radius_ratio ['achromatic']``: ``chromatic`` for passband dependent radius ratios and ``achromatic`` for passband independent radius ratios.
 - ``npop [200]``: number of parameter vectors in the optimisation and MCMC.
 - ``toi [None]``: ``TOI`` object with the ExoFOP TOI information.
 - ``klims [(0.005, 0.25)]``: radius ratio limits for the transit model. The upper limit needs to be increased if modelling an EB.


Light curve cleanup
*******************

The first step is to look at the raw data, because this can already show any major issues with the observations (clouds, for example).
This is done with the ``TFOPAnalysis.plot_raw_light_curves`` method after the ``TFOPAnalysis`` has been initialised.

Next, we can trim sections out of the light curve and remove outliers. The commands and how to use them are documented in the
notebook template. We can also bin the observations using the ``TFOPAnalysis.downsample`` method. While binning should be avoided in many
science cases, we can generally bin to 60 seconds without any loss in information.

Creating the ExoFOP output
**************************

The possible blends are plotted using ``TFOPAnalysis.plot_possible_blends`` method:

.. code-block:: python

    ta.plot_possible_blends(cid=CID, aid=AID, caid=CAID)

where ``CID`` is a comparison star ID that should correspond to a bright (but not saturated) star outside of the 2 arcmin
circle, ``AID`` is the aperture index to use for the blending calculation, and ``CAID`` is the (optional) comparison star
aperture index that can be set if the comparison star is much brighter than the possible sources of blends.

Transit modelling
*****************

Wrapping up the analysis
************************

After finishing the notebook, make sure you fill the report file that has been created under the
``submit`` directory. The analysis code prefills most of the required information, but not the final
analysis conclusions. These should clearly state whether a transit signal occurs on the target and if
the fitted transit signal shows significant chromatic variability. Also include any note you believe
can be useful for people reading the report in ExoFOP and trying to decide if the observations show
support for a planet transit or something else.


Clearing the field
------------------

In cases where the candidate signal is considered too shallow to be securely confirmed with MuSCAT2, we can opt to let
the target star saturate in order to get good photometry from the faint nearby stars to rule out blended EBs. In these
cases we can't do transit fitting, but the main task will be carried out by the ``TFOPAnalysis.plot_possible_blends()``
method.

However, since the main target is saturated, TFOPAnalysis needs to be initialized with a flag letting it know that the
target saturation is not an issue, and blend plotting routine needs some extra information to be able to show the expected
transit depths. First, the TFOPAnalysis needs to be initialized as


.. code-block:: python

    ta = TFOPAnalysis(..., clear_field_only=True)

and after this, the blend plotting can be done as

.. code-block:: python

    fr = ta.flux_ratio(PBI, S1, S2)
    ta.plot_possible_blends(CID, AID, CAID, c_flux_factor=fr)

where ``c_flux_factor`` is the flux ratio between the target star and the comparison star. If the target is not saturated
in one passband, this can be calculated with ``TFOPAnalysis.flux_ratio`` method. If the target is saturated in all
observations, the ratio can be calculated, for example, based on GAIA catalog values.

Possible issues
---------------

- Reference passband needs to be set if ``r`` band is missing.
-