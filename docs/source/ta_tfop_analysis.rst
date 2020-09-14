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

.. toctree::
   :maxdepth: 2

   ta_tfop_analysis_1
   ta_tfop_analysis_2
   ta_tfop_analysis_3

