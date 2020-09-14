MuSCAT2 light curve reduction and transit analysis
==================================================

The MuSCAT2 transit analysis package (``muscat2ta``) does quite a bit more than just transit analysis.
The two main classes ``muscat2ta.TransitAnalysis`` and ``muscat2ta.TFOPAnalysis`` take in the MuSCAT2
photometry stored as NetCDF files

The package offers
tools for light curve reduction and detrending (whether the light curve contains a transit or not)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ta_pre_analysis
   ta_no_transit
   ta_transit_analysis
   ta_tfop_analysis