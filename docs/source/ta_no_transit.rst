Light curve reduction without a transit
---------------------------------------

The pipeline has two main use-cases with non-transit observations

1. Reduction of a non-normalized and non-detrended light curves for a set of stars.
2. Reduction of normalized and detrended relative light curve.

The first one is the most straight-forward, and the product will be a fits file with the light curves for
a set of stars and the covariates stored as binary tables. The second will require optimization, and will
produce a fits file with a normalized and detrended light curve.

Initialization
**************

If the observations do not include a transit, `TransitAnalysis` can be initialized with `with_transit` set to
`False`.

.. code-block:: python

    ta = TransitAnalysis(<TARGET>, <NIGHT>, tid=<TID>, cids=<CIDS>, with_transit=False)

This simplifies the light curve model significantly. Next, we can either export the raw light curves for the
target and comparison stars directly without detrending, or we can optimize the comparison stars, apertures,
and a linear baseline model.

We usually don't need to set any priors, but we can add flares to the model as explained in `flares`_.

Exporting raw light curves
**************************

In some situations all we want is to extract a set of light curves without detrending or normalization. This can
be done with the `TransitAnalysis.save_raw_fits` method.

.. code-block:: python

    ta.save_raw_fits(plot=True, save=True)

The method plots a reference frame with the target and comparison stars marked and the raw light curves for the
target and comparison stars (normalized for visualization), and saves the light curves and covariates as binary
tables in a multi-extension fits file.

**Note:** For stellar variability studies, care should be taken to use the same set of comparison stars from one
night to another (so that the final relative light curve makes sense). Also, enough comparison stars should be
chosen that some can be excluded if they themselves show variability.

Exporting detrended light curves
********************************

In some other cases we want to extract a detrended and normalized light curve. This can be achieved by first
optimizing the light curve model, and then calling the `TransitAnalysis.save_fits` method.

.. code-block:: python

    ta.optimize_global()
    ta.plot_light_curve()
    ta.save_fits()

Here ´ta.save_fits´ saves the detrended and normalized relative light curve and the covariates as binary tables in
a multi-extension fits file. The global optimization should be ran until the progress bar turns red, and the
`ta.plot_light_curve` method should be used to check that the final fit makes sense.
