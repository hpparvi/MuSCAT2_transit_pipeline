Secondary eclipse modelling
===========================

Secondary eclipses are modelled using ``muscat2ta.M2EclipseLPF``. The eclipse modelling LPF uses the reduced fits-format light curves,
so you need to first carry out a per-night light curve reduction using the standard ``muscat2ta.TransitAnalysis`` class. The class
can be used in its simplest form as

.. code-block:: python

    from muscat2ta import M2EclipseLPF

    lpf = M2EclipseLPF(name)

where ``name`` is the analysis name that will be used also when saving the modelling results, and it also takes a number of arguments that
can be used to tell where to look for data and tune the analysis. The full signature is

.. code-block:: python

    M2EclipseLPF(name,
                 datadir: Union[str, Path] = 'results',
                 pattern: str = '*.fits',
                 downsample: Optional[float] = None,
                 model_baseline: bool = True)

where ``datadir`` is the directory where the light curves are stored, ``pattern`` is a glob pattern used to identify the light curves,
``downsample`` sets binning in time (in seconds), and ``model_baseline`` sets whether to use a linear baseline model or not.