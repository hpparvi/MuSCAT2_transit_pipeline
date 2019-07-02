# MuSCAT2 Data Analysis Pipeline

## Introduction
The MuSCAT2 transit analysis pipeline consists of a set of Python scripts and classes that aim to make the analysis of 
MuSCAT2 photometry easy and painless. The pipeline covers the reduction of generic (non-transit) photometry, transit 
analysis, and more specific TESS follow up analysis. The pipeline is mainly aimed to be used from inside a Jupyter 
notebook, but it can also be used from inside a Python script.



## The main tasks of the pipeline

1. Visualisation of the raw data.
1. Selection of the comparison stars and best target and comparison star apertures.
1. 


The pipeline can either aim to identify the best apertures and reference stars automatically, or it can be given a set 
of reference stars and apertures to use by the user. The reference star and aperture selection is done during a global 
optimisation step where the reference star apertures are free parameters in the light curve model. 

The pipeline has one main high-level component, the TransitAnalysis classes. The class contains the methods for the 
comparison star selection, transit model fitting, MCMC sampling, etc. 

# Installation

## Install the basic prerequisites

The pipeline requires a set of Python packages that can be easily installed either using `pip` or 
`conda`:

`numpy`, `scipy`, `astropy`, `tqdm`, `traitlets`, `pandas`, `xarray`, `photutils`, `matplotlib`, `astroquery`, `corner`,
`seaborn`, `numba`, `uncertainties`

## Install emcee

The pipeline needs the latest version of `emcee` that is not available through conda or pip at the time of writing. It
is best installed directly from GitHub.

## Install PyTransit
    git clone https://github.com/hpparvi/PyTransit.git
    cd PyTransit
    python setup.py install
    
## Install LDTk
    git clone https://github.com/hpparvi/ldtk.git
    cd ldtk
    python setup.py install
    
## Install the MuSCAT2 transit analysis pipeline

    git clone https://github.com/hpparvi/MuSCAT2_transit_pipeline.git
    cd MuSCAT2_transit_pipeline
    python setup.py install
    python setup.py develop

## Pre-analysis steps

These steps are common to all analyses (TFOP, transit modelling, reduction of transitless light curves, etc.)

  1. Execute `m2init <target_name>` to create an analysis directory `<target_name>` with the default directory structure.
  1. Copy the photometry from each night to `<target_name>/photometry/<yyyymmdd>` subdirectories.
  1. Move into the analysis directory and execute `m2nbtemplate <target_name> <night>` to create a template notebook.
  1. Open the template notebook in Jupyter.

## Transit analysis
### Single transit

The template notebook begins with a cell initialising the main `TransitAnalysis` class

    ta = TransitAnalysis(target, night, tid=TID, cids=CIDS)

where `target` is the target name, `night` is the observing night, `tid` is the target ID, and `cids` is a list of IDs 
of reference stars to be included into the reference star optimisation (marked in the photometry reference frames). 
The `TransitAnalysis` class has options to tailor the analysis, but this information should be enough for the basic use 
cases. 



### Multiple transits



### TFOP analysis

1. After the generic pre-analysis steps, copy one .fits file with its corresponding .wcs file from the MuSCAT2 NAS directory
 photometry_org to the photometry directory (this step will be removed in the future)
1. Follow the template notebook 

### Observations without a (known) transit

First, if we're not expecting to see a transit in the light curve, it is useful to let `TransitAnalysis` know that. This 
can be done by setting the `with_transit` argument to `False`.

    ta = TransitAnalysis(target, night, tid=TID, cids=CIDS, with_transit=False)

## Tuning the analysis

### OpenCL

The pipeline can use PyTransit's OpenCL transit model for transit modelling, which can significantly accelerate the 
analysis. This can be done by initialising `TransitAnalysis` with the `with_opencl` argument set to `True`.

    ta = TransitAnalysis(target, night, tid=TID, cids=CIDS, with_opencl=True)

### Trimming the light curves

The light curves can be trimmed from the beginning and end by setting the `mjd_start` and `mjd_end` in the `TransitAnalysis`
initialisation. This may be necessary if a part of the light curve is corrupted, or has strong systematics due to large
airmass or similar.

### Restricting apertures used in the optimisation 

The apreture ranges used in the optimisation can be constrained by the `aperture_lims` argument in `TransitAnalysis` 
initialisation.