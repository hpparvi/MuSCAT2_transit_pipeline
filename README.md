# MuSCAT2 transit analysis pipeline

A Python-based transit analysis pipeline for MuSCAT2-observed multicolour photometry developed in collaboration with the Instituto de Astrofísica de Canarias (IAC), XXX, YYY, and ZZZ. **(please fill in all the relevant institutes)**

## Overview

MuSCAT2 photometry pipeline calculates aperture photometry for a set of aperture sizes. The transit analysis pipeline

1. Selects the optimal aperture sizes based on noise characteristics
2. Carries out multicolour transit modelling using Gaussian processes (GPs) for systematics
3. Creates plots for the light curves, systematics, and parameter posteriors (parameter-parameter joint plots and marginal posteriors)

## Requirements

 - Python 3
 - NumPy, SciPy, scikit-learn, astropy, IPython, matplotlib
 - PyTransit

## Installation

    git clone XXX
    cd XXX
    python setup.py install [--user]

## Usage

    m2tm input_file

## Contributors

- Hannu Parviainen
- XXX XXX
- YYY YYY

&copy; 2017 Hannu Parviainen
