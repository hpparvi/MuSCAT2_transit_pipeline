# MuSCAT2 transit analysis pipeline

A Python-based transit analysis pipeline for MuSCAT2-observed multicolour photometry developed in collaboration with the [Instituto de Astrofísica de Canarias (IAC)](http://www.iac.es), University of Tokyo (UT), [National Astronomical Observatory of Japan (NAOJ)](http://www.nao.ac.jp), [The Graduate University for Advanced Studies (SOKENDAI)](http://guas-astronomy.jp), and [Astrobiology Center (ABC)](http://abc-nins.jp).

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

## Collaborators

- Instituto de Astrofísica de Canarias
- University of Tokyo
- National Astronomical Observatory of Japan
- The Graduate University for Advanced Studies
- Astrobiology Center

## Contributors

- Hannu Parviainen
- XXX XXX
- YYY YYY

&copy; 2017 Hannu Parviainen
