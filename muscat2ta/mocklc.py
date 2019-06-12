#  MuSCAT2 photometry and transit analysis pipeline
#  Copyright (C) 2019  Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import warnings
import exodata
import numpy as np
import matplotlib.pyplot as pl
from os.path import join, split

from numpy import array, linspace, zeros, zeros_like, cos, atleast_2d, diag, arange, tile, newaxis, arccos
from numpy.random import multivariate_normal, seed

from exodata.astroquantities import Quantity as Qty

from pytransit import MandelAgol as MA
from pytransit.orbits_f import orbits as of
from tpc import SpectrumTool, Instrument, TabulatedFilter
from tpc.filter import *

warnings.filterwarnings("ignore", category=UserWarning)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    exocat = exodata.OECDatabase(join(split(__file__)[0], '../ext/oec/systems/'))

try:
    from george import GP
    from george.kernels import ExpKernel, Matern32Kernel
    with_george = True
except ImportError:
    with_george = False

class MockLC:
    pb_names = 'g r i z'.split()
    pb_centers = 1e-9*array([470, 640, 780, 900])
    npb = len(pb_names)

    def __init__(self, planet_name='wasp-80b', **kwargs):
        self.t_exposure_d = Qty(kwargs.get('exptime', 60), 's').rescale('d')
        self.t_baseline_d = Qty(kwargs.get('bltime', 0.5), 'h').rescale('d')
        self.ldcs = kwargs.get('ldcs', array([[0.80,0.02], [0.61,0.16], [0.45,0.20], [0.36,0.20]]))

        self.filters = 'g r i z'.split()
        self.npb = len(self.filters)
        
        self.planet = planet = exocat.searchPlanet(planet_name)
        self.star = star = planet.star

        self.p = kwargs.get('p', None) or float(planet.P)
        self.k = kwargs.get('k', None) or float(planet.R.rescale('R_s') / star.R)
        self.a = kwargs.get('a', None) or float(planet.getParam('semimajoraxis').rescale('R_s') / star.R)
        self.i = float(planet.getParam('inclination').rescale('rad'))
        self.b = kwargs.get('b', self.a * cos(self.i))
        self.i = arccos(self.b / self.a)
        self.duration_d = Qty(of.duration_eccentric_f(self.p, self.k, self.a, self.i, 0, 0, 1), 'd')

        # Contamination
        # -------------
        qe_be = TabulatedFilter('1024B_eXcelon',
                                [300, 325, 350, 400, 450, 500, 700, 800, 850, 900, 950, 1050, 1150],
                                [0.0, 0.1, 0.25, 0.60, 0.85, 0.92, 0.96, 0.85, 0.70, 0.50, 0.30, 0.05, 0.0])
        qe_b = TabulatedFilter('2014B',
                               [300, 350, 500, 550, 700, 800, 1000, 1050],
                               [0.10, 0.20, 0.90, 0.96, 0.90, 0.75, 0.11, 0.05])
        qes = qe_be, qe_b, qe_be, qe_be

        instrument = Instrument('MuSCAT2', (sdss_g, sdss_r, sdss_i, sdss_z), qes)
        self.contaminator = SpectrumTool(instrument, "i'")

        self.i_contamination = kwargs.get('i_contamination', 0.0)
        self.cnteff = kwargs.get('contaminant_temperature', None) or float(star.T)
        self.k0 = self.k/np.sqrt(1-self.i_contamination)
        self.contamination = self.contaminator.contamination(self.i_contamination, float(star.T), self.cnteff)

    @property
    def t_total_d(self):
        return self.duration_d + 2*self.t_baseline_d
        
    @property
    def duration_h(self):
        return self.duration_d.rescale('h')

    @property
    def n_exp(self):
        return int(self.t_total_d // self.t_exposure_d)
    
    def __call__(self, rseed=0, ldcs=None, wnsigma=None, rnsigma=None, rntscale=0.5):
        return self.create(rseed, ldcs, wnsigma, rnsigma, rntscale)
    
    def create(self, rseed=0, ldcs=None, wnsigma=None, rnsigma=None, rntscale=0.5, nights=1):
        ldcs = ldcs if ldcs is not None else self.ldcs
        seed(rseed)
        
        self.time = linspace(-0.5*float(self.t_total_d), 0.5*float(self.t_total_d), self.n_exp)
        self.time = (tile(self.time, [nights, 1]) + (self.p*arange(nights))[:,newaxis]).ravel()
        self.npt = self.time.size

        self.transit = zeros([self.npt, 4])
        for i, (ldc, c) in enumerate(zip(ldcs, self.contamination)):
            self.transit[:, i] = MA().evaluate(self.time, self.k0, ldc, 0, self.p, self.a, self.i, c=c)

        # White noise
        # -----------
        if wnsigma is not None:
            self.wnoise = multivariate_normal(zeros(atleast_2d(self.transit).shape[1]), diag(wnsigma)**2, self.npt)
        else:
            self.wnoise = zeros_like(self.transit)

        # Red noise
        # ---------
        if rnsigma and with_george:
            self.gp = GP(rnsigma**2 * ExpKernel(rntscale))
            self.gp.compute(self.time)
            self.rnoise = self.gp.sample(self.time, self.npb).T
            self.rnoise -= self.rnoise.mean(0)
        else:
            self.rnoise = zeros_like(self.transit)
            
        # Final light curve
        # -----------------
        self.time_h = Qty(self.time, 'd').rescale('h')
        self.flux = self.transit + self.wnoise + self.rnoise
        return self.time_h, self.flux
    
    
    def plot(self, figsize=(13,4)):
        fig,axs = pl.subplots(1,3, figsize=figsize, sharex=True, sharey=True)
        yshift = 0.01*arange(4)
        axs[0].plot(self.time_h, self.flux + yshift)
        axs[1].plot(self.time_h, self.transit + yshift)
        axs[2].plot(self.time_h, 1 + self.rnoise + yshift)
        pl.setp(axs, xlabel='Time [h]', xlim=self.time_h[[0,-1]])
        pl.setp(axs[0], ylabel='Normalised flux')
        [pl.setp(ax, title=title) for ax,title in 
           zip(axs, 'Transit model + noise, Transit model, Red noise'.split(', '))]
        fig.tight_layout()
        return fig, axs


    def plot_color_difference(self, figsize=(13,4)):
        fig, axs = pl.subplots(2, 3, figsize=figsize, sharex=True, sharey=True)
        [ax.plot(self.time_h, 100*(fl - self.transit[:, -1])) for ax, fl in zip(axs[0], self.transit[:, :-1].T)]
        [ax.plot(self.time_h, 100*(fl - self.flux[:, -1])) for ax, fl in zip(axs[1], self.flux[:, :-1].T)]
        [pl.setp(ax, title='F$_{}$ - F$_z$'.format(pb)) for ax,pb in zip(axs[0], self.pb_names[:-1])]
        pl.setp(axs[:, 0], ylabel='$\Delta F$ [%]')
        pl.setp(axs[1, :], xlabel='Time [h]')
        pl.setp(axs, xlim=self.time_h[[0, -1]])
        fig.tight_layout()
        return fig
