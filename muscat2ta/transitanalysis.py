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

from glob import glob
from pathlib import Path
from time import strftime
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
import xarray as xa
from astropy.io import fits as pf
from astropy.stats import mad_std
from astropy.table import Table
from corner import corner
from matplotlib.pyplot import subplots, setp, figure, figtext
from numpy import (array, arange, min, max, sqrt, inf, floor, diff, percentile, median, full_like, concatenate,
                   zeros_like, full, ones_like, s_, zeros, ndarray, transpose, squeeze)
from numpy.random import permutation

from muscat2ph.catalog import get_m2_coords
from muscat2ph.phdata import PhotometryData
from muscat2ta.m2lpf import M2LPF
from tqdm.auto import tqdm


def get_files(droot, target, night):
    ddata = droot.joinpath(target, night)
    files, pbs = [], []
    for pb in 'g r i z_s'.split():
        fname = ddata.joinpath(f'{target}_{night}_{pb}.nc')
        if fname.exists():
            files.append(fname)
            pbs.append(pb)
    return files, pbs

class TransitAnalysis:
    def __init__(self, target: str, date: str, tid: int, cids: list, dataroot: Path = None, exptime_min: float = 30.,
                 nlegendre: int = 0,  npop: int = 200,  mjd_start: float = -inf, mjd_end: float = inf,
                 aperture_lims: tuple = (0, inf), passbands: tuple = ('g', 'r', 'i', 'z_s'),
                 use_opencl: bool = False, with_transit: bool = True):

        self.target = target
        self.date = date
        self.tid = tid
        self.cids = list(cids)
        self.npop = npop
        self.etime = exptime_min
        self.models = None
        self.pbs = passbands

        self.nlegendre = nlegendre
        self.aperture_limits = aperture_lims
        self.use_opencl = use_opencl
        self.with_transit = with_transit

        # Define directories and names
        # ----------------------------
        self.datadir = datadir = Path(dataroot or 'photometry').joinpath(date)
        if not datadir.exists():
            raise IOError("Data directory doesn't exist.")
        self.basename = basename = f"{self.target}_{self.date}"

        self._dres = Path("results")
        self._dplot = Path("plots")
        self._dres.mkdir(exist_ok=True)
        self._dplot.mkdir(exist_ok=True)

        # Get the target coordinates
        # --------------------------
        self.target_coordinates = get_m2_coords(self.target)

        # Read in the data
        # ----------------
        files, pbs = get_files(dataroot, target, date)
        self.phs = [PhotometryData(f, tid, cids, objname=target, objskycoords=self.target_coordinates,
                                   mjd_start=mjd_start, mjd_end=mjd_end) for f in files]

        if len(self.phs) == 0:
            raise ValueError('No photometry files found.')

        self.lpf = M2LPF(target, self.phs, tid, cids, pbs, aperture_lims=aperture_lims, use_opencl=use_opencl,
                         with_transit=with_transit, n_legendre=nlegendre)

        self.pv = None


    def print_ptp_scatter(self):
        r1s = [res.std() for res in self.gplpf.residuals(self.gp_pv)]
        r2s = [(res - pre).std() for res, pre in zip(self.gplpf.residuals(self.gp_pv), self.gplpf.predict(self.gp_pv))]
        for r1, r2, pb in zip(r1s, r2s, 'g r i z'.split()):
            print('{} {:5.0f} ppm -> {:5.0f} ppm'.format(pb, 1e6 * r1, 1e6 * r2))

    def set_prior(self, i, p):
        self.lpf.set_prior(i, p)

    def add_t14_prior(self, mean: float, std: float):
        self.lpf.add_t14_prior(mean, std)

    def add_as_prior(self, mean: float, std: float):
        self.lpf.add_as_prior(mean, std)

    def add_ldtk_prior(self, teff: float, logg: float, z: float, uncertainty_multiplier: float = 3., pbs: tuple = ('g', 'r', 'i', 'z')):
        self.lpf.add_ldtk_prior(teff, logg, z, uncertainty_multiplier, pbs)

    def optimize(self, niter: int = 1000, pop: ndarray = None):
        self.lpf.optimize_global(niter, self.npop, pop, label='Optimizing the model')
        self.pv = self.lpf.de.minimum_location

    def sample(self, niter: int = 1000, thin: int = 5, repeats: int = None, reset=True):
        repeats = repeats or 1
        for i in tqdm(range(repeats)):
            self.lpf.sample_mcmc(niter, thin=thin, label='Sampling the model', reset=(reset or i != 0))

    def posterior_samples(self, burn: int = 0, thin: int = 1, include_ldc: bool = False) -> pd.DataFrame:
        return self.lpf.posterior_samples(burn, thin, include_ldc)

    def plot_mcmc_chains(self, pid: int = 0, alpha: float = 0.1, thin: int = 1, ax = None):
        return self.lpf.plot_mcmc_chains(pid, alpha, thin, ax)

    def plot_basic_posteriors(self, burn: int = 0, thin: int = 1):
        df = self.posterior_samples(burn, thin, False)
        df['k'] = sqrt(df.k2)
        df.drop(['k2'], axis=1, inplace=True)
        return corner(df)

    def plot_light_curve(self, method='de', figsize=(13, 8)):
        assert method in ('de', 'mc')
        fig, _ = self.lpf.plot_light_curves(model=method, figsize=figsize)
        return fig

    @property
    def savefile_name(self):
        return '{}_{}_fit.nc'.format(self.target, self.date)

    def load(self):
        ds = xa.open_dataset(self.savefile_name).load()
        ds.close()
        return ds

    def save(self):
        delm = None
        if self.lpf.de:
            delm = xa.DataArray(self.lpf.de.population, dims='pvector lm_parameter'.split(),
                                coords={'lm_parameter': self.lpf.ps.names})

        lmmc = None
        if self.lpf.sampler is not None:
            lmmc = xa.DataArray(self.lpf.sampler.chain, dims='pvector step lm_parameter'.split(),
                                coords={'lm_parameter': self.lpf.ps.names},
                                attrs={'ndim': self.lpf.de.n_par, 'npop': self.lpf.de.n_pop})

        ds = xa.Dataset(data_vars={'de_population_lm': delm, 'lm_mcmc': lmmc},
                        attrs={'created': strftime('%Y-%m-%d %H:%M:%S'),
                               'obsnight': self.date,
                               'target': self.target})

        ds.to_netcdf(self.savefile_name)

    def save_fits(self):
        phdu = pf.PrimaryHDU()
        phdu.header.update(target=self.target, night=self.date)
        hdul = pf.HDUList(phdu)

        lpf = self.lpf
        pv = lpf.de.minimum_location
        time = lpf.timea
        target_flux = lpf.target_flux(pv)
        reference_flux = lpf.reference_flux(pv)
        baseline = squeeze(lpf.baseline(pv))
        if lpf.with_transit:
            transit = lpf.transit_model(pv).astype('d')
        else:
            transit = ones_like(time)
        detrended_flux = target_flux / reference_flux / baseline

        for i, pb in enumerate(self.pbs):
            sl = lpf.lcslices[i]
            df = Table(transpose([time[sl], detrended_flux[sl], target_flux[sl],
                                     reference_flux[sl], baseline[sl], transit[sl]]),
                       names='time_bjd flux flux_trg flux_ref baseline model'.split(),
                       meta={'extname': f"flux_{pb}", 'filter': pb, 'trends': 'linear', 'wn': lpf.wn})
            hdul.append(pf.BinTableHDU(df))

        for i, pb in enumerate(self.pbs):
            df = Table(lpf.covariates[i], names='intercept sky xshift yshift entropy'.split(),
                       meta={'extname': f'aux_{pb}'})
            hdul.append(pf.BinTableHDU(df))
        hdul.writeto(self._dres.joinpath(f'{self.target}_{self.date}.fits'), overwrite=True)

    def plot_final_fit(self, model='linear', figwidth: float = 13):
        lpf = self.models[model]
        fig = figure(figsize=(figwidth, 1.4142*figwidth))
        if lpf.toi is None:
            figtext(0.05, 0.99, f"MuSCAT2 - {lpf.name}", size=33, weight='bold', va='top')
            figtext(0.05, 0.95, f"20{self.date[:2]}-{self.date[2:4]}-{self.date[4:]}", size=25, weight='bold', va='top')
        else:
            figtext(0.05, 0.99, f"MuSCAT2 - TOI {lpf.toi.toi}", size=33, weight='bold', va='top')
            figtext(0.05, 0.95, f"TIC {lpf.toi.tic}\n20{self.date[:2]}-{self.date[2:4]}-{self.date[4:]}", size=25, weight='bold',
                    va='top')

        # Light curve plots
        # -----------------
        figtext(0.05, 0.875, f"Raw light curve, model, and residuals", size=20, weight='bold', va='bottom')
        fig.add_axes((0.03, 0.87, 0.96, 0.001), facecolor='k', xticks=[], yticks=[])
        lpf.plot_light_curves(model='mc', fig=fig,
                              gridspec=dict(top=0.82, bottom=0.39, left=0.1, right=0.95, wspace=0.03, hspace=0.5))

        # Parameter posterior plots
        # -------------------------
        figtext(0.05, 0.325, f"Parameter posteriors", size=20, weight='bold', va='bottom')
        fig.add_axes((0.03, 0.32, 0.96, 0.001), facecolor='k', xticks=[], yticks=[])
        lpf.plot_posteriors(fig=fig, gridspec=dict(top=0.30, bottom=0.05, left=0.1, right=0.95, wspace=0.03, hspace=0.3))
        fig.add_axes((0.03, 0.01, 0.96, 0.001), facecolor='k', xticks=[], yticks=[])

        if lpf.toi is None:
            figname = f"{self.target}_{self.date}_transit_fit.pdf"
        else:
            figname = f"TIC{lpf.toi.tic}-{str(lpf.toi.toi).split('.')[1]}_20{self.date}_MuSCAT2_transit_fit.pdf"
        fig.savefig(self._dplot.joinpath(figname))
