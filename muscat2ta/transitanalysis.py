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
from pathlib import Path
from time import strftime

from astropy.stats import mad_std
from astropy.utils.exceptions import AstropyWarning
from matplotlib.axis import Axis
from numba import njit

warnings.simplefilter('ignore', category=AstropyWarning)

import pandas as pd
import xarray as xa
from astropy.io import fits as pf
from astropy.table import Table
from corner import corner
from matplotlib.pyplot import figure, figtext, setp, subplots
from muscat2ph.catalog import get_m2_coords, get_coords
from muscat2ph.phdata import PhotometryData
from numpy import (sqrt, inf, ones_like, ndarray, transpose, squeeze, atleast_1d, ceil, floor, array, isfinite, median,
                   nanmedian, arange, digitize, zeros, nan, full, clip)

from pytransit.param import NormalPrior as NP, UniformPrior as UP

from .m2lpf import M2LPF

__all__ = ["TransitAnalysis", "NP", "UP"]

@njit
def downsample_time(time, flux, inttime=30.):
    duration = 24. * 60. * 60. * (time.max() - time.min())
    nbins = int(ceil(duration / inttime))
    bins = arange(nbins)
    edges = time[0] + bins * inttime / 24 / 60 / 60
    bids = digitize(time, edges) - 1
    bt, bf, be = full(nbins, nan), zeros(nbins), zeros(nbins)
    for i, bid in enumerate(bins):
        bmask = bid == bids
        if bmask.sum() > 0:
            bt[i] = time[bmask].mean()
            bf[i] = flux[bmask].mean()
            if bmask.sum() > 2:
                be[i] = flux[bmask].std() / sqrt(bmask.sum())
            else:
                be[i] = nan
    m = isfinite(bt)
    return bt[m], bf[m], be[m]


def get_files(droot, target, night, passbands: tuple = ('g', 'r', 'i', 'z_s')):
    ddata = droot.joinpath(night)
    files, pbs = [], []
    for pb in passbands:
        fname = ddata.joinpath(f'{target}_{night}_{pb}.nc')
        if fname.exists():
            files.append(fname)
            pbs.append(pb)
    return files, pbs

class TransitAnalysis:
    def __init__(self, target: str, date: str, tid: int, cids: list, dataroot: Path = None, exptime_min: float = 30.,
                 nlegendre: int = 0,  npop: int = 200,  mjd_start: float = -inf, mjd_end: float = inf,
                 excluded_mjd_ranges: tuple = None,
                 aperture_lims: tuple = (0, inf), passbands: tuple = ('g', 'r', 'i', 'z_s'),
                 use_opencl: bool = False, with_transit: bool = True, with_contamination: bool = False,
                 radius_ratio: str = 'achromatic', klims=(0.005, 0.25),
                 catalog_name: str = None):

        self.target: str = target
        self.date: str = date
        self.tid: int = tid
        self.cids: list = list(cids)
        self.npop: int = npop
        self.etime: float = exptime_min
        self.pbs: tuple = passbands

        self.nlegendre = nlegendre
        self.aperture_limits = aperture_lims
        self.use_opencl = use_opencl
        self.with_transit = with_transit
        self.with_contamination = with_contamination
        self.toi = None

        self._old_de_population = None
        self._old_de_fitness = None

        # Define directories and names
        # ----------------------------
        self.dataroot = Path(dataroot or 'photometry')
        self.datadir = datadir = self.dataroot.joinpath(date)
        if not datadir.exists():
            raise IOError("Data directory doesn't exist.")
        self.basename = basename = f"{self.target}_{self.date}"

        self._dres = Path("results")
        self._dplot = Path("plots")
        self._dres.mkdir(exist_ok=True)
        self._dplot.mkdir(exist_ok=True)

        # Get the target coordinates
        # --------------------------
        self.target_coordinates = get_coords(catalog_name or self.target)

        # Read in the data
        # ----------------
        files, pbs = get_files(self.dataroot, target, date, passbands)
        self.phs = [PhotometryData(f, tid, cids, objname=target, objskycoords=self.target_coordinates,
                                   mjd_start=mjd_start, mjd_end=mjd_end, excluded_ranges=excluded_mjd_ranges) for f in files]

        if len(self.phs) == 0:
            raise ValueError('No photometry files found.')

        self.lpf = M2LPF(target, self.phs, tid, cids, pbs, aperture_lims=aperture_lims, use_opencl=use_opencl,
                         with_transit=with_transit, with_contamination=with_contamination,
                         n_legendre=nlegendre, radius_ratio=radius_ratio, klims=klims)
        if with_transit:
            self.lpf.set_prior(0, NP(self.lpf.times[0].mean(), 0.2*self.lpf.times[0].ptp()))

        self.pbs = self.lpf.passbands
        self.pv = None


    def print_ptp_scatter(self):
        r1s = [res.std() for res in self.gplpf.residuals(self.gp_pv)]
        r2s = [(res - pre).std() for res, pre in zip(self.gplpf.residuals(self.gp_pv), self.gplpf.predict(self.gp_pv))]
        for r1, r2, pb in zip(r1s, r2s, 'g r i z'.split()):
            print('{} {:5.0f} ppm -> {:5.0f} ppm'.format(pb, 1e6 * r1, 1e6 * r2))

    def apply_normalized_limits(self, iapt: int, lower: float = -inf, upper: float = inf, plot: bool = True,
                                apply: bool = True, npoly: int = 0, iterations: int = 5, erosion: int = 0) -> None:
        self.lpf.apply_normalized_limits(iapt, lower, upper, plot, apply, npoly, iterations, erosion)

    def downsample(self, exptime: float) -> None:
        self.lpf.downsample(exptime)

    def set_radius_ratio_prior(self, kmin, kmax):
        self.lpf.set_radius_ratio_prior(kmin, kmax)

    def set_prior(self, parameter, prior, *nargs) -> None:
        self.lpf.set_prior(parameter, prior, *nargs)

    def add_t14_prior(self, mean: float, std: float):
        self.lpf.add_t14_prior(mean, std)

    def add_as_prior(self, mean: float, std: float):
        self.lpf.add_as_prior(mean, std)

    def add_ldtk_prior(self, teff: tuple, logg: tuple, z: tuple, uncertainty_multiplier: float = 3., pbs: tuple = ('g', 'r', 'i', 'z')):
        self.lpf.add_ldtk_prior(teff, logg, z, uncertainty_multiplier, pbs)

    def freeze_photometry(self):
        self.lpf.freeze_photometry()

    def optimize(self, niter: int = 1000, pop: ndarray = None, plot_convergence: bool = True, plot_lc: bool = False):
        self.lpf.optimize_global(niter, self.npop, pop, label='Optimizing the model')
        self.pv = self.lpf.de.minimum_location

        if plot_lc:
            self.plot_light_curve()

    def sample(self, niter: int = 1000, thin: int = 5, repeats: int = 1, reset=True) -> None:
        self.lpf.sample_mcmc(niter, thin=thin, repeats=repeats, label='Sampling the model', reset=reset)

    def posterior_samples(self, burn: int = 0, thin: int = 1, derived_parameters=True) -> pd.DataFrame:
        return self.lpf.posterior_samples(burn, thin, derived_parameters=derived_parameters)

    def plot_mcmc_chains(self, pid: int = 0, alpha: float = 0.1, thin: int = 1, ax = None):
        return self.lpf.plot_mcmc_chains(pid, alpha, thin, ax)

    def plot_basic_posteriors(self, burn: int = 0, thin: int = 1):
        df = self.posterior_samples(burn, thin)
        return corner(df.iloc[:,:8])

    def plot_light_curve(self, method='de', figsize=(13, 8), save=False):
        assert method in ('de', 'mc')
        fig, _ = self.lpf.plot_light_curves(model=method, figsize=figsize)
        if save:
            fig.savefig(self._dplot.joinpath(f'{self.basename}_{self.lpf.radius_ratio}_k_lc.pdf'))
        return fig

    def plot_relative_light_curve(self, pbi: int, sid: int, cid: int, aid: int = -1,
                                  btime: float = 300., ax: Axis = None, figsize: tuple = None):
        """Plots a relative light curve `sid/cid` for passband `pbi`.

        Parameters
        ----------
        pbi: int
            Passband index
        sid: int
            Star index
        cid: int
            Comparison star index
        aid: int
            Aperture index
        btime: float
            Bin width for the overplotted binned flux
        ax: Axis
            The matplotlib Axis to plot the image
        figsize: tuple
            Figure size (only if no axis is given)

        Returns
        -------
            Axis

        """
        """Plot the raw flux of star `sid` normalized to the median of star `tid`."""
        fig, ax = subplots(figsize=figsize) if ax is None else (None, ax)

        ph = self.phs[pbi]
        time = ph.bjd
        t0 = floor(time[0])

        flux = array(ph.flux[:, sid, aid] / ph.flux[:, self.tid, aid].median())
        m = isfinite(time) & isfinite(flux)
        time, flux = time[m], flux[m]
        fratio = median(flux)
        flux /= array(ph.flux[:, cid, aid])
        flux /= median(flux)

        # Flux median and std for y limits and outlier marking
        # ----------------------------------------------------
        m = nanmedian(flux)
        s = mad_std(flux[isfinite(flux)])

        # Mark outliers
        # -------------
        mask = abs(flux - m) > 5 * s
        outliers = clip(flux[mask], m - 5.75 * s, m + 4.75 * s)

        # Plot the unbinned flux
        # ----------------------
        ax.plot(time - t0, flux, marker='.', ls='', alpha=0.1)
        ax.plot(time[mask] - t0, outliers, 'k', ls='', marker=6, ms=10)

        # Plot the binned flux
        # --------------------
        bt, bf, be = downsample_time(time, flux, btime)
        ax.plot(bt - t0, bf, 'k', drawstyle='steps-mid')

        try:
            ax.text(0.02, 1.01, f"Star {sid:d}, separation {self.distances[sid]:4.2f}'", size='small', ha='left',
                    va='bottom', transform=ax.transAxes)
        except AttributeError:
            pass
        ax.text(0.98, 1.01, f"F$_\star$/F$_0$ {fratio:4.3f}", size='small', ha='right', va='bottom',
                transform=ax.transAxes)
        setp(ax, xlim=time[[0, -1]] - t0, ylim=(m - 6 * s, m + 4 * s))
        return fig, ax

    @property
    def savefile_name(self):
        return f'{self.target}_{self.date}_{self.lpf.radius_ratio}_k'

    def load(self):
        ds = xa.open_dataset(self._dres.joinpath(self.savefile_name+'.nc')).load()
        ds.close()
        return ds

    def save(self):
        delm = None
        if self.lpf.de:
            dep = self.lpf.de.population.copy()
            dep[:,0] += self.lpf.tref
            delm = xa.DataArray(dep, dims='pvector lm_parameter'.split(), coords={'lm_parameter': self.lpf.ps.names})

        lmmc = None
        if self.lpf.sampler is not None:
            chain = self.lpf.sampler.chain.copy()
            chain[:,:,0] += self.lpf.tref
            lmmc = xa.DataArray(chain, dims='pvector step lm_parameter'.split(),
                                coords={'lm_parameter': self.lpf.ps.names},
                                attrs={'ndim': self.lpf.de.n_par, 'npop': self.lpf.de.n_pop})

        ds = xa.Dataset(data_vars={'de_population_lm': delm, 'lm_mcmc': lmmc},
                        attrs={'created': strftime('%Y-%m-%d %H:%M:%S'),
                               'obsnight': self.date,
                               'target': self.target})

        ds.to_netcdf(self._dres.joinpath(self.savefile_name+'.nc'))

    def save_fits(self):
        phdu = pf.PrimaryHDU()
        phdu.header.update(target=self.target, night=self.date)
        hdul = pf.HDUList(phdu)

        lpf = self.lpf
        pv = lpf.de.minimum_location
        time = lpf.timea

        baseline = squeeze(lpf.baseline(pv))
        if lpf.with_transit:
            transit = squeeze(lpf.transit_model(pv).astype('d'))
        else:
            transit = ones_like(time)

        if not self.lpf.photometry_frozen:
            target_flux = lpf.target_flux(pv)
            reference_flux = lpf.reference_flux(pv)
            relative_flux = target_flux / reference_flux
            detrended_flux = relative_flux / baseline
        else:
            target_flux = lpf._target_flux
            reference_flux = lpf._reference_flux
            relative_flux = self.lpf.ofluxa
            detrended_flux = relative_flux / baseline

        if self.lpf.cids.size == 0:
            reference_flux = ones_like(target_flux)

        for i, pb in enumerate(self.pbs):
            sl = lpf.lcslices[i]
            df = Table(transpose([time[sl] + self.lpf.tref, detrended_flux[sl], relative_flux[sl], target_flux[sl],
                                  reference_flux[sl], baseline[sl], transit[sl]]),
                       names='time_bjd flux flux_rel flux_trg flux_ref baseline model'.split(),
                       meta={'extname': f"flux_{pb}", 'filter': pb, 'trends': 'linear', 'wn': lpf.wn[i],
                             'radrat': lpf.radius_ratio})
            hdul.append(pf.BinTableHDU(df))

        for i, pb in enumerate(self.pbs):
            df = Table(lpf.covariates[i], names='sky xshift yshift entropy'.split(),
                       meta={'extname': f'aux_{pb}'})
            hdul.append(pf.BinTableHDU(df))
        hdul.writeto(self._dres.joinpath(self.savefile_name+'.fits'), overwrite=True)

    def plot_raw_light_curves(self, pb: int, sids: int, aids: int):
        sids = atleast_1d(sids)
        aids = atleast_1d(aids)
        nrows = int(ceil(sids.size / 2))
        ncols = min(sids.size, 2)

        fig, axs = subplots(nrows, ncols, figsize=(13, (4 if nrows == 1 else 2 * nrows)), constrained_layout=True,
                            sharex='all', squeeze=False)
        ph = self.phs[pb]
        for sid, aid, ax in zip(sids, aids, axs.flat):
            f = ph.flux[:, sid, aid]
            f /= f.median()
            ax.plot(ph.bjd - self.lpf.tref, f)
            ax.set_title(f"Star {sid}, aperture {aid}, scatter {float(f.diff('mjd').std('mjd') / sqrt(2)):.4f}")
            setp(ax, xlabel=f"Time - {self.lpf.tref:.0f} [d]", ylabel='Normalised flux')

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
