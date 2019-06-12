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


from pathlib import Path
from time import strftime

import astropy.units as u
import pandas as pd
import xarray as xa
from astropy.coordinates import SkyCoord, FK5
from astropy.io import fits as pf
from astropy.stats import mad_std
from astropy.visualization import simple_norm
from astropy.wcs import WCS
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import subplots, setp, figure, figtext
from muscat2ph.catalog import get_toi
from numba import njit
from numpy import sqrt, nan, zeros, digitize, sin, arange, ceil, where, isfinite, inf, ndarray, argsort, array, \
    floor, median, nanmedian, clip, mean, percentile, full, pi, concatenate, atleast_2d
from numpy.random import normal
from photutils import SkyCircularAperture
from tqdm.auto import tqdm

from .transitanalysis import TransitAnalysis

@njit
def as_from_dkp(d, p, k):
    """Assumes b=0"""
    return sqrt((1.0+k)**2) / sin(pi*d/p)

def true_depth(observed_depth, fratio):
    return (observed_depth * (1 + fratio)) / fratio

def true_radius_ratio(observed_depth, fratio):
    return sqrt(true_depth(observed_depth, fratio))

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


def tmodel(time, toi, fratio=None):
    tc = toi.epoch[0]
    p = toi.period[0]
    tn = round((time.mean() - tc) / p)
    time = time - (tc + tn * p)

    flux = where(abs(time) <= 0.5 * toi.duration[0] / 24, 1.0 - toi.depth[0] * 1e-6, 1.0)

    if fratio is None:
        return flux
    else:
        return 1 + ((flux - 1) * (1 + fratio)) / fratio


class TFOPAnalysis(TransitAnalysis):
    def __init__(self, target: str, date: str, tid: int, cids: list, dataroot: Path = None, exptime_min: float = 30.,
                 nlegendre: int = 0,  npop: int = 200,  mjd_start: float = -inf, mjd_end: float = inf,
                 aperture_lims: tuple = (0, inf), passbands: tuple = ('g', 'r', 'i', 'z_s'),
                 use_opencl: bool = False, with_transit: bool = True, with_contamination: bool = False):

        super().__init__(target, date, tid, cids, dataroot=dataroot, exptime_min=exptime_min,
                 nlegendre=nlegendre,  npop=npop,  mjd_start=mjd_start, mjd_end=mjd_end,
                 aperture_lims=aperture_lims, passbands=passbands,
                 use_opencl=use_opencl, with_transit=with_transit, with_contamination=with_contamination)

        # Get the TOI information
        # -----------------------
        self.toi = get_toi(float(target.lower().strip('toi')))
        self.ticname = 'TIC{:d}-{}'.format(self.toi.tic, str(self.toi.toi).split('.')[1])

        # Calculate star separations
        # --------------------------
        sc = SkyCoord(array(self.phs[0]._ds.centroids_sky), frame=FK5, unit=(u.deg, u.deg))
        self.distances = sc[tid].separation(sc).arcmin


    def create_example_frame(self, plot=True, figsize=(13, 13)):
        fits_files = list(self.datadir.glob('MCT*fits'))
        assert len(fits_files) > 0, 'No example frame fits-files found.'
        f = fits_files[0]
        f1 = pf.open(f)
        f2 = pf.open(f.with_suffix('.wcs'))
        h1 = f1[0].header.copy()
        h1.append(('COMMENT', '*********************'))
        h1.append(('COMMENT', '  WCS '))
        h1.append(('COMMENT', '*********************'))
        h1.extend(f2[0].header, unique=True, bottom=True)
        f1[0].header = h1
        filter = h1['filter']
        f1.writeto(self._dres.joinpath(f'{self.ticname}_20{self.date}_MuSCAT2_{filter}_frame.fits'), overwrite=True)

        if plot:
            wcs = WCS(f1[0].header)
            data = f1[0].data.astype('d')
            norm = simple_norm(data, stretch='log')
            fig = figure(figsize=figsize, constrained_layout=True)
            ax = fig.add_subplot(111, projection=wcs)
            ax.imshow(data, origin='image', norm=norm, cmap=cm.gray_r)
            apts = SkyCircularAperture(self.phs[0].centroids_sky,
                                       float(self.phs[0]._flux.aperture[self.lpf.aid]) * u.pixel)
            apts.to_pixel(wcs).plot(color='w')


    def plot_fit(self, model: str = 'de', figsize: tuple = (13, 8), save=False):
        fig, axs = self.lpf.plot_light_curves(model=model, figsize=figsize)
        ptype = 'fit' if model == 'de' else 'mcmc'
        if save:
            fig.savefig(self._dres.joinpath(f"{self.ticname}_20{self.date}_MuSCAT2_{ptype}.pdf"))
        return fig, axs

    def plot_possible_blends(self, ncols: int = 3, max_separation: float = 2.5, figwidth: float = 13,
                             axheight: float = 2.5, pbs: tuple = None, save: bool = True, close: bool = False):
        m = self.distances < max_separation
        sids = where(m)[0]
        stars = sids[argsort(self.distances[m])]
        nstars = len(stars)
        nrows = int(ceil(nstars / ncols))
        naxs = nrows * ncols
        nempty = naxs - nstars

        phs = self.phs if pbs is None else [self.phs[i] for i in pbs]
        passbands = self.passbands if pbs is None else [self.passbands[i] for i in pbs]
        plotname = self._dres.joinpath(f"{self.ticname}_20{self.date}_MuSCAT2_raw_lcs.pdf")
        pdf = PdfPages(plotname) if save else None

        for pb, ph in zip(passbands, phs):
            fig, axs = subplots(nrows, ncols, figsize=(figwidth, nrows * axheight), constrained_layout=True,
                                sharex='all')
            for i, istar in enumerate(tqdm(stars, leave=False)):
                self.plot_single_raw(axs.flat[i], ph, istar)
            [axs.flat[naxs - i - 1].remove() for i in range(nempty)]
            fig.suptitle(f"{self.ticname} 20{self.date} {pb}-band MuSCAT2 raw fluxes normalised to median target flux",
                         size='large')
            if save:
                pdf.savefig(fig)
            if close:
                fig.close()
        if save:
            pdf.close()

    def plot_single_raw(self, ax, ph, sid, aid=-1, btime=300., nsamples=500):
        """Plot the raw flux of star `sid` normalized to the median of star `tid`."""
        toi = self.toi
        flux = array(ph.flux[:, sid, aid] / ph.flux[:, self.tid, aid].median())
        time = ph.bjd
        t0 = floor(time[0])

        m = isfinite(time) & isfinite(flux)
        time, flux = time[m], flux[m]
        fratio = median(flux)
        flux /= fratio

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

        # Plot the model
        # --------------
        if fratio > toi.depth[0] * 1e-6:
            fmodel = tmodel(time, toi, None if sid == self.tid else fratio) - 4 * s
            ax.plot(time - t0, fmodel, 'k', ls='-', alpha=0.5)

        # Transit centre, ingress, and egress
        # -----------------------------------
        tn = round((mean(ph.bjd) - toi.epoch[0]) / toi.period[0])
        center = normal(*toi.epoch, size=nsamples) + tn * normal(*toi.period, size=nsamples) - t0
        ax.axvspan(*percentile(center, [16, 84]), alpha=0.15)
        ax.axvline(median(center), lw=1)

        ingress = center - 0.5 * normal(*toi.duration / 24, size=nsamples)
        ax.axvspan(*percentile(ingress, [16, 84]), alpha=0.25, ymin=0.95, ymax=1.)
        ax.axvline(median(ingress), ymin=0.97, ymax=1., lw=1)

        egress = center + 0.5 * normal(*toi.duration / 24, size=nsamples)
        ax.axvspan(*percentile(egress, [16, 84]), alpha=0.25, ymin=0.95, ymax=1.)
        ax.axvline(median(egress), ymin=0.97, ymax=1., lw=1)

        ax.text(0.02, 1.01, f"Star {sid:d}, separation {self.distances[sid]:4.2f}'", size='small', ha='left',
                va='bottom', transform=ax.transAxes)
        ax.text(0.98, 1.01, f"F$_\star$/F$_0$ {fratio:4.3f}", size='small', ha='right', va='bottom',
                transform=ax.transAxes)
        setp(ax, yticks=[], xlim=time[[0, -1]] - t0, ylim=(m - 6 * s, m + 4 * s))


    def plot_final_fit(self, figwidth: float = 13, save: bool = True, close: bool = False) -> None:
        lpf = self.lpf
        fig = figure(figsize = (figwidth, 1.4142*figwidth))
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
        plotname = self._dres.joinpath(f"{self.ticname}_20{self.date}_MuSCAT2_transit_fit.pdf")

        if save:
            fig.savefig(plotname)
        if close:
            fig.close()

    def plot_covariates(self, figsize=(13, 5), close=False):
        cols = 'Sky, Airmass, X Shift [pix], Y Shift [pix], Aperture entropy'.split(',')
        with PdfPages(self._dres.joinpath(f"{self.ticname}_20{self.date}_MuSCAT2_covariates.pdf")) as pdf:
            for ipb, ph in enumerate(self.phs):
                fig, axs = subplots(2, 2, figsize=figsize, sharex=True)
                aux, time = ph.aux, ph.bjd
                t0 = floor(time.min())
                for i in range(4):
                    ax = axs.flat[i]
                    ax.plot(time - t0, aux[:, i + 1], 'k')
                    setp(ax, ylabel=cols[i])
                setp(axs, xlim=time[[0, -1]] - t0)
                setp(axs[-1, :], xlabel='Time - {:7.0f} [BJD]'.format(t0))
                fig.suptitle('{}  20{} MuSCAT2 {} covariates'.format(self.ticname, self.date, self.passbands[ipb]))
                fig.tight_layout()
                fig.subplots_adjust(hspace=0, top=0.9, bottom=0.01)
                pdf.savefig(fig)
                if close:
                    fig.close()

    def export_tables(self):
        for i, ph in enumerate(self.phs):
            data = concatenate([atleast_2d(ph.bjd).T,
                                array(ph.flux[:, :, ph._rset.tap]),
                                array(ph.aux[:, 1:])], axis=1)
            df = pd.DataFrame(data, columns=['BJD_TDB'] + [f'flux_star_{i}' for i in range(ph.flux.shape[1])] + ['sky',
                                                                                                           'airmass',
                                                                                                           'xshift',
                                                                                                           'yshift',
                                                                                                           'sky_entropy'])
            df.to_csv(self._dres.joinpath(f'{self.ticname}_20{self.date}_MuSCAT2_{self.passbands[i]}_measurements.tbl'),
                      index=False, sep=" ")

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

        ds = xa.Dataset(data_vars={'de_population': delm, 'mcmc': lmmc},
                        attrs={'created': strftime('%Y-%m-%d %H:%M:%S'),
                               'obsnight': self.date,
                               'target': self.target})

        ds.to_netcdf(self.savefile_name)