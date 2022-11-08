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
from typing import Optional

from astropy.utils.exceptions import AstropyDeprecationWarning
import warnings
warnings.simplefilter('ignore', category = AstropyDeprecationWarning)

import astropy.units as u
import pandas as pd
import xarray as xa
from astropy.coordinates import SkyCoord, FK5
from astropy.io import fits as pf
from astropy.nddata import Cutout2D
from astropy.stats import mad_std
from astropy.visualization import simple_norm
from astropy.wcs import WCS
from astroquery.mast import Catalogs
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import subplots, setp, figure, figtext, close
from pytransit.orbits import epoch
from uncertainties import ufloat

from muscat2ph.catalog import get_toi, get_toi_or_tic_coords
from numba import njit
from numpy import sqrt, nan, zeros, digitize, sin, arange, ceil, where, isfinite, inf, ndarray, argsort, array, \
    floor, median, nanmedian, clip, mean, percentile, full, pi, concatenate, atleast_2d, ones, asarray
from numpy.random import normal
from photutils import SkyCircularAperture, CircularAperture
from tqdm.auto import tqdm

from pytransit import NormalPrior as NP, UniformPrior as UP

from .transitanalysis import TransitAnalysis, downsample_time


@njit
def as_from_dkp(d, p, k):
    """Assumes b=0"""
    return sqrt((1.0+k)**2) / sin(pi*d/p)


def true_depth(observed_depth, fratio):
    return (observed_depth * (1 + fratio)) / fratio


def true_radius_ratio(observed_depth, fratio):
    return sqrt(true_depth(observed_depth, fratio))


def tmodel(time, toi, depth):
    tc = toi.epoch[0]
    p = toi.period[0]
    tn = round((time.mean() - tc) / p)
    time = time - (tc + tn * p)
    return where(abs(time) <= 0.5 * toi.duration[0] / 24, 1.0 - depth, 1.0)



class TFOPAnalysis(TransitAnalysis):
    def __init__(self, target: str, date: str, tid: int, cids: list, dataroot: Path = None,
                 nlegendre: int = 0,  npop: int = 200,  mjd_start: float = -inf, mjd_end: float = inf,
                 excluded_mjd_ranges: tuple = None,
                 aperture_lims: tuple = (0, inf), passbands: tuple = ('g', 'r', 'i', 'z_s'),
                 use_opencl: bool = False, with_transit: bool = True, with_contamination: bool = False,
                 radius_ratio: str = 'chromatic', excluded_stars=(), toi=None, klims=(0.005, 0.25),
                 clear_field_only: bool = False, check_saturation: bool = True,
                 contamination_model: str = 'physical',
                 contamination_reference_passband: Optional[str] = None,
                 files=None, pbs=None):

        super().__init__(target, date, tid, cids, dataroot=dataroot,
                 nlegendre=nlegendre,  npop=npop,  mjd_start=mjd_start, mjd_end=mjd_end,
                 excluded_mjd_ranges=excluded_mjd_ranges,
                 aperture_lims=aperture_lims, passbands=passbands,
                 use_opencl=use_opencl, with_transit=with_transit, with_contamination=with_contamination,
                 radius_ratio=radius_ratio, klims=klims, init_lpf=(not clear_field_only), check_saturation=check_saturation,
                 contamination_model=contamination_model, contamination_reference_passband=contamination_reference_passband,
                         files=files, pbs=pbs)

        # Get the TOI information
        # -----------------------
        if toi is not None:
            self.toi = toi
            self.ticname = 'TIC{:d}-{}'.format(int(self.toi.tic), str(self.toi.toi).split('.')[1])
            if self.lpf:
                self.lpf.toi = self.toi
        else:
            try:
                self.toi = get_toi(float(target.lower().strip('toi')))
                self.ticname = 'TIC{:d}-{}'.format(int(self.toi.tic), str(self.toi.toi).split('.')[1])
                if self.lpf:
                    self.lpf.toi = self.toi
            except ValueError:
                warnings.warn(f"Couldn't identify {target.lower()} and no TOI given")

        #TODO: Clean passbands vs pbs
        if self.lpf:
            self.passbands = self.lpf.passbands
        else:
            self.passbands = self.pbs

        self.excluded_stars = excluded_stars

        # Set priors
        # ----------
        self.t0 = ufloat(*self.toi.epoch)
        self.pr = ufloat(*self.toi.period)
        self.t14 = ufloat(*self.toi.duration/24)
        if self.lpf:
            self.ep = epoch(self.lpf.timea.mean(), self.t0.n, self.pr.n)
            self.tc = self.t0 + self.ep * self.pr
            tmin, tmax = self.lpf.timea.min(), self.lpf.timea.max()
        else:
            self.ep = epoch(array(self.phs[0]._bjd).mean(), self.t0.n, self.pr.n)
            self.tc = self.t0 + self.ep * self.pr
            tmin, tmax = array(self.phs[0]._bjd).min(), array(self.phs[0]._bjd).max()

        self.transit_start = self.tc - 0.5 * ufloat(*self.toi.duration / 24)
        self.transit_center = self.tc
        self.transit_end = self.tc + 0.5 * ufloat(*self.toi.duration / 24)

        # Calculate transit probabilities
        # -------------------------------
        ns = 2000
        tc_samples = normal(self.tc.n, self.tc.s, ns)
        td_samples = normal(self.t14.n, self.t14.s, ns)

        d_ingress_in_window = (tmin < tc_samples - 0.5 * td_samples) & (tc_samples - 0.5 * td_samples < tmax)
        d_egress_in_window = (tmin < tc_samples + 0.5 * td_samples) & (tc_samples + 0.5 * td_samples < tmax)
        d_full_transit_in_window = d_ingress_in_window & d_egress_in_window
        d_transit_before_window = tmin > tc_samples + 0.5 * td_samples
        d_transit_after_window = tmax < tc_samples - 0.5 * td_samples
        d_transit_covers_whole_window = (tmin > tc_samples - 0.5 * td_samples) & (tc_samples + 0.5 * td_samples > tmax)

        self.p_transit_not_in_window = d_transit_before_window.mean() + d_transit_after_window.mean()
        self.p_full_transit_in_window = d_full_transit_in_window.mean()
        self.p_ingress_in_window = d_ingress_in_window.mean()
        self.p_egress_in_window = d_egress_in_window.mean()
        self.p_transit_covers_whole_window = d_transit_covers_whole_window.mean()

        # Set priors
        # ----------
        if self.lpf:
            self.set_prior('tc', NP(self.tc.n, self.tc.s))
            self.set_prior('p', NP(*self.toi.period))
            self.add_t14_prior(self.toi.duration[0] / 24, 0.5*self.toi.duration[1] / 24)
            self.lpf.add_inside_window_prior()
            for p in self.lpf.ps[self.lpf._sl_k2]:
                p.prior = UP(0.25 * self.toi.depth[0] * 1e-6, 4 * self.toi.depth[0] * 1e-6)
            self.lpf.aid = 4

        self.distances = self.phs[0].distances_arcmin

    def create_example_frame(self, plot=True, figsize=(12, 12), markersize=25, loffset=0):
        fits_files = list(self.datadir.glob('*_frame.fits'))
        assert len(fits_files) > 0, 'No example frame fits-files found.'

        if plot and f.with_suffix('.wcs').exists():
            wcs = WCS(f1[0].header)
            data = f1[0].data.astype('d')
            fig = self.plot_tfop_field(data, wcs, figsize, markersize, loffset, subfield_radius=None)
            fig.savefig(self._dres.joinpath(f"{self.ticname}_{self.date}_MuSCAT2_{filter}_frame.pdf"))
            fig = self.plot_tfop_field(data, wcs, figsize, markersize, loffset, subfield_radius=2.5)
            fig.savefig(self._dres.joinpath(f"{self.ticname}_{self.date}_MuSCAT2_{filter}_frame_5_arcmin.pdf"))
            close(fig)
            fig = self.plot_tfop_field(data, wcs, figsize, markersize, loffset, subfield_radius=1)
            fig.savefig(self._dres.joinpath(f"{self.ticname}_{self.date}_MuSCAT2_{filter}_frame_2_arcmin.pdf"))
            close(fig)

    def plot_tfop_field(self, data, wcs=None, figsize=(9, 9), markersize=25, loffset=0, subfield_radius=None):
        sc = self.target_coordinates or get_toi_or_tic_coords(self.toi.tic)

        if subfield_radius:
            r = subfield_radius * u.arcmin
            co = Cutout2D(data, sc, (2 * r, 2 * r), mode='partial', fill_value=median(data), wcs=wcs)
            data, wcs = co.data, co.wcs

        height, width = data.shape

        ct = Catalogs.query_region(sc, radius=0.033, catalog="Gaia", version=2).to_pandas()

        mean_flux = ct.phot_g_mean_flux.values
        max_depths = 1e6 * mean_flux[1:] / (mean_flux[1:] + mean_flux[0])
        depth_mask = max_depths > self.toi.depth[0]

        if wcs is not None:
            sc_all = SkyCoord(ct.ra.values * u.deg, ct.dec.values * u.deg)
            xy_all = sc_all.to_pixel(wcs)
        else:
            xy_all = self.phs[0].centroids_pix

        fig = figure(figsize=figsize, constrained_layout=True)
        ax = fig.add_subplot(111, projection=wcs)
        ax.grid()

        # Plot the image
        # --------------
        norm = simple_norm(data, stretch='log', min_percent=40, max_percent=100)
        ax.imshow(data, origin='image', norm=norm, cmap=cm.gray_r)

        # Plot apertures
        # --------------
        amask = ones(self.phs[0].centroids_pix.shape[0], bool)
        if self.excluded_stars:
            amask[self.excluded_stars] = 0

        if wcs is not None:
            apts = SkyCircularAperture(self.phs[0].centroids_sky[amask],
                                       float(self.phs[0]._flux.aperture[self.lpf.aid]) * u.pixel)
            apts.to_pixel(wcs).plot(color='k', linestyle='--')
            apts_xy = apts.to_pixel(wcs)
        else:
            apts_xy = CircularAperture(xy_all[amask], float(self.phs[0]._flux.aperture[self.lpf.aid]))

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        for istar, (x, y) in enumerate(apts_xy.positions[1:]):
            if (xlim[0] <= x <= xlim[1]) and (ylim[0] <= y <= ylim[1]) and istar not in self.excluded_stars:
                ax.text(x + loffset + sqrt(2 * markersize), y + loffset + sqrt(2 * markersize), istar + 1,
                        size='larger')

        # Plot the circle of inclusion
        # ----------------------------
        ci = CircularAperture(xy_all[0], 1 / (0.44 * u.arcsec).to(u.arcmin).value)
        ci.plot(ax=ax, color='0.4', ls=':', lw=2, alpha=0.5)
        if ci.positions[0] + ci.r + 10 < xlim[1]:
            ax.text(ci.positions[0] + ci.r + 10, ci.positions[1], "r = 1'", size='larger')

        ci = CircularAperture(xy_all[0], 2 / (0.44 * u.arcsec).to(u.arcmin).value)
        ci.plot(ax=ax, color='0.4', ls=':', lw=2)
        if ci.positions[0] + ci.r + 10 < xlim[1]:
            ax.text(ci.positions[0] + ci.r + 10, ci.positions[1], "r = 2'", size='larger')

        # Plot the target
        # ---------------
        CircularAperture(xy_all[0], 0.9 * markersize).plot(color='k', lw=2)
        CircularAperture(xy_all[0], 1.1 * markersize).plot(color='k', lw=2)

        if wcs is not None:
            # Plot the possible contaminants
            # ------------------------------
            ap_cnt = SkyCircularAperture(sc_all[1:][depth_mask], markersize * u.pixel)
            ap_cnt.to_pixel(wcs).plot(color='k')

            # Plot the stars inside the 2' radius but too faint to create the transit signal
            # ------------------------------------------------------------------------------
            ap_rest = SkyCircularAperture(sc_all[1:][~depth_mask], markersize * u.pixel)
            ap_rest.to_pixel(wcs).plot(color='k', linestyle=':')
        else:
            CircularAperture(xy_all[1:], markersize).plot(color='k', lw=1, ls='--')

        # Plot the image scale
        # --------------------
        xlims = ax.get_xlim()
        py = 0.96 * ax.get_ylim()[1]
        x0, x1 = 0.3 * xlims[1], 0.7 * xlims[1]
        if wcs is not None:
            scale = SkyCoord.from_pixel(x0, py, wcs).separation(SkyCoord.from_pixel(x1, py, wcs))
            ax.annotate('', xy=(x0, py), xytext=(x1, py), arrowprops=dict(arrowstyle='|-|', lw=1.5, color='k'))
            ax.text(width / 2, py - 7.5, "{:3.1f}'".format(scale.arcmin), va='top', ha='center', size='larger')
        ax.set_title(
            f"MuSCAT2 TIC {int(self.toi.tic)} (TOI {self.toi.toi}) {self.date[-2:]}.{self.date[2:4]}.20{self.date[:2]}",
            size='x-large')
        ax.set_ylabel('Dec', size='x-large')
        ax.set_xlabel('RA', size='x-large')
        fig.subplots_adjust(bottom=0.03, top=0.98, left=0.1, right=0.98)
        return fig

    def plot_fit(self, model: str = 'de', figsize: tuple = (13, 8), save=False, plot_priors=True):
        fig, axs = self.lpf.plot_light_curves(model=model, figsize=figsize)
        npb = self.lpf.npb
        if plot_priors:
            def plot_vprior(t, ymin: float = 0, ymax: float = 1):
                [[ax.axvspan(t.n - s * t.s, t.n + s * t.s, alpha=0.1, ymin=ymin, ymax=ymax) for s in (1, 2, 3)] for ax
                 in fig.axes[npb:]]
                [ax.axvline(t.n, ymin=ymin, ymax=ymax) for ax in fig.axes[npb:]]

            plot_vprior(self.transit_center - self.lpf._tref, 0.03, 0.9)
            plot_vprior(self.transit_start - self.lpf._tref, 0.93, 0.98)
            plot_vprior(self.transit_end - self.lpf._tref, 0.93, 0.98)

            [ax.axhline(1 - self.toi.depth[0] * 1e-6, ls=':') for ax in fig.axes[npb:3 * npb]]

        ptype = 'fit' if model == 'de' else 'mcmc'
        if save:
            fig.savefig(self._dres.joinpath(f"{self.ticname}_{self.date}_MuSCAT2_{ptype}.pdf"))
        return fig, axs

    def plot_possible_blends(self, cid: int, aid: int, caid: int = None, stars: list = None, c_flux_factor: float = None,
                             ncols: int = 3, nrows: int = 4,
                             max_separation: float = 2.5, figwidth: float = 13, axheight: float = 2.5,
                             pbs: tuple = None, save: bool = True, close_figures: bool = False) -> None:
        """
        Plots the normalised relative fluxes for the possible contaminant stars.

        Plots the normalised relative fluxes for stars within `max_separation` (in arcmin) radius of the target star.
        Also plots the expected times for ingress, egress, and transit centre, and the transit depth required to cause
        the observed signal if blended with the target star.

        Parameters
        ----------
        cid: int
            Comparison star ID.
        aid: int
            Aperture ID.
        stars: array-like, optional
            Optional list of star IDs to include into the plot.
        ncols: int, optional
            Number of columns per page.
        nrows: int, optional
            Number of rows per page.
        max_separation: float, optional
            Maximum separation in arcmin.
        figwidth: float, optional
            Figure width.
        axheight: float, optional
            Height of a single axis.
        pbs: array-like, optional
            Passbands
        save: bool, optional
            Save the figures if `True`
        close_figures: bool, optional
            Close the figures if `True`

        Returns
        -------
            None
        """
        caid = caid if caid is not None else aid

        if stars is None:
            m_excl = ones(self.distances.size, bool)
            if self.excluded_stars:
                m_excl[list(self.excluded_stars)] = 0
            m = (self.distances < max_separation) & m_excl
            sids = where(m)[0]
            stars = sids[argsort(self.distances[m])]
        stars = asarray(stars)

        phs = self.phs if pbs is None else [self.phs[i] for i in pbs]
        passbands = self.passbands if pbs is None else [self.passbands[i] for i in pbs]

        nstars = len(stars)
        stars_per_page = nrows * ncols

        for pbi, (pb, ph) in enumerate(zip(passbands, phs)):
            plotname = self._dres.joinpath(f"{self.ticname}_{self.date}_MuSCAT2_{pb}_possible_blends.pdf")
            pdf = PdfPages(plotname) if save else None

            if c_flux_factor is not None:
                ref_flux = c_flux_factor * nanmedian(ph._flux[:,cid,aid])
            else:
                ref_flux = None

            for i, istar in enumerate(tqdm(stars, leave=False)):
                if i % stars_per_page == 0:
                    fig, axs = subplots(nrows, ncols, figsize=(figwidth, nrows * axheight), sharex='all')
                    iax = 0
                try:
                    self.plot_single_raw(pbi, istar, cid=cid, taid=aid, caid=caid, ax=axs.flat[iax], reference_flux=ref_flux)
                except:
                    pass
                iax += 1

                if ((i+1) % stars_per_page == 0) or i == nstars - 1:
                    setp(axs[:,0], ylabel='Normalized flux')
                    fig.suptitle(f"{self.ticname} 20{self.date} {pb}-band MuSCAT2 relative normalised fluxes", size='large')
                    fig.tight_layout(rect=(0, 0, 1, 0.95))
                    if i == nstars - 1:
                        for j in range(stars_per_page - iax):
                            axs.flat[iax+j].remove()
                    if save:
                        pdf.savefig(fig)
                    if close_figures:
                        close(fig)
            if save:
                pdf.close()

    def plot_single_raw(self, phi: int, sid: int, cid: int, taid: int = -1, caid: int = -1, btime=300., nsamples=500,
                        reference_flux: float = None, ax=None):
        """Plot the raw flux of star `sid` normalized to the median of star `tid`."""

        fig, ax = subplots() if ax is None else (None, ax)
        ph = self.phs[phi]
        toi = self.toi

        m = isfinite(ph._bjd) & isfinite(ph._flux[:,sid,taid]) & isfinite(ph._flux[:,cid,caid]) & ph.exclusion_mask
        time = ph._bjd[m]
        t0 = floor(time[0])
        flux = array(ph._flux[m, sid, taid] / ph._flux[m, cid, caid])
        flux /= median(flux)

        if reference_flux is not None:
            target_flux = reference_flux
        else:
            target_flux = nanmedian(ph._flux[m, self.tid, taid])
            if not isfinite(target_flux):
                print("WARNING: The target is completely saturated, blending analysis will fail")
                target_flux = 1.

        fratio = nanmedian(array(ph._flux[m, sid, taid])) / target_flux

        # Mark outliers
        # -------------
        s = mad_std(flux)
        mask = abs(flux - 1) > 5 * s
        outliers = clip(flux[mask], 1 - 5.75 * s, 1 + 4.75 * s)

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
        depth = 1e-6 * toi.depth[0] / fratio
        ep = epoch(time, toi.epoch[0], toi.period[0])
        timec = time - (toi.epoch[0] + ep * toi.period[0])
        fmodel = where(abs(timec) <= 0.5 * toi.duration[0] / 24, 1.0 - depth, 1.0) - 3 * s
        ax.plot(time - t0, fmodel, 'k', ls='-', alpha=0.5)

        # Transit centre, ingress, and egress
        # -----------------------------------
        tn = round((mean(ph._bjd) - toi.epoch[0]) / toi.period[0])
        center = normal(*toi.epoch, size=nsamples) + tn * normal(*toi.period, size=nsamples) - t0
        ax.axvspan(*percentile(center, [16, 84]), alpha=0.15)
        ax.axvline(median(center), lw=1)

        ingress = center - 0.5 * normal(*toi.duration / 24, size=nsamples)
        ax.axvspan(*percentile(ingress, [16, 84]), alpha=0.25, ymin=0.95, ymax=1.)
        ax.axvline(median(ingress), ymin=0.97, ymax=1., lw=1)

        egress = center + 0.5 * normal(*toi.duration / 24, size=nsamples)
        ax.axvspan(*percentile(egress, [16, 84]), alpha=0.25, ymin=0.95, ymax=1.)
        ax.axvline(median(egress), ymin=0.97, ymax=1., lw=1)

        ax.text(0.02, 1.01, f"Star {sid:d} -- separation {self.distances[sid]:4.2f}'", size='small', ha='left',
                va='bottom', transform=ax.transAxes)
        ax.text(0.98, 1.01, f"$F_\star \;/\;F_0\;= $ {fratio:4.3f} -- $\Delta F = {depth:6.5f}$", size='small',
                ha='right', va='bottom',
                transform=ax.transAxes)
        setp(ax, xlim=time[[0, -1]] - t0, ylim=(1 - 6 * s, 1 + 4 * s))

    def plot_final_fit(self, figwidth: float = 13, save: bool = True, close: bool = False) -> None:

        def bic_evidence(bic):
            if bic < 2:
                return "no"
            elif 2 < bic < 6:
                return "positive"
            elif 6 < bic < 10:
                return "strong"
            else:
                return "very strong"

        lpf = self.lpf
        fig = figure(figsize=(figwidth, 1.4142 * figwidth))
        if self.toi is None:
            figtext(0.05, 0.99, f"MuSCAT2 - {lpf.name}", size=33, weight='bold', va='top')
            figtext(0.05, 0.95, f"20{self.date[:2]}-{self.date[2:4]}-{self.date[4:]}", size=25, weight='bold', va='top')
        else:
            figtext(0.05, 0.99, f"MuSCAT2 - TOI {self.toi.toi}", size=33, weight='bold', va='top')
            figtext(0.05, 0.95, f"TIC {self.toi.tic}\n20{self.date[:2]}-{self.date[2:4]}-{self.date[4:]}", size=25,
                    weight='bold',
                    va='top')

        # Light curve plots
        # -----------------
        figtext(0.05, 0.875, f"Raw light curve, model, and residuals", size=20, weight='bold', va='bottom')
        fig.add_axes((0.03, 0.87, 0.96, 0.001), facecolor='k', xticks=[], yticks=[])
        lpf.plot_light_curves(model='mc', fig=fig,
                              gridspec=dict(top=0.82, bottom=0.39, left=0.1, right=0.95, wspace=0.03, hspace=0.5))

        def plot_vprior(t, ymin: float = 0, ymax: float = 1):
            [[ax.axvspan(t.n - s * t.s, t.n + s * t.s, alpha=0.1, ymin=ymin, ymax=ymax) for s in (1, 2, 3)] for ax
             in fig.axes[1 + self.lpf.npb:]]
            [ax.axvline(t.n, ymin=ymin, ymax=ymax) for ax in fig.axes[1 + self.lpf.npb:]]

        plot_vprior(self.transit_center - self.lpf._tref, 0.03, 0.9)
        plot_vprior(self.transit_start - self.lpf._tref, 0.93, 0.98)
        plot_vprior(self.transit_end - self.lpf._tref, 0.93, 0.98)

        # Transit depths
        # --------------
        [ax.axhline(1.0 - self.toi.depth[0] * 1e-6, ls=':') for ax in fig.axes[1 + self.lpf.npb:1 + 3*self.lpf.npb]]

        bic = self.lpf.transit_bic()

        # Parameter posterior plots
        # -------------------------
        figtext(0.05, 0.325,
                f"Combined LC and posteriors {'' if bic > 6 else '(WARNING: no strong evidence for a transit)'}",
                size=20, weight='bold', va='bottom')
        fig.add_axes((0.03, 0.32, 0.96, 0.001), facecolor='k', xticks=[], yticks=[])
        lpf.plot_posteriors(fig=fig,
                            gridspec=dict(top=0.30, bottom=0.05, left=0.1, right=0.95, wspace=0.03, hspace=0.3))
        fig.add_axes((0.03, 0.01, 0.96, 0.001), facecolor='k', xticks=[], yticks=[])
        plotname = self._dres.joinpath(f"{self.ticname}_{self.date}_MuSCAT2_transit_fit.pdf")

        ptext = (f"P(full transit in window) = {self.p_full_transit_in_window:4.2f},\n"
                 f"P(ingress) = {self.p_ingress_in_window:4.2f},  "
                 f"P(egress) = {self.p_egress_in_window:4.2f},\n"
                 f"P(misses window) = {self.p_transit_not_in_window:4.2f},  "
                 f"P(spans window) = {self.p_transit_covers_whole_window:4.2f},\n"
                 f"Transit model BIC: {bic:.2f} $\longmapsto$ {bic_evidence(bic)} evidence for a transit in the data")
        figtext(0.32, 0.95, ptext, size=16, va='top', ha='left')

        fig.axes[2+4*self.lpf.npb].cla()
        self.lpf.plot_combined_and_binned(ax=fig.axes[2+4*self.lpf.npb], plot_unbinned=False)

        if save:
            fig.savefig(plotname)
        if close:
            fig.close()

    def plot_covariates(self, figsize=(13, 5), close=False):
        cols = 'Sky, Airmass, X Shift [pix], Y Shift [pix], Aperture entropy'.split(',')
        with PdfPages(self._dres.joinpath(f"{self.ticname}_{self.date}_MuSCAT2_covariates.pdf")) as pdf:
            for ipb, ph in enumerate(self.phs):
                fig, axs = subplots(2, 2, figsize=figsize, sharex=True)
                aux, time = ph.nmaux, ph._bjd
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
            data = concatenate([atleast_2d(ph._bjd[ph.exclusion_mask]).T,
                                array(ph._flux[ph.exclusion_mask, :, ph._rset.tap]),
                                array(ph.nmaux[ph.exclusion_mask, 1:])], axis=1)
            df = pd.DataFrame(data, columns=['BJD_TDB'] + [f'flux_star_{i}' for i in range(ph.flux.shape[1])] + ['sky',
                                                                                                           'airmass',
                                                                                                           'xshift',
                                                                                                           'yshift',
                                                                                                           'sky_entropy'])
            df.to_csv(self._dres.joinpath(f'{self.ticname}_{self.date}_MuSCAT2_{self.passbands[i]}_measurements.tbl'),
                      index=False, sep=" ")

    def finalize(self):
        from pathlib import Path
        from string import Template
        from shutil import copy2
        import muscat2ta
        dsubmit = Path('submit')
        dsubmit.mkdir(exist_ok=True)

        files = list(self.datadir.glob('*.png'))
        for src_path in files:
            trg_path = dsubmit.joinpath(src_path.name.replace(self.target, self.ticname))
            copy2(src_path, trg_path)

        files = list(self.datadir.glob('*frame.fits'))
        for src_path in files:
            trg_path = dsubmit.joinpath(src_path.name)
            copy2(src_path, trg_path)

        files = list(self._dres.glob('*.pdf')) + list(self._dres.glob('*.tbl'))
        for src_path in files:
            trg_path = dsubmit.joinpath(src_path.name)
            copy2(src_path, trg_path)

        readme = (Path(muscat2ta.__path__[0]) / '..' / 'data' / 'templates' / 'tfop_readme.txt').resolve()
        report = (Path(muscat2ta.__path__[0]) / '..' / 'data' / 'templates' / 'tfop_report.txt').resolve()

        copy2(readme, dsubmit / f"{self.ticname}_{self.date}_MuSCAT2_readme.txt")

        try:
            tap, raps = self.lpf.frozen_apertures
        except (IndexError, AttributeError):
            tap, raps = '-', '-'

        df = self.posterior_samples()

        with open(dsubmit / f"{self.ticname}_{self.date}_MuSCAT2_report.txt", "w") as f:
            t = Template(report.read_text())
            f.write(t.safe_substitute(ticname=self.ticname, night=self.date, tid=self.tid, cids=self.cids,
                                      tap=tap, raps=raps,
                                      tc=round(df.tc.median(), 7), tce=round(df.tc.std(), 7),
                                      t14=round(df.t14.median(),5), t14e=round(df.t14.std(), 5)))

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
                               'target': self.target,
                               'tref': self.lpf._tref})

        ds.to_netcdf(self._dres.joinpath(self.savefile_name+'.nc'))