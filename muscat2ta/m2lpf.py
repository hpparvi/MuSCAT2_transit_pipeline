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
import seaborn as sb
from os.path import join, split

from ldtk import LDPSetCreator
from matplotlib.pyplot import subplots, setp, figure
from muscat2ph.catalog import get_toi
from numba import njit, prange
from numpy import atleast_2d, zeros, exp, log, array, nanmedian, concatenate, ones, arange, where, diff, inf, arccos, \
    sqrt, squeeze, floor, linspace, pi, c_, any, all, percentile, median, repeat, mean, newaxis, isfinite
from numpy.random import permutation
from pytransit import QuadraticModel, QuadraticModelCL
from pytransit.contamination import SMContamination
from pytransit.contamination.filter import sdss_g, sdss_r, sdss_i, sdss_z
from pytransit.contamination.instrument import Instrument
from pytransit.lpf.lpf import BaseLPF
from pytransit.orbits.orbits_py import as_from_rhop, duration_eccentric, i_from_ba, d_from_pkaiews
from pytransit.param.parameter import NormalPrior as NP, UniformPrior as UP, LParameter, PParameter
from scipy.stats import logistic, norm
from uncertainties import ufloat


@njit
def transit_inside_obs(pvp, tmin, tmax, limit_min: float = 10.):
    limit = limit_min/60./24.
    a = as_from_rhop(pvp[:,2], pvp[:,1])
    i = i_from_ba(pvp[:, 3], a)
    duration = d_from_pkaiews(pvp[:,1], sqrt(pvp[:,4]), a, i, 0, 0, 1)
    ingress = pvp[:,0] + 0.5*duration
    egress = pvp[:,0] + 0.5*duration
    return (ingress < tmax - limit) & (egress > tmin + limit)


@njit(parallel=True, cache=False, fastmath=True)
def lnlike_logistic_v(o, m, e, lcids):
    m = atleast_2d(m)
    npv = m.shape[0]
    npt = o.size
    lnl = zeros(npv)
    for i in prange(npv):
        for j in range(npt):
            k = lcids[j]
            t = exp((o[j]-m[i,j])/e[i,k])
            lnl[i] += log(t / (e[i,k]*(1.+t)**2))
    return lnl

@njit(parallel=True, cache=False, fastmath=True)
def lnlike_logistic_vb(o, m, e):
    m = atleast_2d(m)
    npv = m.shape[0]
    npt = o.size
    lnl = zeros(npv)
    for i in prange(npv):
        t = exp((o-m[i,:])/e[i,0])
        lnl[i] = sum(log(t / (e[i,0]*(1.+t)**2)))
    return lnl

class M2LPF(BaseLPF):
    def __init__(self, target: str, photometry: list, tid: int, cids: list, aid: int,
                 filters: tuple, use_opencl: bool = False, period: float = 5., use_toi_info=True):
        self.use_opencl = use_opencl
        self.planet = None
        self.aid = aid
        self.tid = tid
        self.cids = cids

        self.covnames = 'intercept sky airmass xshift yshift entropy'.split()

        times =  [array(ph.bjd) for ph in photometry]
        fluxes = [array(ph.flux[:, tid, aid]) for ph in photometry]
        fluxes = [f/nanmedian(f) for f in fluxes]

        self.t0 = floor(times[0].min())
        times = [t - self.t0 for t in times]

        self._tmin = times[0].min()
        self._tmax = times[0].max()

        covariates = []
        for ph in photometry:
            covs = concatenate([ones([ph._fmask.sum(), 1]), array(ph.aux)[:,[1,3,4]]], 1)
            covariates.append(covs)

        self.airmasses = [array(ph.aux[:,2]) for ph in photometry]

        wns = [ones(ph.nframes) for ph in photometry]

        if use_opencl:
            import pyopencl as cl
            ctx = cl.create_some_context()
            queue = cl.CommandQueue(ctx)
            tm = QuadraticModelCL(klims=(0.005, 0.25), nk=512, nz=512, cl_ctx=ctx, cl_queue=queue)
        else:
            tm = QuadraticModel(interpolate=True, klims=(0.005, 0.25), nk=512, nz=512)

        super().__init__(target, filters, times, fluxes, wns, arange(len(photometry)), covariates, tm = tm)

        self.refs = []
        for ip, ph in enumerate(photometry):
            self.refs.append([array(ph.flux[:, cid, aid]) for cid in self.cids])

        self.refa = zeros((self.ofluxa.size, len(cids)))
        for ic, cid in enumerate(cids):
            for ip, ph in enumerate(photometry):
                self.refa[self.lcslices[ip], ic] = ph.flux[:, cid, aid]

        if 'toi' in self.name and use_toi_info:
            self.toi = toi = get_toi(float(self.name.strip('toi')))
            tn = round((self.times[0].mean() - (toi.epoch[0] - self.t0)) / toi.period[0])
            epoch = ufloat(*toi.epoch)
            period = ufloat(*toi.period)
            tc = epoch - self.t0 + tn*period
            self.set_prior(0, NP(tc.n, tc.s))
            self.set_prior(1, NP(*toi.period))
            self.add_t14_prior(toi.duration[0]/24, 0.5*toi.duration[1]/24)
        else:
            p = self.planet.P if self.planet else period
            t0 = times[0].mean()
            tce = times[0].ptp() / 10
            self.set_prior(0, NP(t0, tce))
            self.set_prior(1, NP( p, 1e-5))


    def _init_p_baseline(self):
        c = []
        for ilc in range(self.nlc):
            c.append(LParameter(f'ci_{ilc:d}', 'intercept_{ilc:d}', '', NP(1.0, 0.03), bounds=( 0.5, 1.5)))
            c.append(LParameter(f'cs_{ilc:d}', 'sky_{ilc:d}',       '', NP(0.0, 0.01), bounds=(-0.5, 0.5)))
            c.append(LParameter(f'ca_{ilc:d}', 'airmass_{ilc:d}',   '', UP(0.0, 1.00), bounds=( 0.0, 1.0)))
            c.append(LParameter(f'cx_{ilc:d}', 'xshift_{ilc:d}',    '', NP(0.0, 0.01), bounds=(-0.5, 0.5)))
            c.append(LParameter(f'cy_{ilc:d}', 'yshift_{ilc:d}',    '', NP(0.0, 0.01), bounds=(-0.5, 0.5)))
        self.ps.add_lightcurve_block('ccoef', 5, self.nlc, c)
        self._sl_ccoef = self.ps.blocks[-1].slice
        self._start_ccoef = self.ps.blocks[-1].start

        c = []
        for ilc in range(self.nlc):
            for irf in range(len(self.cids)):
                c.append(LParameter(f'ref_{irf:d}_{ilc:d}', 'comparison_star_{irf:d}_{ilc:d}', '', UP(0.0, 1.0), bounds=( 0.0, 1.0)))
        self.ps.add_lightcurve_block('rstars', len(self.cids), self.nlc, c)
        self._sl_ref = self.ps.blocks[-1].slice
        self._start_ref = self.ps.blocks[-1].start


    def _init_p_planet(self):
        """Planet parameter initialisation.
        """
        pk2 = [PParameter('k2', 'area_ratio', 'A_s', UP(0.005**2, 0.25**2), (0.005**2, 0.25**2))]
        self.ps.add_passband_block('k2', 1, 1, pk2)
        self._pid_k2 = repeat(self.ps.blocks[-1].start, self.npb)
        self._start_k2 = self.ps.blocks[-1].start
        self._sl_k2 = self.ps.blocks[-1].slice

    def _init_instrument(self):
        self.instrument = Instrument('MuSCAT2', [sdss_g, sdss_r, sdss_i, sdss_z])
        self.cm = SMContamination(self.instrument, "i'")

    def lnlikelihood(self, pv):
        flux_m = self.flux_model(pv)
        wn = 10**(atleast_2d(pv)[:,self._sl_err])
        return lnlike_logistic_v(self.ofluxa, flux_m, wn, self.lcids)

    def flux_model(self, pv):
        baseline    = self.baseline(pv)
        trends      = self.trends(pv)
        model_flux = self.transit_model(pv, copy=True)
        return baseline * model_flux + trends

    def reference_flux(self, pv):
        pv = atleast_2d(pv)
        p = where(pv[:, self._sl_ref] < 0.5, 0, 1)
        r = zeros((pv.shape[0], self.ofluxa.size))
        nref = len(self.cids)
        for ipb, sl in enumerate(self.lcslices):
            for i in range(nref):
                r[:, sl] += (self.refs[ipb][i] * p[:, ipb * nref + i: ipb * nref + i + 1])
            r[:, sl] = r[:, sl] / median(r[:, sl], 1)[:, newaxis]
        return where(isfinite(r), r, inf)


    def extinction(self, pv):
        pv = atleast_2d(pv)
        ext = zeros((pv.shape[0], self.timea.size))
        for i, sl in enumerate(self.lcslices):
            st = self._start_ccoef + i * 5
            ext[:, sl] = exp(- pv[:, st + 2] * self.airmasses[i][:, newaxis]).T
            ext[:, sl] /= mean(ext[:, sl], 1)[:, newaxis]
        return squeeze(ext)

    def baseline(self, pv):
        pv = atleast_2d(pv)
        bl = zeros((pv.shape[0], self.timea.size))
        for i,sl in enumerate(self.lcslices):
            st = self._start_ccoef + i*5
            p = pv[:, st:st+6]
            bl[:, sl] = (self.covariates[i] @ p[:,[0,1,3,4]].T).T
        bl = bl * self.extinction(pv) * self.reference_flux(pv)
        return bl

    def ldprior(self, pv):
        ld = pv[:, self._sl_ld]
        lds = ld[:,::2] + ld[:, 1::2]
        return where(any(diff(ld[:,::2], 1) > 0., 1) | any(diff(ld[:,1::2], 1) > 0., 1), -inf, 0)

    def inside_obs_prior(self, pv):
        return where(transit_inside_obs(pv, self._tmin, self._tmax, 20.), 0., -inf)

    def lnprior(self, pv):
        pv = atleast_2d(pv)
        return super().lnprior(pv) + self.ldprior(pv) + self.inside_obs_prior(pv) + self.additional_priors(pv)

    def add_t14_prior(self, mean: float, std: float) -> None:
        """Add a normal prior on the transit duration.

        Parameters
        ----------
        mean
        std

        Returns
        -------

        """
        def T14(pv):
            pv = atleast_2d(pv)
            a = as_from_rhop(pv[:,2], pv[:,1])
            t14 = duration_eccentric(pv[:,1], sqrt(pv[:,4]), a, arccos(pv[:,3] / a), 0, 0, 1)
            return norm.logpdf(t14, mean, std)
        self.lnpriors.append(T14)

    def add_ldtk_prior(self, teff: tuple, logg: tuple, z: tuple,
                       uncertainty_multiplier: float = 3,
                       pbs: tuple = ('g', 'r', 'i', 'z')) -> None:
        """Add a LDTk-based prior on the limb darkening.

        Parameters
        ----------
        teff
        logg
        z
        uncertainty_multiplier
        pbs

        Returns
        -------

        """
        fs = {n: f for n, f in zip('g r i z'.split(), (sdss_g, sdss_r, sdss_i, sdss_z))}
        filters = [fs[k] for k in pbs]
        self.ldsc = LDPSetCreator(teff, logg, z, filters)
        self.ldps = self.ldsc.create_profiles(1000)
        self.ldps.resample_linear_z()
        self.ldps.set_uncertainty_multiplier(uncertainty_multiplier)
        def ldprior(pv):
            pv = atleast_2d(pv)
            lnl = zeros(pv.shape[0])
            for i in range(pv.shape[0]):
                lnl[i] = self.ldps.lnlike_tq(pv[i, self._sl_ld])
            return lnl
        self.lnpriors.append(ldprior)

    def plot_light_curves(self, model: str = 'de', figsize: tuple = (13, 8), fig=None, gridspec=None):
        if fig is None:
            fig = figure(figsize=figsize, constrained_layout=True)

        gs = dict(height_ratios=(0.5, 2, 2, 1))
        if gridspec:
            gs.update(gridspec)

        axs = fig.subplots(4, self.nlc, sharey='row', gridspec_kw=gs)

        if model == 'de':
            pv = self.de.minimum_location
            err = 10 ** pv[self._sl_err]

        elif model == 'mc':
            fc = array(self.posterior_samples(include_ldc=True))
            pv = permutation(fc)[:300]
            err = 10 ** median(pv[:, self._sl_err], 0)
        else:
            raise NotImplementedError("Light curve plotting `model` needs to be either `de` or `mc`")

        ps = [50, 16, 84]
        tm = percentile(atleast_2d(self.transit_model(pv)), ps, 0)
        fm = percentile(atleast_2d(self.flux_model(pv)), ps, 0)
        bl = percentile(atleast_2d(self.baseline(pv)), ps, 0)
        t0 = self.t0

        for i, sl in enumerate(self.lcslices):
            t = self.timea[sl]
            axs[1, i].plot(t, self.ofluxa[sl], '.', alpha=0.5)
            axs[1, i].plot(t, fm[0][sl], 'k', lw=2)
            axs[2, i].plot(t, self.ofluxa[sl] / bl[0][sl], '.', alpha=0.5)
            if model == 'mc':
                axs[2, i].fill_between(t, tm[1][sl], tm[2][sl], facecolor='darkblue', alpha=0.25)
            axs[2, i].plot(t, tm[0][sl], 'k', lw=2)
            axs[3, i].plot(t, self.ofluxa[sl] - fm[0][sl], '.', alpha=0.5)

            res = self.ofluxa[sl] - fm[0][sl]
            x = linspace(-4 * err, 4 * err)
            axs[0, i].hist(1e3 * res, 'auto', density=True, alpha=0.5)
            axs[0, i].plot(1e3 * x, logistic(0, 1e3 * err[i]).pdf(1e3 * x), 'k')
            axs[0, i].text(0.05, 0.95, f"$\sigma$ = {(1e3 * err[i] * pi / sqrt(3)):5.2f} ppt",
                           transform=axs[0, i].transAxes, va='top')

        [ax.set_title(f"MuSCAT2 {t}", size='large') for ax, t in zip(axs[0], self.passbands)]
        [setp(ax.get_xticklabels(), visible=False) for ax in axs[1:3].flat]
        setp(axs[1, 0], ylabel='Transit + Systematics')
        setp(axs[2, 0], ylabel='Transit - Systematics')
        setp(axs[3, 0], ylabel='Residuals')
        setp(axs[3, :], xlabel=f'Time - {self.t0:9.0f} [BJD]')
        setp(axs[0, :], xlabel='Residual [ppt]', yticks=[])
        [sb.despine(ax=ax, offset=5, left=True) for ax in axs[0]]
        return fig, axs


    def plot_posteriors(self, figsize: tuple = (13, 5), fig=None, gridspec=None):
        if fig is None:
            fig = figure(figsize=figsize, constrained_layout=True)
        axs = fig.subplots(2, 3, gridspec_kw=gridspec)

        df = self.posterior_samples(include_ldc=True)
        df = df.iloc[:, :5].copy()
        df['k'] = sqrt(df.k2)

        names = 'stellar density, impact parameter, transit depth, radius ratio'.split(', ')

        # Transit centre
        # --------------
        p = self.ps.priors[0]
        t0 = floor(df.tc.mean())
        trange = p.mean - t0 - 3 * p.std, p.mean - t0 + 3 * p.std
        x = linspace(*trange)
        axs[0, 0].hist(df.tc - t0, 50, density=True, range=trange, alpha=0.5, edgecolor='k', histtype='stepfilled')
        axs[0, 0].set_xlabel(f'Transit center - {t0:9.0f} [BJD]')
        axs[0, 0].fill_between(x, exp(p.logpdf(x + t0)), alpha=0.5, edgecolor='k')

        # Period
        # ------
        p = self.ps.priors[1]
        trange = p.mean - 3 * p.std, p.mean + 3 * p.std
        x = linspace(*trange)
        axs[0, 1].hist(df.pr, 50, density=True, range=trange, alpha=0.5, edgecolor='k', histtype='stepfilled')
        axs[0, 1].set_xlabel('Period [days]')
        axs[0, 1].fill_between(x, exp(p.logpdf(x)), alpha=0.5, edgecolor='k')
        setp(axs[0, 1], xticks=df.pr.quantile([0.05, 0.5, 0.95]))

        # Rest without priors
        # -------------------
        for i, ax in enumerate(axs.flat[2:]):
            ax.hist(df.iloc[:, i + 2], 50, density=True, alpha=0.5, edgecolor='k', histtype='stepfilled')
            ax.set_xlabel(names[i])

        # TFOP Transit depth estimates
        axs[1, 1].axvline(1e-6 * self.toi.depth[0], c='0.5', lw=2)
        axs[1, 2].axvline(sqrt(1e-6 * self.toi.depth[0]), c='0.5', lw=2)

        setp(axs, yticks=[])
        return fig, axs
