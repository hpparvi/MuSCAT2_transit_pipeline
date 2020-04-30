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

from astropy.stats import sigma_clip, mad_std
from ldtk import LDPSetCreator
from matplotlib.pyplot import subplots, setp, figure
from muscat2ph.catalog import get_toi
from numba import njit, prange
from numpy import atleast_2d, zeros, exp, log, array, nanmedian, concatenate, ones, arange, where, diff, inf, arccos, \
    sqrt, squeeze, floor, linspace, pi, c_, any, all, percentile, median, repeat, mean, newaxis, isfinite, pad, clip, \
    delete, s_, log10, argsort, atleast_1d, tile, any, fabs, zeros_like, sort, ones_like, fmin, digitize, ceil, full, \
    nan

from numpy.random import permutation, uniform, normal
from pytransit import QuadraticModel, QuadraticModelCL, BaseLPF, LinearModelBaseline
from pytransit.contamination import SMContamination
from pytransit.contamination.filter import sdss_g, sdss_r, sdss_i, sdss_z
from pytransit.contamination.instrument import Instrument
from pytransit.lpf.lpf import map_pv, map_ldc
from pytransit.orbits.orbits_py import as_from_rhop, duration_eccentric, i_from_ba, d_from_pkaiews, epoch
from pytransit.param.parameter import NormalPrior as NP, UniformPrior as UP, LParameter, PParameter, ParameterSet, \
    GParameter
from scipy.ndimage import binary_erosion
from scipy.stats import logistic, norm
from uncertainties import ufloat

def downsample_time(time, values, inttime=30.):
    if values.ndim == 1:
        values = atleast_2d(values).T
    duration = 24. * 60. * 60. * time.ptp()
    nbins = int(ceil(duration / inttime))
    bins = arange(nbins)
    edges = time[0] + bins * inttime / 24 / 60 / 60
    bids = digitize(time, edges) - 1
    bt, bv = full(nbins, nan), zeros((nbins, values.shape[1]))
    for i, bid in enumerate(bins):
        bmask = bid == bids
        if bmask.sum() > 0:
            bt[i] = time[bmask].mean()
            bv[i,:] = values[bmask,:].mean(0)
    m = isfinite(bt)
    return bt[m], squeeze(bv[m])

@njit
def running_mean(time, flux, npt, width_min):
    btime = linspace(time.min(), time.max(), npt)
    bflux, berror = zeros(npt), zeros(npt)
    for i in range(npt):
        m = fabs(time - btime[i]) < 0.5 * width_min / 60 / 24
        bflux[i] = flux[m].mean()
        berror[i] = flux[m].std() / sqrt(m.sum())
    return btime, bflux, berror


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
def lnlike_logistic_v1d(o, m, e, lcids):
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


@njit
def contaminate(flux, cnt, lcids, pbids):
    flux = atleast_2d(flux)
    npv = flux.shape[0]
    npt = flux.shape[1]
    for ipv in range(npv):
        for ipt in range(npt):
            c = cnt[ipv, pbids[lcids[ipt]]]
            flux[ipv, ipt] = c + (1.-c)*flux[ipv, ipt]
    return flux


@njit
def change_depth(relative_depth, flux, lcids, pbids):
    npt = lcids.size
    npop = relative_depth.shape[0]
    flux = atleast_2d(flux)
    flux2 = zeros_like(flux)
    for ipv in range(npop):
        for ipt in range(npt):
            flux2[ipv, ipt] = (flux[ipv, ipt] - 1.) * relative_depth[ipv, pbids[lcids[ipt]]] + 1.
    return flux2


@njit(fastmath=True)
def map_pv_achromatic_nocnt(pv):
    pv = atleast_2d(pv)
    pvt = zeros((pv.shape[0], 7))
    pvt[:,0]   = sqrt(pv[:,4])
    pvt[:,1:3] = pv[:,0:2]
    pvt[:,  3] = as_from_rhop(pv[:,2], pv[:,1])
    pvt[:,  4] = i_from_ba(pv[:,3], pvt[:,3])
    return pvt


@njit(fastmath=True)
def map_pv_achromatic_cnt(pv):
    pv = atleast_2d(pv)
    pvt = zeros((pv.shape[0], 7))
    pvt[:, 0] = sqrt(pv[:, 5])
    pvt[:, 1:3] = pv[:, 0:2]
    pvt[:, 3] = as_from_rhop(pv[:, 2], pv[:, 1])
    pvt[:, 4] = i_from_ba(pv[:, 3], pvt[:, 3])
    return pvt


class M2BaseLPF(BaseLPF):
    def __init__(self, target: str, use_opencl: bool = False, n_legendre: int = 0,
                 with_transit=True, with_contamination=False,
                 radius_ratio: str = 'achromatic', noise_model='white', klims=(0.005, 0.25),
                 contamination_model: str = 'physical',
                 contamination_reference_passband: str = "r'",
                 bin=None):

        assert radius_ratio in ('chromatic', 'achromatic')
        assert noise_model in ('white', 'gp')

        self.use_opencl = use_opencl
        self.planet = None

        self.with_transit = with_transit
        self.with_contamination = with_contamination
        self.achromatic_transit = radius_ratio == 'achromatic'
        self.noise_model = noise_model
        self.radius_ratio = radius_ratio
        self.n_legendre = n_legendre
        self.contamination_reference_passband = contamination_reference_passband

        filters, times, fluxes, covariates, wns, pbids, nids = self._read_data()
        tref = floor(min([t.min() for t in times]))

        if use_opencl:
            import pyopencl as cl
            ctx = cl.create_some_context()
            queue = cl.CommandQueue(ctx)
            tm = QuadraticModelCL(klims=klims, nk=1024, nz=1024, cl_ctx=ctx, cl_queue=queue)
        else:
            tm = QuadraticModel(interpolate=True, klims=klims, nk=1024, nz=1024)

        self.wns = wns
        BaseLPF.__init__(self, target, filters, times, fluxes, pbids=pbids, covariates=covariates, wnids=nids, tm=tm,
                         tref=tref)

    def _read_data(self):
        raise NotImplementedError

    def _init_baseline(self):
        self._add_baseline_model(LinearModelBaseline(self))

    def _init_parameters(self):
        self.ps = ParameterSet()
        if self.with_transit:
            self._init_p_orbit()
            self._init_p_planet()
            self._init_p_limb_darkening()
        #self._init_p_baseline()
        #self._init_p_noise()
        self.ps.freeze()

    def _init_p_planet(self):
        """Planet parameter initialisation.
        """

        # 1. Achromatic radius ratio
        # --------------------------
        if self.radius_ratio == 'achromatic':
            if not self.with_contamination:
                pk2 = [PParameter('k2', 'area_ratio', 'A_s', UP(0.005**2, 0.25**2), (0.005**2, 0.25**2))]
                self.ps.add_passband_block('k2', 1, 1, pk2)
                self._pid_k2 = repeat(self.ps.blocks[-1].start, self.npb)
                self._start_k2 = self.ps.blocks[-1].start
                self._sl_k2 = self.ps.blocks[-1].slice
            else:
                pk2 = [PParameter('k2_app', 'apparent_area_ratio', 'A_s', UP(0.005**2, 0.25**2), (0.005**2, 0.25**2))]
                pcn = [GParameter('k2_true', 'true_area_ratio', 'A_s', UP(0.005**2, 0.75**2), bounds=(1e-8, inf)),
                       GParameter('teff_h', 'host_teff', 'K', UP(2500, 12000), bounds=(2500, 12000)),
                       GParameter('teff_c', 'contaminant_teff', 'K', UP(2500, 12000), bounds=(2500, 12000))]
                self.ps.add_passband_block('k2', 1, 1, pk2)
                self._pid_k2 = repeat(self.ps.blocks[-1].start, self.npb)
                self._start_k2 = self.ps.blocks[-1].start
                self._sl_k2 = self.ps.blocks[-1].slice
                self.ps.add_global_block('contamination', pcn)
                self._pid_cn = arange(self.ps.blocks[-1].start, self.ps.blocks[-1].stop)
                self._sl_cn = self.ps.blocks[-1].slice
                self._additional_log_priors.append(lambda pv: where(pv[:, 4] < pv[:, 5], 0, -inf))

        # 2. Chromatic radius ratio
        # -------------------------
        else:
            pk2 = [PParameter(f'k2_{pb}', f'area_ratio {pb}', 'A_s', UP(0.005**2, 0.25**2), (0.005**2, 0.25**2)) for pb in self.passbands]
            self.ps.add_passband_block('k2', 1, self.npb, pk2)
            self._pid_k2 = arange(self.npb) + self.ps.blocks[-1].start
            self._start_k2 = self.ps.blocks[-1].start
            self._sl_k2 = self.ps.blocks[-1].slice

            if self.with_contamination:
                pcn = [GParameter('cnt_ref', 'Reference contamination', '', UP(0., 1.), (0., 1.)),
                       GParameter('teff_h', 'host_teff', 'K', UP(2500, 12000), bounds=(2500, 12000)),
                       GParameter('teff_c', 'contaminant_teff', 'K', UP(2500, 12000), bounds=(2500, 12000))]
                self.ps.add_global_block('contamination', pcn)
                self._pid_cn = arange(self.ps.blocks[-1].start, self.ps.blocks[-1].stop)
                self._sl_cn = self.ps.blocks[-1].slice


    def _init_instrument(self):
        filters = {'g': sdss_g, 'r': sdss_r, 'i':sdss_i, 'z_s':sdss_z}
        self.instrument = Instrument('MuSCAT2', [filters[pb] for pb in self.passbands])
        self.cm = SMContamination(self.instrument, self.contamination_reference_passband)

    def add_ldtk_prior(self, teff: tuple, logg: tuple, z: tuple, uncertainty_multiplier: float = 3, **kwargs) -> None:
        from ldtk import sdss_g, sdss_r, sdss_i, sdss_z
        passbands = [f for f,pb in zip((sdss_g, sdss_r, sdss_i, sdss_z), 'g r i z'.split()) if pb in ' '.join(self.passbands)]
        BaseLPF.add_ldtk_prior(self, teff, logg, z, passbands, uncertainty_multiplier, **kwargs)


    def set_radius_ratio_prior(self, kmin, kmax):
        for p in self.ps[self._sl_k2]:
            p.prior = UP(kmin ** 2, kmax ** 2)
            p.bounds = [kmin ** 2, kmax ** 2]
        self.ps.thaw()
        self.ps.freeze()

    def downsample(self, exptime: float) -> None:
        raise NotImplementedError

    def _transit_model_achromatic_nocnt(self, pvp, copy=True):
        return super().transit_model(pvp, copy)

    def _transit_model_achromatic_cnt(self, pvp, copy=True):
        pvp = atleast_2d(pvp)
        cnt = zeros((pvp.shape[0], self.npb))
        pvt = map_pv_achromatic_cnt(pvp)
        ldc = map_ldc(pvp[:, self._sl_ld])
        flux = self.tm.evaluate_pv(pvt, ldc, copy=copy)
        for i, pv in enumerate(pvp):
            if (2500 < pv[6] < 12000) and (2500 < pv[7] < 12000):
                cnref = 1. - pv[4] / pv[5]
                cnt[i, :] = self.cm.contamination(cnref, pv[6], pv[7])
            else:
                cnt[i, :] = -inf
        return contaminate(flux, cnt, self.lcids, self.pbids)

    def _transit_model_chromatic_nocnt(self, pvp, copy=True):
        pvp = atleast_2d(pvp)
        mean_ar = pvp[:, self._sl_k2].mean(1)
        pvv = zeros((pvp.shape[0], pvp.shape[1] - self.npb + 1))
        pvv[:, :4] = pvp[:, :4]
        pvv[:, 4] = mean_ar
        pvv[:, 5:] = pvp[:, 4 + self.npb:]
        pvt = map_pv(pvv)
        ldc = map_ldc(pvp[:, self._sl_ld])
        flux = self.tm.evaluate_pv(pvt, ldc, copy)
        rel_ar = pvp[:, self._sl_k2] / mean_ar[:, newaxis]
        return change_depth(rel_ar, flux, self.lcids, self.pbids)

    def _transit_model_chromatic_cnt(self, pvp, copy=True):
        pvp = atleast_2d(pvp)
        flux = self._transit_model_chromatic_nocnt(pvp, copy)
        cnt = zeros((pvp.shape[0], self.npb))
        for i, (cnref, teffh, teffc) in enumerate(pvp[:, self._sl_cn]):
            if (0. <= cnref <= 1.) and (2500 < teffh < 12000) and (2500 < teffc < 12000):
                cnt[i, :] = self.cm.contamination(cnref, teffh, teffc)
            else:
                cnt[i, :] = -inf
        return contaminate(flux, cnt, self.lcids, self.pbids)

    def transit_model(self, pv, copy=True):
        pv = atleast_2d(pv).copy()
        pv[:,0] -= self._tref
        if self.with_transit:
            if self.achromatic_transit:
                if self.with_contamination:
                    return self._transit_model_achromatic_cnt(pv, copy)
                else:
                    return  self._transit_model_achromatic_nocnt(pv, copy)
            else:
                if self.with_contamination:
                    return self._transit_model_chromatic_cnt(pv, copy)
                else:
                    return self._transit_model_chromatic_nocnt(pv, copy)
        else:
            return 1.

    def flux_model(self, pv):
        baseline    = self.baseline(pv)
        trends      = self.trends(pv)
        model_flux = self.transit_model(pv, copy=True)
        return baseline * model_flux + trends

    def lnlikelihood(self, pv):
        flux_m = self.flux_model(pv)
        wn = 10**(atleast_2d(pv)[:,self._sl_wn])
        return lnlike_logistic_v1d(self.ofluxa, flux_m, wn, self.lcids)

    def ldprior(self, pv):
        ld = pv[:, self._sl_ld]
        #lds = ld[:,::2] + ld[:, 1::2]
        #return where(any(diff(lds, 1) > 0., 1), -inf, 0.)
        return where(any(diff(ld[:,::2], 1) > 0., 1) | any(diff(ld[:,1::2], 1) > 0., 1), -inf, 0)

    def inside_obs_prior(self, pv):
        return where(transit_inside_obs(pv, self._tmin, self._tmax, 20.), 0., -inf)

    def lnprior(self, pv):
        pv = atleast_2d(pv)
        lnp = self.ps.lnprior(pv)
        if self.with_transit:
            lnp += self.additional_priors(pv) + self.ldprior(pv)
            #lnp += self.ldprior(pv) + self.inside_obs_prior(pv) + self.additional_priors(pv)
        return lnp

    def add_inside_window_prior(self, min_duration=20):
        self._iptmin = self.timea.min()
        self._iptmax = self.timea.max()
        self._ipmdur = min_duration / 60 / 24
        def is_inside_window(pv):
            pv = atleast_2d(pv)
            tc = pv[:, 0]
            a = as_from_rhop(pv[:, 2], pv[:, 1])
            t14 = duration_eccentric(pv[:, 1], sqrt(pv[:, 4]), a, arccos(pv[:, 3] / a), 0, 0, 1)
            ingress = tc - 0.5 * t14
            egress = tc + 0.5 * t14
            inside_limits = fmin(egress - self._iptmin - self._ipmdur, self._iptmax - ingress - self._ipmdur) > 0
            return where(inside_limits, 0, -inf)
        self.lnpriors.append(is_inside_window)

    def transit_bic(self):
        from scipy.optimize import minimize

        def bic(ll1, ll2, d1, d2, n):
            return ll2 - ll1 + 0.5 * (d1 - d2) * log(n)

        pv_all = self.de.minimum_location.copy()
        pv_bl = pv_all[self._sl_bl]
        wn = atleast_2d(10 ** pv_all[self._sl_err])

        def lnlikelihood_baseline(self, pv):
            pv_all[self._sl_bl] = pv
            flux_m = self.baseline(pv_all)

            if self.photometry_frozen:
                lnl = squeeze(lnlike_logistic_v1d(self.ofluxa, flux_m, wn, self.lcids))
            else:
                lnl = squeeze(lnlike_logistic_v(self.relative_flux(pv), flux_m, wn, self.lcids))

            return lnl if lnl == lnl else -inf

        ll_no_transit = -minimize(lambda pv: -lnlikelihood_baseline(self, pv), pv_bl).fun
        ll_with_transit = float(self.lnlikelihood(self.de.minimum_location.copy()))
        d_with_transit = len(self.ps)
        d_no_transit = d_with_transit - self._start_bl
        return bic(ll_no_transit, ll_with_transit, d_no_transit, d_with_transit, self.timea.size)

    def posterior_samples(self, burn: int = 0, thin: int = 1, derived_parameters: bool = True, add_tref = True):
        df = BaseLPF.posterior_samples(self, burn, thin, derived_parameters)
        if add_tref:
            df.tc += self._tref
        return df

    def plot_light_curves(self, model: str = 'de', figsize: tuple = (13, 8), fig=None, gridspec=None,
                          ylim_transit=None, ylim_residuals=None, bin_width=None):
        if fig is None:
            fig = figure(figsize=figsize, constrained_layout=True)

        gs = dict(height_ratios=(0.5, 2, 2, 1))
        if gridspec:
            gs.update(gridspec)

        axs = fig.subplots(4, self.nlc, sharey='row', gridspec_kw=gs, squeeze=False)

        if model == 'de':
            pv = self.de.minimum_location
            err = 10 ** pv[self._sl_err]
            if not self.photometry_frozen:
                self.set_ofluxa(pv)

        elif model == 'mc':
            fc = array(self.posterior_samples(derived_parameters=False, add_tref=False))
            pv = permutation(fc)[:300]
            err = 10 ** median(pv[:, self._sl_err], 0)
            if not self.photometry_frozen:
                self.set_ofluxa(median(pv, 0))
        else:
            raise NotImplementedError("Light curve plotting `model` needs to be either `de` or `mc`")

        ps = [50, 16, 84]
        if self.with_transit:
            tm = percentile(atleast_2d(self.transit_model(pv)), ps, 0)
        else:
            tm = percentile(atleast_2d(ones(self.timea.size)), ps, 0)
        fm = percentile(atleast_2d(self.flux_model(pv)), ps, 0)
        bl = percentile(atleast_2d(self.baseline(pv)), ps, 0)

        for i, sl in enumerate(self.lcslices):
            t = self.timea[sl]
            axs[1, i].plot(t, self.ofluxa[sl], '.', alpha=0.5)
            axs[1, i].plot(t, fm[0][sl], 'k', lw=2)
            axs[2, i].plot(t, self.ofluxa[sl] / bl[0][sl], '.', alpha=0.5)
            if model == 'mc':
                axs[2, i].fill_between(t, tm[1][sl], tm[2][sl], facecolor='darkblue', alpha=0.25)
            axs[2, i].plot(t, tm[0][sl], 'k', lw=2)
            axs[3, i].plot(t, self.ofluxa[sl] - fm[0][sl], '.', alpha=0.5)

            #if bin_width:
            #    bt, bf, be = downsample_time(t, self.ofluxa[sl] / bl[0][sl], 300)
            #    axs[2, i].plot(bt, bf, 'ok')

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
        setp(axs[3, :], xlabel=f'Time - {self._tref:9.0f} [BJD]')
        setp(axs[0, :], xlabel='Residual [ppt]', yticks=[])

        if ylim_transit is not None:
            setp(axs.flat[self.nlc:3 * self.nlc], ylim=ylim_transit)
        else:
            setp(axs.flat[self.nlc:2 * self.nlc], ylim=array([0.995, 1.005]) * percentile(self.ofluxa, [2, 98]))
            setp(axs.flat[2 * self.nlc:3 * self.nlc],
                 ylim=array([0.995, 1.005]) * percentile(self.ofluxa / bl[0], [2, 98]))

        if ylim_residuals is not None:
            setp(axs.flat[3 * self.nlc:], ylim=ylim_residuals)
        else:
            setp(axs.flat[3 * self.nlc:], ylim=array([0.995, 1.005]) * percentile(self.ofluxa / bl[0] - tm[0], [2, 98]))

        setp(axs[1:,:], xlim=(self.timea.min(), self.timea.max()))

        [sb.despine(ax=ax, offset=5, left=True) for ax in axs[0]]
        return fig, axs

    def plot_posteriors(self, figsize: tuple = (13, 5), fig=None, gridspec=None, plot_k=True):
        if fig is None:
            fig = figure(figsize=figsize, constrained_layout=True)
        axs = fig.subplots(2, 3, gridspec_kw=gridspec)

        df = self.posterior_samples()
        df = df.iloc[:, :self._sl_ld.start].copy()

        # Transit depth and radius ratio
        # ------------------------------
        k2cols = [c for c in df.columns if 'k2' in c]
        def plot_estimates(x, p, ax, bwidth=0.8, color='k'):
            ax.bar(x, p[4, :] - p[3, :], bwidth, p[3, :], alpha=0.25, fc=color)
            ax.bar(x, p[2, :] - p[1, :], bwidth, p[1, :], alpha=0.25, fc=color)
            [ax.plot((xx - 0.47 * bwidth, xx + 0.47 * bwidth), (pp[[0, 0]]), 'k') for xx, pp in zip(x, p.T)]

        x = arange(self.nlc)
        plot_estimates(x, 1e6 * df[k2cols].quantile([0.5, 0.16, 0.84, 0.025, 0.975]).values, axs[0, 0])
        if self.toi is not None:
            axs[0, 0].axhline(self.toi.depth[0], c='k', ls='--')
        setp(axs[0, 0], ylabel='Transit depth [ppm]', xticks=x, xticklabels=self.passbands)

        if plot_k:
            plot_estimates(x, sqrt(df[k2cols].quantile([0.5, 0.16, 0.84, 0.025, 0.975]).values), axs[1, 0])
            if self.toi is not None:
                axs[1, 0].axhline(sqrt(1e-6 * self.toi.depth[0]), c='k', ls='--')
            setp(axs[1, 0], ylabel='Radius ratio', xticks=x, xticklabels=self.passbands)
        else:
            axs[1, 0].remove()

        # Transit centre
        # --------------
        p = self.ps.priors[0]
        trange = p.mean - 3 * p.std, p.mean + 3 * p.std
        x = linspace(*trange)
        axs[0, 1].hist(df.tc - self._tref, 50, density=True, range=trange, alpha=0.5, edgecolor='k',
                       histtype='stepfilled')
        axs[0, 1].set_xlabel(f'Transit center - {self._tref:9.0f} [BJD]')
        axs[0, 1].fill_between(x, exp(p.logpdf(x)), alpha=0.5, edgecolor='k')

        # Period
        # ------
        p = self.ps.priors[1]
        trange = p.mean - 3 * p.std, p.mean + 3 * p.std
        x = linspace(*trange)
        axs[1, 1].hist(df.p, 50, density=True, range=trange, alpha=0.5, edgecolor='k', histtype='stepfilled')
        axs[1, 1].set_xlabel('Period [days]')
        axs[1, 1].fill_between(x, exp(p.logpdf(x)), alpha=0.5, edgecolor='k')
        setp(axs[1, 1], xticks=df.p.quantile([0.05, 0.5, 0.95]))

        # Rest without priors
        # -------------------
        names = 'stellar density, impact parameter'.split(', ')
        for i, ax in enumerate(axs.flat[[2, 5]]):
            ax.hist(df.iloc[:, i + 2], 50, density=True, alpha=0.5, edgecolor='k', histtype='stepfilled')
            ax.set_xlabel(names[i])

        sb.despine(fig, offset=10)
        setp(axs[:, 1:], yticks=[])
        return fig, axs

    def plot_chains(self, pids=(0, 1, 2, 3, 4)):
        fig, axs = subplots(len(pids), 1, figsize=(13, 10), constrained_layout=True, sharex='all')
        x = arange(self.sampler.chain.shape[1])
        for i, (pid, ax) in enumerate(zip(pids, axs)):
            pes = percentile(self.sampler.chain[:, :, pid], [50, 16, 84, 0.5, 99.5], 0)
            ax.fill_between(x, pes[3], pes[4])
            ax.fill_between(x, pes[1], pes[2])
            ax.plot(pes[0], 'k')
            setp(ax, ylabel=self.ps.names[pid])
        setp(axs, xlim=(0, x[-1]))

    def plot_running_mean(self, figsize=(13, 5), errors=True, combine=False, remove_baseline=True, ylim=None, npt=100,
                     width_min=10):
        pv = self.de.minimum_location
        rflux = self.relative_flux(pv)
        if remove_baseline:
            rflux /= squeeze(self.baseline(pv))

        if combine:
            bt, bf, be = running_mean(self.timea, rflux, npt, width_min)
            fig, axs = subplots(figsize=figsize, constrained_layout=True)
            if errors:
                axs.errorbar(bt, bf, be, drawstyle='steps-mid', c='k')
            else:
                axs.plot(bt, bf, drawstyle='steps-mid', c='k')
            axs.fill_between(bt, bf - 3 * be, bf + 3 * be, alpha=0.2, step='mid')

        else:
            rfluxes = [rflux[sl] for sl in self.lcslices]
            fig, axs = subplots(1, self.nlc, figsize=figsize, constrained_layout=True, sharey='all')
            for i, ax in enumerate(axs):
                bt, bf, be = running_mean(self.times[i], rfluxes[i], npt, width_min)
                if errors:
                    ax.errorbar(bt, bf, be, drawstyle='steps-mid', c='k')
                else:
                    ax.plot(bt, bf, drawstyle='steps-mid', c='k')

        if ylim:
            setp(axs, ylim=ylim)
