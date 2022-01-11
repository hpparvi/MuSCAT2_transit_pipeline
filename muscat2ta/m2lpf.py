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

from astropy.stats import sigma_clip, mad_std
from astropy.table import Table
from ldtk import LDPSetCreator
from matplotlib.pyplot import subplots, setp, figure
from astropy.io import fits as pf

from numba import njit, prange
from numpy import atleast_2d, zeros, exp, log, array, nanmedian, concatenate, ones, arange, where, diff, inf, arccos, \
    sqrt, squeeze, floor, linspace, pi, c_, any, all, percentile, median, repeat, mean, newaxis, isfinite, pad, clip, \
    delete, s_, log10, argsort, atleast_1d, tile, any, fabs, zeros_like, sort, ones_like, fmin, digitize, ceil, full, \
    nan, transpose, isscalar, empty
from numpy.polynomial import Polynomial

import astropy.units as u
from numpy.random import permutation, uniform, normal
from pytransit import QuadraticModel, QuadraticModelCL, BaseLPF, LinearModelBaseline
from pytransit.contamination import SMContamination
from pytransit.contamination.filter import sdss_g, sdss_r, sdss_i, sdss_z
from pytransit.contamination.instrument import Instrument
from pytransit.lpf.lpf import map_pv, map_ldc
from pytransit.orbits.orbits_py import as_from_rhop, duration_eccentric, i_from_ba, d_from_pkaiews, epoch
from pytransit.param.parameter import NormalPrior as NP, UniformPrior as UP, LParameter, PParameter, ParameterSet, \
    GParameter
from pytransit.lpf.loglikelihood.wnloglikelihood import WNLogLikelihood
from pytransit.utils.de import DiffEvol
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
    if values.ndim == 2:
        bt, bv = full(nbins, nan), zeros((nbins, values.shape[1]))
    else:
        bt, bv = full(nbins, nan), zeros((nbins, values.shape[1], values.shape[2]))
    for i, bid in enumerate(bins):
        bmask = bid == bids
        if bmask.sum() > 0:
            bt[i] = time[bmask].mean()
            if values.ndim == 2:
                bv[i,:] = values[bmask,:].mean(0)
            else:
                bv[i,:,:] = values[bmask,:,:].mean(0)
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
def lnlike_logistic_v(o, m, e, lcids):
    m = atleast_2d(m)
    npv = m.shape[0]
    npt = m.shape[1]
    lnl = zeros(npv)
    for i in prange(npv):
        for j in range(npt):
            k = lcids[j]
            t = exp((o[i,j]-m[i,j])/e[i,k])
            lnl[i] += log(t / (e[i,k]*(1.+t)**2))
    return lnl

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

@njit(parallel=True, cache=False, fastmath=True)
def lnlike_logistic_vbb(o, m, e):
    m = atleast_2d(m)
    npv = m.shape[0]
    npt = o.size
    lnl = zeros(npv)
    for i in prange(npv):
        t = exp((o-m[i,:])/e[i,0])
        lnl[i] = sum(log(t / (e[i,0]*(1.+t)**2)))
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

@njit(parallel=False, fastmath=True)
def flare(time, lcids, pbids, pvp, npb, nflares):
    pvp = atleast_2d(pvp)
    npt = time.size
    npv = pvp.shape[0]
    flux = zeros((npv,npt))
    for i in range(npt):
        for j in range(npv):
            for k in range(nflares):
                if time[i] >= pvp[j, k*(2+npb)]:
                    pbid = pbids[lcids[i]]
                    ii = k*(2+npb)
                    flux[j,i] = pvp[j, ii+2+pbid]*exp(-((time[i]-pvp[j,ii])/pvp[j,ii+1]))
    return flux

@njit
def create_reference_flux(reference_fluxes, ref_mask, ref_aperture_ids):
    npv = ref_mask.shape[0]
    nref = reference_fluxes.shape[1]
    flux = zeros((npv, reference_fluxes.shape[0]))
    for ipv in range(npv):
        for iref in range(nref):
            if ref_mask[ipv, iref]:
                flux[ipv] += reference_fluxes[:, iref, ref_aperture_ids[ipv, iref]]
        flux[ipv] /= median(flux[ipv])
    return flux


class M2WNLogLikelihood(WNLogLikelihood):
    def __call__(self, pvp, model):
        err = 10 ** atleast_2d(pvp)[:, self.pv_slice]
        flux_m = self.lpf.flux_model(pvp)
        if self.lpf.photometry_frozen:
            return lnlike_logistic_v1d(self.lpf.ofluxa, flux_m, err, self.lpf.lcids)
        else:
            return lnlike_logistic_v(self.lpf.relative_flux(pvp), flux_m, err, self.lpf.lcids)

class M2LPF(BaseLPF):
    def __init__(self, target: str, photometry: list, tid: int, cids: list,
                 filters: tuple, aperture_lims: tuple = (0, inf), use_opencl: bool = False,
                 n_legendre: int = 0, use_toi_info=True, with_transit=True, with_contamination=False,
                 radius_ratio: str = 'achromatic', noise_model='white', klims=(0.005, 0.75),
                 contamination_model: str = 'physical',
                 contamination_reference_passband: str = "r'"):
        assert radius_ratio in ('chromatic', 'achromatic')
        assert noise_model in ('white', 'gp')
        assert contamination_model in ('physical', 'direct')

        self.use_opencl = use_opencl
        self.planet = None

        self.photometry_frozen = False
        self.with_transit = with_transit
        self.with_contamination = with_contamination
        self.contamination_model = contamination_model
        self.achromatic_transit = radius_ratio == 'achromatic'
        self.noise_model = noise_model
        self.radius_ratio = radius_ratio
        self.n_legendre = n_legendre
        self.n_flares = 0
        self.contamination_reference_passband = contamination_reference_passband

        # Set photometry
        # --------------
        self.phs = photometry
        self.nph = len(photometry)
        self.aid = 1

        # Set the aperture ranges
        # -----------------------
        self.min_apt = amin = min(max(aperture_lims[0], 0), photometry[0].flux.aperture.size)
        self.max_apt = amax = max(min(aperture_lims[1], photometry[0].flux.aperture.size), 0)
        self.napt = amax-amin

        self.toi = None

        # Target and comparison star IDs
        # ------------------------------
        self.tid = atleast_1d(tid).astype('int')
        if self.tid.size == 1:
            self.tid = tile(self.tid, self.nph)

        self.cids = atleast_2d(cids).astype('int')
        if self.cids.shape[0] == 1:
            self.cids = atleast_2d(tile(self.cids, (self.nph, 1)))

        assert self.tid.size == self.nph
        assert self.cids.shape[0] == self.nph

        masks = []
        for i, ph in enumerate(self.phs):
            ids = concatenate([[self.tid[i]], self.cids[i]]).astype('int')
            nanmask = isfinite(ph._flux[:, ids, amin:amax+1]).all(['star', 'aperture']).values
            masks.append(nanmask)
            ph._fmask &= nanmask

        times, fluxes = [], []
        for i,ph in enumerate(photometry):
            try:
                times.append(array(ph.bjd))
                f = array(ph.flux[:, self.tid[i], self.aid])
                fluxes.append(f / nanmedian(f))
            except:
                pass

        self.apertures = ones(len(times)).astype('int')

        self._tmin = times[0].min()
        self._tmax = times[0].max()

        self.covnames = 'airmass xshift yshift entropy'.split()
        covariates = []
        for ph in photometry:
            try:
                covs = array(ph.aux)[:,[2,3,4,5]]
                covs[:, 1:] = (covs[:, 1:] - median(covs[:, 1:], 0)) / covs[:, 1:].std(0)
                covariates.append(covs)
            except:
                pass

        wns = [ones(ph.nframes) for ph in photometry]

        if use_opencl:
            import pyopencl as cl
            ctx = cl.create_some_context()
            queue = cl.CommandQueue(ctx)
            tm = QuadraticModelCL(klims=klims, nk=1024, nz=1024, cl_ctx=ctx, cl_queue=queue)
        else:
            tm = QuadraticModel(interpolate=False, klims=klims, nk=1024, nz=1024)

        BaseLPF.__init__(self, target, filters, times, fluxes, wns, arange(len(photometry)), covariates,
                         arange(len(fluxes)), tm = tm, tref=floor(times[0].min()))
        self._start_flares = len(self.ps)

        # Create the target and reference star flux arrays
        # ------------------------------------------------
        self.target_fluxes, self.target_median_fluxes, self.reference_fluxes = [], [], []
        for ip, ph in enumerate(photometry):
            ids = concatenate([[self.tid[ip]], self.cids[ip]])
            flux = ph.flux[:, ids, amin:amax + 1]
            self.target_fluxes.append(array(flux[:, 0, :] / flux[:, 0, :].median('mjd')))
            self.reference_fluxes.append(array(flux[:, 1:, :]))
            self.target_median_fluxes.append(flux[:, 0, :].median('mjd'))
        self.target_median_fluxes = array(self.target_median_fluxes)

        if self.cids.shape[1] == 1:
            self.set_prior('ref_on_0', 'NP', 0.75, 1e-3)

    def _init_parameters(self):
        self.ps = ParameterSet()
        if self.with_transit:
            self._init_p_orbit()
            self._init_p_planet()
            self._init_p_limb_darkening()
        self._init_p_photometry()
        if self.ps:
            self.ps.freeze()

    def _init_baseline(self):
        self._add_baseline_model(LinearModelBaseline(self))

    def _init_lnlikelihood(self):
        self._add_lnlikelihood_model(M2WNLogLikelihood(self))

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
                if self.contamination_model == 'physical':
                    pk2 = [PParameter('k2_app', 'apparent_area_ratio', 'A_s', UP(0.005 ** 2, 0.25 ** 2),
                                      (0.005 ** 2, 0.25 ** 2))]
                    self.ps.add_passband_block('k2', 1, 1, pk2)
                    self._pid_k2 = repeat(self.ps.blocks[-1].start, self.npb)
                    self._start_k2 = self.ps.blocks[-1].start
                    self._sl_k2 = self.ps.blocks[-1].slice
                    pcn = [GParameter('k2_true', 'true_area_ratio', 'A_s', UP(0.005**2, 0.75**2), bounds=(1e-8, inf)),
                           GParameter('teff_h', 'host_teff', 'K', UP(2500, 12000), bounds=(2500, 12000)),
                           GParameter('teff_c', 'contaminant_teff', 'K', UP(2500, 12000), bounds=(2500, 12000))]
                    self.ps.add_global_block('contamination', pcn)
                    self._pid_cn = arange(self.ps.blocks[-1].start, self.ps.blocks[-1].stop)
                    self._sl_cn = self.ps.blocks[-1].slice

                    def contamination_prior(pvp):
                        return where(diff(pvp[:, 4:6])[:, 0] > 0, 0, -inf)
                    self.add_prior(contamination_prior)
                elif self.contamination_model == 'direct':
                    pk2 = [PParameter('k2', 'area_ratio', 'A_s', UP(0.005 ** 2, 0.25 ** 2), (0.005 ** 2, 0.25 ** 2))]
                    self.ps.add_passband_block('k2', 1, 1, pk2)
                    self._pid_k2 = repeat(self.ps.blocks[-1].start, self.npb)
                    self._start_k2 = self.ps.blocks[-1].start
                    self._sl_k2 = self.ps.blocks[-1].slice
                    pcn = []
                    for pb in self.passbands:
                        pcn.append(GParameter(f'c_{pb}', f'{pb} contamination', '', UP(0, 1), bounds=(0, 1)))
                    self.ps.add_global_block('contamination', pcn)
                    self._pid_cn = arange(self.ps.blocks[-1].start, self.ps.blocks[-1].stop)
                    self._sl_cn = self.ps.blocks[-1].slice
                else:
                    raise NotImplementedError

        # 2. Chromatic radius ratio
        # -------------------------
        elif self.radius_ratio == 'chromatic':
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

        else:
            raise NotImplementedError


    def _init_p_photometry(self):
        if not self.photometry_frozen:
            c = [LParameter(f'tap', f'target_aperture', '', UP(0.0, 0.999*(self.napt)),  bounds=(0.0, 0.999*(self.napt)))]
            self.ps.add_global_block('apertures', c)
            self._sl_tap = self.ps.blocks[-1].slice
            self._start_tap = self.ps.blocks[-1].start

            if self.cids.size > 0:
                c = []
                for irf in range(self.cids.shape[1]):
                    c.append(LParameter(f'ref_on_{irf:d}', f'use_comparison_star_{irf:d}', '', UP(0.0, 1.0), bounds=( 0.0, 1.0)))
                self.ps.add_global_block('ref_include', c)
                self._sl_ref_include = self.ps.blocks[-1].slice
                self._start_ref_include = self.ps.blocks[-1].start

                c = []
                for irf in range(self.cids.shape[1]):
                    c.append(LParameter(f'ref_apt_{irf:d}', f'comparison_star_aperture_{irf:d}', '', UP(0.0, 0.999*(self.napt)), bounds=( 0.0, 0.999*(self.napt))))
                self.ps.add_global_block('ref_apts', c)
                self._sl_ref_apt = self.ps.blocks[-1].slice
                self._start_ref_apt = self.ps.blocks[-1].start

    def _init_instrument(self):
        filters = {'g': sdss_g, 'r': sdss_r, 'i':sdss_i, 'z_s':sdss_z}
        self.instrument = Instrument('MuSCAT2', [filters[pb] for pb in self.passbands])
        self.cm = SMContamination(self.instrument, self.contamination_reference_passband)

    def add_flare(self, loc, amp=(0, 0.2)):
        self.n_flares += 1
        iflare = self.n_flares

        ps = self.ps
        ps.thaw()
        fp = [GParameter(f'f{iflare}s', f'flare {iflare} start time', 'd', NP(*loc), [-inf, inf]),
              GParameter(f'f{iflare}w', f'flare {iflare} width', '-', UP(0, 0.01), [0, inf])]
        for j in range(self.npb):
            fp.append(GParameter(f'f{iflare}a{j}', f'flare {iflare} amplitude {j}', '-', UP(*amp), [0, inf]))
        ps.add_global_block(f'flare_{iflare}', fp)
        ps.freeze()

        if iflare == 1:
            self._start_flares = ps.blocks[-1].start
        self._sl_flares = slice(self._start_flares, ps.blocks[-1].stop)

    def create_pv_population(self, npop=50):
        pvp = self.ps.sample_from_prior(npop)

        if hasattr(self, '_sl_lm'):
            for p in self.ps[self._sl_lm]:
                if 'lm_i' in p.name:
                    pvp[:, p.pid] = normal(1.0, 0.005, npop)
                else:
                    pvp[:, p.pid] = normal(0.0, 0.005, npop)

        if self.with_transit:

            # With LDTk
            # ---------
            # Use LDTk to create the sample if LDTk has been initialised.
            if self.ldps:
                istart = self._start_ld
                cms, ces = self.ldps.coeffs_tq()
                for i, (cm, ce) in enumerate(zip(cms.flat, ces.flat)):
                    pvp[:, i + istart] = normal(cm, ce, size=pvp.shape[0])

            # No LDTk
            # -------
            # Ensure that the total limb darkening decreases towards
            # red passbands.
            else:
                pvv = uniform(size=(npop, 2*self.npb))
                pvv[:, ::2] = sort(pvv[:, ::2], 1)[:, ::-1]
                pvv[:, 1::2] = sort(pvv[:, 1::2], 1)[:, ::-1]
                pvp[:,self._sl_ld] = pvv

        return pvp

    def remove_outliers(self, iapt: int, lower: float = 5, upper: float = 5, plot: bool = False, apply: bool = True) -> None:
        fluxes = []
        if plot:
            fig, axs = subplots(1, self.npb, figsize=(13, 4), sharey='all')
        for ipb in range(self.npb):
            ft = self.target_fluxes[ipb][:, iapt] / self.reference_fluxes[ipb][:, :, iapt].sum(1)
            ft /= median(ft)
            dft = diff(ft)
            mask = ~sigma_clip(dft, sigma_lower=inf, sigma_upper=5, stdfunc=mad_std).mask
            ids = where(mask)[0] + 1
            if plot:
                axs[ipb].plot(self.times[ipb][ids], ft[ids], '.')
                nids = where(~mask)[0] + 1
                axs[ipb].plot(self.times[ipb][nids], ft[nids], 'kx')
            if apply:
                fluxes.append(ft[ids])
                self.times[ipb] = self.times[ipb][ids]
                self.target_fluxes[ipb] = self.target_fluxes[ipb][ids, :]
                self.reference_fluxes[ipb] = self.reference_fluxes[ipb][ids, :, :]
                self.covariates[ipb] = self.covariates[ipb][ids, :]
        if apply:
            self._init_data(self.times, fluxes, pbids=self.pbids, covariates=self.covariates, wnids=self.noise_ids)
            self._baseline_models[0].init_data()
        if plot:
            fig.tight_layout()

    def cut(self, tstart: float = -inf, tend: float = inf, plot: bool = True, apply: bool = True, aid: int = None) -> None:
        fluxes = []

        aid = aid if aid is not None else self.aid

        if plot:
            fig, axs = subplots(1, self.npb, figsize=(13, 4), sharey='all')
            axs = atleast_1d(axs)

        for ipb in range(self.npb):
            ft = self.target_fluxes[ipb][:, aid] / nanmedian(self.target_fluxes[ipb][:, aid])

            mask = ~((self.times[ipb] - self._tref > tstart) & (self.times[ipb] - self._tref < tend))

            if plot:
                axs[ipb].plot(self.times[ipb][mask] - self._tref, ft[mask], '.')
                axs[ipb].plot(self.times[ipb][~mask] - self._tref, ft[~mask], 'kx')
            if apply:
                fluxes.append(ft[mask])
                self.times[ipb] = self.times[ipb][mask]
                self.target_fluxes[ipb] = self.target_fluxes[ipb][mask, :]
                self.reference_fluxes[ipb] = self.reference_fluxes[ipb][mask, :, :]
                self.covariates[ipb] = self.covariates[ipb][mask, :]
        if apply:
            self._init_data(self.times, fluxes, pbids=self.pbids, covariates=self.covariates, wnids=self.noise_ids)
            self._baseline_models[0].init_data()
        if plot:
            fig.tight_layout()

    def apply_relative_limits(self, iapt: int, lower: float = -inf, upper: float = inf, plot: bool = True,
                                apply: bool = True, npoly: int = 0, iterations: int = 5, erosion: int = 0) -> None:
        fluxes = []
        if plot:
            fig, axs = subplots(1, self.npb, figsize=(13, 4), sharey='all')
            axs = atleast_1d(axs)
        for ipb in range(self.npb):
            ft = self.target_fluxes[ipb][:, iapt] / self.reference_fluxes[ipb][:, :, iapt].mean(1)
            ft /= nanmedian(ft)
            mask = ones(ft.size, bool)

            if npoly > 0:
                for i in range(iterations):
                    p = Polynomial.fit(self.times[ipb][mask], ft[mask], npoly)
                    baseline = p(self.times[ipb])
                    mask[mask] &= ((ft / baseline)[mask] > lower) & ((ft / baseline)[mask] < upper)
            else:
                baseline = ones_like(ft)
                mask &= (ft > lower) & (ft < upper)

            if erosion > 0:
                mask = binary_erosion(mask, iterations=erosion)

            if plot:
                axs[ipb].plot(self.times[ipb][mask], ft[mask], '.')
                axs[ipb].plot(self.times[ipb][~mask], ft[~mask], 'kx')
                axs[ipb].plot(self.times[ipb], baseline, 'k-')
                axs[ipb].plot(self.times[ipb], baseline * upper, 'k--')
                axs[ipb].plot(self.times[ipb], baseline * lower, 'k--')
            if apply:
                fluxes.append(ft[mask])
                self.times[ipb] = self.times[ipb][mask]
                self.target_fluxes[ipb] = self.target_fluxes[ipb][mask, :]
                self.reference_fluxes[ipb] = self.reference_fluxes[ipb][mask, :, :]
                self.covariates[ipb] = self.covariates[ipb][mask, :]
        if apply:
            self._init_data(self.times, fluxes, pbids=self.pbids, covariates=self.covariates, wnids=self.noise_ids)
            self._baseline_models[0].init_data()
        if plot:
            fig.tight_layout()

    def apply_normalized_limits(self, iapt: int, lower: float = -inf, upper: float = inf, plot: bool = True,
                                apply: bool = True, npoly: int = 0, iterations: int = 5, erosion: int = 0) -> None:
        fluxes = []
        if plot:
            fig, axs = subplots(1, self.npb, figsize=(13, 4), sharey='all')
            axs = atleast_1d(axs)
        for ipb in range(self.npb):
            ft = self.target_fluxes[ipb][:, iapt] / nanmedian(self.target_fluxes[ipb][:, iapt])
            mask = ones(ft.size, bool)

            if npoly > 0:
                for i in range(iterations):
                    p = Polynomial.fit(self.times[ipb][mask], ft[mask], npoly)
                    baseline = p(self.times[ipb])
                    mask[mask] &= ((ft / baseline)[mask] > lower) & ((ft / baseline)[mask] < upper)
            else:
                baseline = ones_like(ft)
                mask &= (ft > lower) & (ft < upper)

            if erosion > 0:
                mask = binary_erosion(mask, iterations=erosion)

            if plot:
                axs[ipb].plot(self.times[ipb][mask], ft[mask], '.')
                axs[ipb].plot(self.times[ipb][~mask], ft[~mask], 'kx')
                axs[ipb].plot(self.times[ipb], baseline, 'k-')
                axs[ipb].plot(self.times[ipb], baseline * upper, 'k--')
                axs[ipb].plot(self.times[ipb], baseline * lower, 'k--')
            if apply:
                fluxes.append(ft[mask])
                self.times[ipb] = self.times[ipb][mask]
                self.target_fluxes[ipb] = self.target_fluxes[ipb][mask, :]
                self.reference_fluxes[ipb] = self.reference_fluxes[ipb][mask, :, :]
                self.covariates[ipb] = self.covariates[ipb][mask, :]
        if apply:
            self._init_data(self.times, fluxes, pbids=self.pbids, covariates=self.covariates, wnids=self.noise_ids)
            self._baseline_models[0].init_data()
        if plot:
            fig.tight_layout()

    def downsample(self, exptime: float) -> None:
        fluxes = []
        for ipb in range(self.npb):
            bt, self.target_fluxes[ipb] = downsample_time(self.times[ipb], self.target_fluxes[ipb], exptime)
            _, self.reference_fluxes[ipb] = downsample_time(self.times[ipb], self.reference_fluxes[ipb], exptime)
            if self.reference_fluxes[ipb].ndim == 2:
                self.reference_fluxes[ipb] = self.reference_fluxes[ipb][:,newaxis,:]
            _, self.covariates[ipb] = downsample_time(self.times[ipb], self.covariates[ipb], exptime)
            self.times[ipb] = bt
            f = self.target_fluxes[ipb][:, -1] / self.reference_fluxes[ipb][:,:,-1].sum(1)
            fluxes.append(f / nanmedian(f))
        self._init_data(self.times, fluxes, pbids=self.pbids, covariates=self.covariates, wnids=self.noise_ids)
        self._baseline_models[0].init_data()

    def set_radius_ratio_prior(self, kmin, kmax):
        for p in self.ps[self._sl_k2]:
            p.prior = UP(kmin ** 2, kmax ** 2)
            p.bounds = [kmin ** 2, kmax ** 2]
        self.ps.thaw()
        self.ps.freeze()

    def target_apertures(self, pv):
        pv = atleast_2d(pv)
        p = floor(clip(pv[:, self._sl_tap], 0., 0.999) * self.napt).astype('int')
        return squeeze(p)

    def reference_apertures(self, pv):
        pv = atleast_2d(pv)
        p = floor(pv[:, self._sl_ref_apt]).astype('int')
        return squeeze(p)

    def set_ofluxa(self, pv):
        self.ofluxa[:] = self.relative_flux(pv)

    def freeze_photometry(self, pv=None):
        if self.photometry_frozen:
            warnings.warn('Trying to freeze already frozen photometry')
            return

        pv = pv if pv is not None else self.de.minimum_location
        ps_orig = self.ps
        self._original_population = pvp = self.de.population.copy()

        if self.cids.size == 0:
            self._frozen_population = delete(pvp, self._sl_tap, 1)
            self._reference_flux = ones_like(self.ofluxa)
            self.raps = []
        else:
            self._frozen_population = delete(pvp, s_[self._sl_tap.start: self._sl_ref_apt.stop], 1)
            self._reference_flux = self.reference_flux(pv)
            self.raps = self.reference_apertures(pv)

        self.taps = self.target_apertures(pv)
        self._target_flux = self.target_flux(pv)
        self.ofluxa[:] = self.relative_flux(pv)

        self.photometry_frozen = True
        self._init_parameters()
        self._baseline_models = []
        self._lnlikelihood_models = []
        self._init_lnlikelihood()
        self._init_baseline()
        self.ps.thaw()
        for p in self.ps:
            iold = ps_orig.names.index(p.name)
            p.prior = ps_orig[iold].prior
            p.bounds = ps_orig[iold].bounds
        if self.n_flares > 0:
            self.ps.add_global_block('flares', ps_orig[self._sl_flares])
            self._sl_flares = self.ps.blocks[-1].slice
            self._start_flares = self.ps.blocks[-1].start
        self.ps.freeze()
        self.de = DiffEvol(self.lnposterior, clip(self.ps.bounds, -1, 1), self.de.n_pop, maximize=True, vectorize=True)
        self.de._population[:, :] = self._frozen_population.copy()
        self.de._fitness[:] = self.lnposterior(self._frozen_population)

    @property
    def frozen_apertures(self):
        return (float(self.phs[0]._ds.aperture[self.taps]) * 0.435 * u.arcsec,
                array(self.phs[0]._ds.aperture[self.raps]) * 0.435 * u.arcsec)

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

    def _transit_model_achromatic_dcnt(self, pvp, copy=True):
        pvp = atleast_2d(pvp)
        pvt = map_pv_achromatic_nocnt(pvp)
        ldc = map_ldc(pvp[:, self._sl_ld])
        cnt = pvp[:, self._sl_cn]
        flux = self.tm.evaluate_pv(pvt, ldc, copy=copy)
        return contaminate(flux, cnt, self.lcids, self.pbids)

    def _transit_model_chromatic_nocnt(self, pvp, copy=True):
        pvp = atleast_2d(pvp)
        pvm = zeros((pvp.shape[0], 6+self.npb))
        pvm[:, :self.npb] = sqrt(pvp[:,self._sl_k2])
        pvm[:, self.npb] = pvp[:, 0] - self._tref
        pvm[:, self.npb+1] = p = pvp[:, 1]
        pvm[:, self.npb+2] = a = as_from_rhop(pvp[:,2], p)
        pvm[:, self.npb+3] = i_from_ba(pvp[:,3], a)
        ldc = map_ldc(pvp[:, self._sl_ld])
        return self.tm.evaluate_pv(pvm, ldc, copy=copy)

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
        if self.with_transit:
            if self.achromatic_transit:
                if self.with_contamination:
                    if self.contamination_model == 'physical':
                        return self._transit_model_achromatic_cnt(pv, copy)
                    else:
                        return self._transit_model_achromatic_dcnt(pv, copy)
                else:
                    return self._transit_model_achromatic_nocnt(pv, copy)
            else:
                if self.with_contamination:
                    return self._transit_model_chromatic_cnt(pv, copy)
                else:
                    return self._transit_model_chromatic_nocnt(pv, copy)
        else:
            return 1.

    def flare_model(self, pvp):
        pvp = atleast_2d(pvp)
        return squeeze(flare(self.timea, self.lcids, self.pbids, pvp[:, self._sl_flares], self.npb, self.n_flares))

    def trends(self, pv):
        """Additive trends"""
        if self.n_flares > 0:
            return self.flare_model(pv)
        else:
            return 0.

    def flux_model(self, pv):
        baseline = self.baseline(pv)
        trends   = self.trends(pv)
        transit  = self.transit_model(pv, copy=True)
        flux = baseline * transit + trends
        return flux.astype('d')

    def relative_flux(self, pv):
        if self.photometry_frozen:
            return self.ofluxa
        else:
            return self.target_flux(pv) / self.reference_flux(pv)

    def target_flux(self, pv):
        if self.photometry_frozen:
            return self._target_flux
        pv = atleast_2d(pv)
        p = floor(clip(pv[:, self._sl_tap], 0.0, self.napt*0.999)).astype('int')
        off = zeros((p.shape[0], self.timea.size))
        for i, sl in enumerate(self.lcslices):
            off[:, sl] = self.target_fluxes[i][:, p].T
        return squeeze(off)

    def reference_flux(self, pvp):
        if self.photometry_frozen:
            return self._reference_flux
        if self.cids.size > 0:
            pvp = atleast_2d(pvp)
            ref_mask = pvp[:, self._sl_ref_include] >= 0.5
            ref_apertures = floor(clip(pvp[:, self._sl_ref_apt], 0.0, 0.999*self.napt)).astype(int)
            ref_fluxes = empty((pvp.shape[0], self.timea.size))
            for i, rf in enumerate(self.reference_fluxes):
                ref_fluxes[:, self.lcslices[i]] = create_reference_flux(rf, ref_mask, ref_apertures)
            return squeeze(where(isfinite(ref_fluxes), ref_fluxes, inf))
        else:
            return 1.

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

    def add_ldtk_prior(self, teff: tuple, logg: tuple, z: tuple,
                       uncertainty_multiplier: float = 3,
                       pbs: tuple = ('g', 'r', 'i', 'z'), cache = None) -> None:
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
        self.ldsc = LDPSetCreator(teff, logg, z, filters,cache=cache)
        self.ldps = self.ldsc.create_profiles(1000)
        self.ldps.resample_linear_z()
        self.ldps.set_uncertainty_multiplier(uncertainty_multiplier)
        def ldprior(pv):
            pv = atleast_2d(pv)
            lnl = zeros(pv.shape[0])
            for i in range(pv.shape[0]):
                lnl[i] = self.ldps.lnlike_tq(pv[i, self._sl_ld])
            return lnl
        self.add_prior(ldprior)

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
        self.add_prior(is_inside_window)

    def transit_bic(self):
        from scipy.optimize import minimize

        def bic(ll1, ll2, d1, d2, n):
            return ll2 - ll1 + 0.5 * (d1 - d2) * log(n)

        pv_all = self.de.minimum_location.copy()
        pv_bl = pv_all[self._sl_lm]
        wn = atleast_2d(10 ** pv_all[self._sl_wn])

        def lnlikelihood_baseline(self, pv):
            pv_all[self._sl_lm] = pv
            flux_m = self.baseline(pv_all)

            if self.photometry_frozen:
                lnl = squeeze(lnlike_logistic_v1d(self.ofluxa, flux_m, wn, self.lcids))
            else:
                lnl = squeeze(lnlike_logistic_v(self.relative_flux(pv), flux_m, wn, self.lcids))

            return lnl if lnl == lnl else -inf

        ll_no_transit = -minimize(lambda pv: -lnlikelihood_baseline(self, pv), pv_bl).fun
        ll_with_transit = float(self.lnlikelihood(self.de.minimum_location.copy()))
        d_with_transit = len(self.ps)
        d_no_transit = d_with_transit - self._start_lm
        #return bic(ll_with_transit, ll_no_transit, d_with_transit, d_no_transit, self.timea.size)
        return bic(ll_no_transit, ll_with_transit, d_no_transit, d_with_transit, self.timea.size)

    def posterior_samples(self, burn: int = 0, thin: int = 1, derived_parameters: bool = True, add_tref = True):
        df = BaseLPF.posterior_samples(self, burn, thin, derived_parameters)
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
            err = 10 ** pv[self._sl_wn]
            if not self.photometry_frozen:
                self.set_ofluxa(pv)

        elif model == 'mc':
            fc = array(self.posterior_samples(derived_parameters=False, add_tref=False))
            pv = permutation(fc)[:300]
            err = 10 ** median(pv[:, self._sl_wn], 0)
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
            t = self.timea[sl] - self._tref
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

        setp(axs[1:,:], xlim=(self.timea.min()-self._tref, self.timea.max()-self._tref))

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
        axs[0, 1].hist(df.tc - self._tref, bins='doane', density=True, range=array(trange) - self._tref, alpha=0.5, edgecolor='k', histtype='stepfilled')
        axs[0, 1].set_xlabel(f'Transit center - {self._tref:.0f} [BJD]')
        axs[0, 1].fill_between(x - self._tref, exp(p.logpdf(x)), alpha=0.5, edgecolor='k')

        # Period
        # ------
        p = self.ps.priors[1]
        trange = p.mean - 3 * p.std, p.mean + 3 * p.std
        x = linspace(*trange)
        axs[1, 1].hist(df.p, bins='doane', density=True, range=trange, alpha=0.5, edgecolor='k', histtype='stepfilled')
        axs[1, 1].set_xlabel('Period [days]')
        axs[1, 1].fill_between(x, exp(p.logpdf(x)), alpha=0.5, edgecolor='k')
        setp(axs[1, 1], xticks=df.p.quantile([0.05, 0.5, 0.95]))

        # Rest without priors
        # -------------------
        names = 'stellar density, impact parameter'.split(', ')
        for i, ax in enumerate(axs.flat[[2, 5]]):
            ax.hist(df.iloc[:, i + 2], bins='doane', density=True, alpha=0.5, edgecolor='k', histtype='stepfilled')
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

    def plot_combined_and_binned(self, binwidth: float = 10., figsize=None, ax=None,
                                 plot_unbinned: bool = False):
        from pytransit.lpf.tesslpf import downsample_time

        if ax is None:
            fig, ax = subplots(1, 1, figsize=figsize)
        else:
            fig, ax = None, ax

        pv = self.de.minimum_location

        sids = argsort(self.timea)

        rflux = self.relative_flux(pv) / squeeze(self.baseline(pv))
        mflux = self.transit_model(pv)

        bt, bfo, be = downsample_time(self.timea[sids], rflux[sids], binwidth / 24 / 60)
        btm, bfm, _ = downsample_time(self.timea[sids], mflux[sids], 2 / 24 / 60)

        if plot_unbinned:
            ax.plot(self.timea - self._tref, rflux, 'k.', alpha=0.1)
        ax.errorbar(bt - self._tref, bfo, be, fmt='ok')
        ax.fill_between(bt - self._tref, bfo - 3 * be, bfo + 3 * be, alpha=0.1, step='mid')
        ax.plot(btm - self._tref, bfm, 'k')
        ax.text(0.95, 0.05, f'Binning: {binwidth} min', ha='right', transform=ax.transAxes)
        setp(ax, ylabel='Normalised flux', xlabel=f'Time - {self._tref:.0f} [BJD]')
        return fig

    def save_fits(self, filename: str) -> None:
        phdu = pf.PrimaryHDU()
        phdu.header.update(target=self.name)
        hdul = pf.HDUList(phdu)

        pv = self.de.minimum_location
        time = self.timea

        baseline = squeeze(self.baseline(pv))
        trends = squeeze(self.trends(pv))
        if isscalar(trends):
            trends = zeros_like(baseline)

        if self.with_transit:
            transit = squeeze(self.transit_model(pv).astype('d'))
        else:
            transit = ones_like(time)

        if not self.photometry_frozen:
            target_flux = self.target_flux(pv)
            reference_flux = self.reference_flux(pv)
            relative_flux = target_flux / reference_flux
            detrended_flux = relative_flux / baseline - trends
        else:
            target_flux = self._target_flux
            reference_flux = self._reference_flux
            relative_flux = self.ofluxa
            detrended_flux = relative_flux / baseline - trends

        if self.cids.size == 0:
            reference_flux = ones_like(target_flux)

        for i, sl in enumerate(self.lcslices):
            df = Table(transpose([time[sl] + self._tref, detrended_flux[sl], relative_flux[sl], target_flux[sl],
                                  reference_flux[sl], baseline[sl], transit[sl], trends[sl]]),
                       names='time_bjd flux flux_rel flux_trg flux_ref baseline model trends'.split(),
                       meta={'extname': f"flux_{self.passbands[i]}", 'filter': self.passbands[i], 'trends': 'linear', 'wn': self.wn[i], 'radrat': self.radius_ratio})
            hdul.append(pf.BinTableHDU(df))

        for i, pb in enumerate(self.passbands):
            df = Table(self.covariates[i], names='sky xshift yshift entropy'.split(), meta={'extname': f'aux_{pb}'})
            hdul.append(pf.BinTableHDU(df))
        hdul.writeto(filename+'.fits', overwrite=True)
