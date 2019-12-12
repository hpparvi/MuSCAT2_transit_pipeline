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
from astropy.table import Table
from ldtk import LDPSetCreator
from matplotlib.pyplot import subplots, setp, figure
from astropy.io import fits as pf
from muscat2ph.catalog import get_toi
from numba import njit, prange
from numpy import atleast_2d, zeros, exp, log, array, nanmedian, concatenate, ones, arange, where, diff, inf, arccos, \
    sqrt, squeeze, floor, linspace, pi, c_, any, all, percentile, median, repeat, mean, newaxis, isfinite, pad, clip, \
    delete, s_, log10, argsort, atleast_1d, tile, any, fabs, zeros_like, sort, ones_like, fmin, digitize, ceil, full, \
    nan, transpose
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import legvander
import astropy.units as u
from numpy.random import permutation, uniform, normal
from pytransit import QuadraticModel, QuadraticModelCL, BaseLPF, LinearModelBaseline
from pytransit.contamination import SMContamination
from pytransit.contamination.filter import sdss_g, sdss_r, sdss_i, sdss_z
from pytransit.contamination.instrument import Instrument
from pytransit.lpf.lpf import BaseLPF, map_pv, map_ldc
from pytransit.orbits.orbits_py import as_from_rhop, duration_eccentric, i_from_ba, d_from_pkaiews, epoch
from pytransit.param.parameter import NormalPrior as NP, UniformPrior as UP, LParameter, PParameter, ParameterSet, \
    GParameter
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


class M2LPF(LinearModelBaseline, BaseLPF):
    def __init__(self, target: str, photometry: list, tid: int, cids: list,
                 filters: tuple, aperture_lims: tuple = (0, inf), use_opencl: bool = False,
                 n_legendre: int = 0, use_toi_info=True, with_transit=True, with_contamination=False,
                 radius_ratio: str = 'achromatic', noise_model='white', klims=(0.005, 0.75)):
        assert radius_ratio in ('chromatic', 'achromatic')
        assert noise_model in ('white', 'gp')

        self.use_opencl = use_opencl
        self.planet = None

        self.photometry_frozen = False
        self.with_transit = with_transit
        self.with_contamination = with_contamination
        self.achromatic_transit = radius_ratio == 'achromatic'
        self.noise_model = noise_model
        self.radius_ratio = radius_ratio
        self.n_legendre = n_legendre

        # Set photometry
        # --------------
        self.phs = photometry
        self.nph = len(photometry)

        # Set the aperture ranges
        # -----------------------
        self.min_apt = amin = min(max(aperture_lims[0], 0), photometry[0].flux.aperture.size)
        self.max_apt = amax = max(min(aperture_lims[1], photometry[0].flux.aperture.size), 0)
        self.napt = amax-amin

        self.toi = None

        # Target and comparison star IDs
        # ------------------------------
        self.tid = atleast_1d(tid)
        if self.tid.size == 1:
            self.tid = tile(self.tid, self.nph)

        self.cids = atleast_2d(cids)
        if self.cids.shape[0] == 1:
            self.cids = atleast_2d(tile(self.cids, (self.nph, 1)))

        assert self.tid.size == self.nph
        assert self.cids.shape[0] == self.nph

        times =  [array(ph.bjd) for ph in photometry]
        fluxes = [array(ph.flux[:, tid, 1]) for tid,ph in zip(self.tid, photometry)]
        fluxes = [f/nanmedian(f) for f in fluxes]
        self.apertures = ones(len(times)).astype('int')

        self.tref = floor(times[0].min())
        times = [t - self.tref for t in times]

        self._tmin = times[0].min()
        self._tmax = times[0].max()

        self.covnames = 'airmass xshift yshift entropy'.split()
        covariates = []
        for ph in photometry:
            covs = array(ph.aux)[:,[2,3,4,5]]
            covs[:, 1:] = (covs[:, 1:] - median(covs[:, 1:], 0)) / covs[:, 1:].std(0)
            covariates.append(covs)

        wns = [ones(ph.nframes) for ph in photometry]

        if use_opencl:
            import pyopencl as cl
            ctx = cl.create_some_context()
            queue = cl.CommandQueue(ctx)
            tm = QuadraticModelCL(klims=klims, nk=1024, nz=1024, cl_ctx=ctx, cl_queue=queue)
        else:
            tm = QuadraticModel(interpolate=True, klims=klims, nk=1024, nz=1024)

        BaseLPF.__init__(self, target, filters, times, fluxes, wns, arange(len(photometry)), covariates, arange(len(photometry)), tm = tm)

        # Create the target and reference star flux arrays
        # ------------------------------------------------
        self.ofluxes = [array(ph.flux[:, self.tid[i], amin:amax+1] / ph.flux[:, self.tid[i], amin:amax+1].median('mjd')) for i,ph in enumerate(photometry)]

        self.refs = []
        for ip, ph in enumerate(photometry):
            self.refs.append([pad(array(ph.flux[:, cid, amin:amax+1]), ((0, 0), (1, 0)), mode='constant') for cid in self.cids[ip]])

    def _init_parameters(self):
        self.ps = ParameterSet()
        if self.with_transit:
            self._init_p_orbit()
            self._init_p_planet()
            self._init_p_limb_darkening()
        self._init_p_baseline()
        self._init_p_noise()
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

    def _init_p_baseline(self):
        LinearModelBaseline._init_p_baseline(self)

        if not self.photometry_frozen:
            c = [LParameter(f'tap', f'target_aperture', '', UP(0.0, 0.999),  bounds=(0.0, 0.999))]
            self.ps.add_global_block('apertures', c)
            self._sl_tap = self.ps.blocks[-1].slice
            self._start_tap = self.ps.blocks[-1].start

            if self.cids.size > 0:
                c = []
                for irf in range(self.cids.shape[1]):
                    c.append(LParameter(f'ref_{irf:d}', f'comparison_star_{irf:d}', '', UP(0.0, 0.999), bounds=( 0.0, 0.999)))
                self.ps.add_global_block('rstars', c)
                self._sl_ref = self.ps.blocks[-1].slice
                self._start_ref = self.ps.blocks[-1].start

    def _init_p_noise(self):
        """Noise parameter initialisation.
        """

        if self.noise_model == 'gp':
            pgp = [GParameter('log_gpa', 'log10_gp_amplitude', '', UP(-4,  0), bounds=(-4,  0)),
                   GParameter('log_gps', 'log10_gp_tscale',    '', UP(-5, 10), bounds=(-5, 10))]
            self.ps.add_global_block('gp', pgp)
            self._sl_gp = self.ps.blocks[-1].slice
            self._start_gp = self.ps.blocks[-1].start
        pns = [LParameter('loge_{:d}'.format(i), 'log10_error_{:d}'.format(i), '', UP(-4, 0), bounds=(-4, 0)) for i in range(self.n_noise_blocks)]
        self.ps.add_lightcurve_block('log_err', 1, self.n_noise_blocks, pns)
        self._sl_err = self.ps.blocks[-1].slice
        self._start_err = self.ps.blocks[-1].start

    def _init_instrument(self):
        filters = {'g': sdss_g, 'r': sdss_r, 'i':sdss_i, 'z_s':sdss_z}
        self.instrument = Instrument('MuSCAT2', [filters[pb] for pb in self.passbands])
        self.cm = SMContamination(self.instrument, self.instrument.filters[0].name)

    def create_pv_population(self, npop=50):
        pvp = self.ps.sample_from_prior(npop)
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
                #for i in range(pvp.shape[0]):
                #    pid = argsort(pvp[i, ldsl][::2])[::-1]
                #    pvp[i, ldsl][::2] = pvp[i, ldsl][::2][pid]
                #    pvp[i, ldsl][1::2] = pvp[i, ldsl][1::2][pid]

        # Baseline coefficients
        # ---------------------
        for i,p in enumerate(self.ps[self._sl_bl]):
            pvp[:, self._start_bl+i] = normal(p.prior.mean, 0.2*p.prior.std, size=npop)

        # Estimate white noise from the data
        # ----------------------------------
        for i in range(self.nlc):
            pvp[:, self._start_err+i] = log10(uniform(0.5*self.wn[i], 2*self.wn[i], size=npop))
        return pvp

    def remove_outliers(self, iapt: int, lower: float = 5, upper: float = 5, plot: bool = False, apply: bool = True) -> None:
        fluxes = []
        if plot:
            fig, axs = subplots(1, self.npb, figsize=(13, 4), sharey='all')
        for ipb in range(self.npb):
            ft = self.ofluxes[ipb][:, iapt] / sum([r[:, iapt] for r in self.refs[ipb]], 0)
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
                self.ofluxes[ipb] = self.ofluxes[ipb][ids, :]
                for icid in range(self.cids.shape[1]):
                    self.refs[ipb][icid] = self.refs[ipb][icid][ids, :]
                self.covariates[ipb] = self.covariates[ipb][ids, :]
        if apply:
            self._init_data(self.times, fluxes, pbids=self.pbids, covariates=self.covariates, wnids=self.noise_ids)
        if plot:
            fig.tight_layout()

    def apply_normalized_limits(self, iapt: int, lower: float = -inf, upper: float = inf, plot: bool = True,
                                apply: bool = True, npoly: int = 0, iterations: int = 5, erosion: int = 0) -> None:
        fluxes = []
        if plot:
            fig, axs = subplots(1, self.npb, figsize=(13, 4), sharey='all')
        for ipb in range(self.npb):
            ft = self.ofluxes[ipb][:, iapt] / nanmedian(self.ofluxes[ipb][:, iapt])
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
                self.ofluxes[ipb] = self.ofluxes[ipb][mask, :]
                for icid in range(self.cids.shape[1]):
                    self.refs[ipb][icid] = self.refs[ipb][icid][mask, :]
                self.covariates[ipb] = self.covariates[ipb][mask, :]
        if apply:
            self._init_data(self.times, fluxes, pbids=self.pbids, covariates=self.covariates, wnids=self.noise_ids)
        if plot:
            fig.tight_layout()

    def downsample(self, exptime: float) -> None:
        nrefs = self.cids.shape[1]
        fluxes = []
        for ipb in range(self.npb):
            bt, self.ofluxes[ipb] = downsample_time(self.times[ipb], self.ofluxes[ipb], exptime)
            for icid in range(nrefs):
                _, self.refs[ipb][icid] = downsample_time(self.times[ipb], self.refs[ipb][icid], exptime)
            _, self.covariates[ipb] = downsample_time(self.times[ipb], self.covariates[ipb], exptime)
            self.times[ipb] = bt
            f = self.ofluxes[ipb][:, -1] / sum([r[:, -1] for r in self.refs[ipb]], 0)
            fluxes.append(f / nanmedian(f))
        self._init_data(self.times, fluxes, pbids=self.pbids, covariates=self.covariates, wnids=self.noise_ids)

    def set_radius_ratio_prior(self, kmin, kmax):
        for p in self.ps[self._sl_k2]:
            p.prior = UP(kmin ** 2, kmax ** 2)
            p.bounds = [kmin ** 2, kmax ** 2]
        self.ps.thaw()
        self.ps.freeze()

    def target_apertures(self, pv):
        pv = atleast_2d(pv)
        p = floor(clip(pv[:, self._sl_tap], 0., 0.999) * (self.napt)).astype('int')
        return squeeze(p)

    def reference_apertures(self, pv):
        pv = atleast_2d(pv)
        p = floor(clip(pv[:, self._sl_ref], 0., 0.999) * (self.napt+1)).astype('int')
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
            self._frozen_population = delete(pvp, s_[self._sl_tap.start: self._sl_ref.stop], 1)
            self._reference_flux = self.reference_flux(pv)
            self.raps = self.reference_apertures(pv)

        self.taps = self.target_apertures(pv)
        self._target_flux = self.target_flux(pv)
        self.ofluxa[:] = self.relative_flux(pv)
        start_tap = self._start_tap
        start_err = self._start_err
        npar_orig = len(ps_orig)

        self.photometry_frozen = True
        self._init_parameters()
        if self.with_transit:
            self.ps.thaw()
            for i in range(0, start_tap):
                self.ps[i].prior = ps_orig[i].prior
                self.ps[i].bounds = ps_orig[i].bounds
            for i,j in enumerate(range(start_err, npar_orig)):
                self.ps[start_tap + i].prior = ps_orig[j].prior
                self.ps[start_tap + i].bounds = ps_orig[j].bounds
            self.ps.freeze()
        self.de = DiffEvol(self.lnposterior, clip(self.ps.bounds, -1, 1), self.de.n_pop, maximize=True, vectorize=True)
        self.de._population[:,:] = self._frozen_population.copy()
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

    def relative_flux(self, pv):
        return self.target_flux(pv) / self.reference_flux(pv)

    def target_flux(self, pv):
        pv = atleast_2d(pv)
        p = floor(clip(pv[:, self._sl_tap], 0., 0.999) * self.napt).astype('int')
        off = zeros((p.shape[0], self.timea.size))
        for i, sl in enumerate(self.lcslices):
            off[:, sl] = self.ofluxes[i][:, p].T
        return squeeze(off)

    def reference_flux(self, pv):
        if self.cids.size > 0:
            pv = atleast_2d(pv)
            p = floor(clip(pv[:, self._sl_ref], 0., 0.999) * self.napt + 1).astype('int')
            r = zeros((pv.shape[0], self.ofluxa.size))
            nref = self.cids.shape[1]
            for ipb, sl in enumerate(self.lcslices):
                for iref in range(nref):
                    r[:, sl] += self.refs[ipb][iref][:, p[:, iref]].T
                r[:, sl] = r[:, sl] / median(r[:, sl], 1)[:, newaxis]
            return squeeze(where(isfinite(r), r, inf))
        else:
            return 1.

    def lnlikelihood(self, pv):
        flux_m = self.flux_model(pv)
        wn = 10**(atleast_2d(pv)[:,self._sl_err])
        if self.photometry_frozen:
            return lnlike_logistic_v1d(self.ofluxa, flux_m, wn, self.lcids)
        else:
            return lnlike_logistic_v(self.relative_flux(pv), flux_m, wn, self.lcids)

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
        self.lnpriors.append(ldprior)

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
        #return bic(ll_with_transit, ll_no_transit, d_with_transit, d_no_transit, self.timea.size)
        return bic(ll_no_transit, ll_with_transit, d_no_transit, d_with_transit, self.timea.size)

    def posterior_samples(self, burn: int = 0, thin: int = 1, derived_parameters: bool = True, add_tref = True):
        df = BaseLPF.posterior_samples(self, burn, thin, derived_parameters)
        if add_tref:
            df.tc += self.tref
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
        setp(axs[3, :], xlabel=f'Time - {self.tref:9.0f} [BJD]')
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
        axs[0, 1].hist(df.tc - self.tref, 50, density=True, range=trange, alpha=0.5, edgecolor='k',
                       histtype='stepfilled')
        axs[0, 1].set_xlabel(f'Transit center - {self.tref:9.0f} [BJD]')
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

    def save_fits(self, filename: str) -> None:
        phdu = pf.PrimaryHDU()
        phdu.header.update(target=self.name)
        hdul = pf.HDUList(phdu)

        pv = self.de.minimum_location
        time = self.timea

        baseline = squeeze(self.baseline(pv))
        if self.with_transit:
            transit = squeeze(self.transit_model(pv).astype('d'))
        else:
            transit = ones_like(time)

        if not self.photometry_frozen:
            target_flux = self.target_flux(pv)
            reference_flux = self.reference_flux(pv)
            relative_flux = target_flux / reference_flux
            detrended_flux = relative_flux / baseline
        else:
            target_flux = self._target_flux
            reference_flux = self._reference_flux
            relative_flux = self.ofluxa
            detrended_flux = relative_flux / baseline

        if self.cids.size == 0:
            reference_flux = ones_like(target_flux)

        for i, sl in enumerate(self.lcslices):
            df = Table(transpose([time[sl] + self.tref, detrended_flux[sl], relative_flux[sl], target_flux[sl],
                                  reference_flux[sl], baseline[sl], transit[sl]]),
                       names='time_bjd flux flux_rel flux_trg flux_ref baseline model'.split(),
                       meta={'extname': f"flux_{self.passbands[i]}", 'filter': self.passbands[i], 'trends': 'linear', 'wn': self.wn[i], 'radrat': self.radius_ratio})
            hdul.append(pf.BinTableHDU(df))

        for i, pb in enumerate(self.passbands):
            df = Table(self.covariates[i], names='sky xshift yshift entropy'.split(), meta={'extname': f'aux_{pb}'})
            hdul.append(pf.BinTableHDU(df))
        hdul.writeto(filename+'.fits', overwrite=True)
