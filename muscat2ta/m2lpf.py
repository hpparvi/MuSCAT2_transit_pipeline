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
    sqrt, squeeze, floor, linspace, pi, c_, any, all, percentile, median, repeat, mean, newaxis, isfinite, pad, clip, \
    delete, s_, log10, argsort, atleast_1d, tile, any, fabs, zeros_like, sort, ones_like
from numpy.polynomial.legendre import legvander
from numpy.random import permutation, uniform, normal
from pytransit import QuadraticModel, QuadraticModelCL
from pytransit.contamination import SMContamination
from pytransit.contamination.filter import sdss_g, sdss_r, sdss_i, sdss_z
from pytransit.contamination.instrument import Instrument
from pytransit.lpf.lpf import BaseLPF, map_pv, map_ldc
from pytransit.orbits.orbits_py import as_from_rhop, duration_eccentric, i_from_ba, d_from_pkaiews, epoch
from pytransit.param.parameter import NormalPrior as NP, UniformPrior as UP, LParameter, PParameter, ParameterSet
from pytransit.utils.de import DiffEvol
from scipy.stats import logistic, norm
from uncertainties import ufloat


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

class M2LPF(BaseLPF):
    def __init__(self, target: str, photometry: list, tid: int, cids: list,
                 filters: tuple, aperture_lims: tuple = (0, inf), use_opencl: bool = False,
                 n_legendre: int = 0, use_toi_info=True, with_transit=True, with_contamination=False,
                 radius_ratio: str = 'achromatic'):
        assert radius_ratio in ('chromatic', 'achromatic')

        self.use_opencl = use_opencl
        self.planet = None

        self.photometry_frozen = False
        self.with_transit = with_transit
        self.with_contamination = with_contamination
        self.chromatic_transit = radius_ratio == 'chromatic'
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

        self.covnames = 'intercept sky airmass xshift yshift entropy'.split()

        times =  [array(ph.bjd) for ph in photometry]
        fluxes = [array(ph.flux[:, tid, 1]) for tid,ph in zip(self.tid, photometry)]
        fluxes = [f/nanmedian(f) for f in fluxes]
        self.apertures = ones(len(times)).astype('int')

        self.tref = floor(times[0].min())
        times = [t - self.tref for t in times]

        self._tmin = times[0].min()
        self._tmax = times[0].max()

        covariates = []
        for ph in photometry:
            covs = concatenate([ones([ph._fmask.sum(), 1]), array(ph.aux)[:,[1,3,4,5]]], 1)
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

        super().__init__(target, filters, times, fluxes, wns, arange(len(photometry)), covariates, arange(len(photometry)), tm = tm)

        self.legendre = [legvander((t - t.min())/(0.5*t.ptp()) - 1, self.n_legendre)[:,1:] for t in self.times]

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
        if self.radius_ratio == 'achromatic':
            pk2 = [PParameter('k2', 'area_ratio', 'A_s', UP(0.005**2, 0.25**2), (0.005**2, 0.25**2))]
            self.ps.add_passband_block('k2', 1, 1, pk2)
            self._pid_k2 = repeat(self.ps.blocks[-1].start, self.npb)
        else:
            pk2 = [PParameter(f'k2_{pb}', f'area_ratio {pb}', 'A_s', UP(0.005**2, 0.25**2), (0.005**2, 0.25**2)) for pb in self.passbands]
            self.ps.add_passband_block('k2', 1, self.npb, pk2)
            self._pid_k2 = arange(self.npb) + self.ps.blocks[-1].start
        self._start_k2 = self.ps.blocks[-1].start
        self._sl_k2 = self.ps.blocks[-1].slice

        if self.with_contamination:
            pcn = [PParameter('cnt_ref', 'Reference contamination', '', UP(0., 1.), (0., 1.))]
            pcn.extend([PParameter(f'cnt_{pb}', 'contamination', '', UP(-1., 1.), (-1., 1.)) for pb in self.passbands[1:]])
            self.ps.add_passband_block('contamination', 1, self.npb, pcn)
            self._pid_cn = arange(self.ps.blocks[-1].start, self.ps.blocks[-1].stop)
            self._sl_cn = self.ps.blocks[-1].slice


    def _init_p_baseline(self):
        c = []
        for ilc in range(self.nlc):
            c.append(LParameter(f'ci_{ilc:d}', 'intercept_{ilc:d}', '', NP(1.0, 0.03), bounds=( 0.5, 1.5)))
            c.append(LParameter(f'cs_{ilc:d}', 'sky_{ilc:d}',       '', NP(0.0, 0.01), bounds=(-0.5, 0.5)))
            c.append(LParameter(f'ca_{ilc:d}', 'airmass_{ilc:d}',   '', NP(0.0, 0.01), bounds=(-0.5, 0.5)))
            c.append(LParameter(f'cx_{ilc:d}', 'xshift_{ilc:d}',    '', NP(0.0, 0.01), bounds=(-0.5, 0.5)))
            c.append(LParameter(f'cy_{ilc:d}', 'yshift_{ilc:d}',    '', NP(0.0, 0.01), bounds=(-0.5, 0.5)))
            c.append(LParameter(f'ce_{ilc:d}', 'entropy_{ilc:d}',   '', NP(0.0, 0.01), bounds=(-0.5, 0.5)))
        self.ps.add_lightcurve_block('ccoef', 6, self.nlc, c)
        self._sl_ccoef = self.ps.blocks[-1].slice
        self._start_ccoef = self.ps.blocks[-1].start

        if self.n_legendre > 0:
            c = []
            for ilc in range(self.nlc):
                for ilg in range(self.n_legendre):
                    c.append(LParameter(f'leg_{ilc:d}_{ilg:d}', f'legendre__{ilc:d}_{ilg:d}', '', NP(0.0, 0.01), bounds=(-0.5, 0.5)))
            self.ps.add_lightcurve_block('legendre_polynomials', self.n_legendre, self.nlc, c)
            self._sl_leg = self.ps.blocks[-1].slice
            self._start_leg = self.ps.blocks[-1].start

        if not self.photometry_frozen:
            c = []
            for ilc in range(self.nlc):
                c.append(LParameter(f'tap_{ilc:d}', f'target_aperture__{ilc:d}', '', UP(0.0, 0.999),  bounds=(0.0, 0.999)))
            self.ps.add_lightcurve_block('apertures', 1, self.nlc, c)
            self._sl_tap = self.ps.blocks[-1].slice
            self._start_tap = self.ps.blocks[-1].start

            if self.cids.size > 0:
                c = []
                for ilc in range(self.nlc):
                    for irf in range(self.cids.shape[1]):
                        c.append(LParameter(f'ref_{irf:d}_{ilc:d}', f'comparison_star_{irf:d}_{ilc:d}', '', UP(0.0, 0.999), bounds=( 0.0, 0.999)))
                self.ps.add_lightcurve_block('rstars', self.cids.shape[1], self.nlc, c)
                self._sl_ref = self.ps.blocks[-1].slice
                self._start_ref = self.ps.blocks[-1].start

    def _init_p_noise(self):
        """Noise parameter initialisation.
        """
        pns = [LParameter('loge_{:d}'.format(i), 'log10_error_{:d}'.format(i), '', UP(-4, 0), bounds=(-4, 0)) for i in
               range(self.n_noise_blocks)]
        self.ps.add_lightcurve_block('log_err', 1, self.n_noise_blocks, pns)
        self._sl_err = self.ps.blocks[-1].slice
        self._start_err = self.ps.blocks[-1].start

    def _init_instrument(self):
        self.instrument = Instrument('MuSCAT2', [sdss_g, sdss_r, sdss_i, sdss_z])
        self.cm = SMContamination(self.instrument, "i'")

    def create_pv_population(self, npop=50):
        pvp = self.ps.sample_from_prior(npop)
        if self.with_transit:
            #for sl in self.ps.blocks[1].slices:
            #    pvp[:,sl] = uniform(0.01**2, 0.25**2, size=(npop, 1))

            if self.with_contamination:
                p = pvp[:, self._sl_cn]
                p[:, 1]  = uniform(size=npop)
                p[:, 1:] = normal(0, 0.2, size=(npop, self.npb - 1))
                p[:, 1:] = clip(p[:, 0][:, newaxis] + p[:, 1:], 0.001, 0.999) - p[:, 0][:, newaxis]

            # With LDTk
            # ---------
            #
            # Use LDTk to create the sample if LDTk has been initialised.
            #
            if self.ldps:
                istart = self._start_ld
                cms, ces = self.ldps.coeffs_tq()
                for i, (cm, ce) in enumerate(zip(cms.flat, ces.flat)):
                    pvp[:, i + istart] = normal(cm, ce, size=pvp.shape[0])

            # No LDTk
            # -------
            #
            # Ensure that the total limb darkening decreases towards
            # red passbands.
            #
            else:
                pvv = uniform(size=(npop, 2*self.npb))
                pvv[:, ::2] = sort(pvv[:, ::2], 1)[:, ::-1]
                pvv[:, 1::2] = sort(pvv[:, 1::2], 1)[:, ::-1]
                pvp[:,self._sl_ld] = pvv
                #for i in range(pvp.shape[0]):
                #    pid = argsort(pvp[i, ldsl][::2])[::-1]
                #    pvp[i, ldsl][::2] = pvp[i, ldsl][::2][pid]
                #    pvp[i, ldsl][1::2] = pvp[i, ldsl][1::2][pid]

        # Estimate white noise from the data
        # ----------------------------------
        for i in range(self.nlc):
            wn = diff(self.ofluxa).std() / sqrt(2)
            pvp[:, self._start_err] = log10(uniform(0.5*wn, 2*wn, size=npop))
        return pvp

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
        self.photometry_frozen = True
        self._init_parameters()
        if self.with_transit:
            for i in range(self._start_ld):
                self.ps[i].prior = ps_orig[i].prior
                self.ps[i].bounds = ps_orig[i].bounds
        self.de = DiffEvol(self.lnposterior, clip(self.ps.bounds, -1, 1), self.de.n_pop, maximize=True, vectorize=True)
        self.de._population[:,:] = self._frozen_population.copy()
        self.de._fitness[:] = self.lnposterior(self._frozen_population)


    def transit_model(self, pv, copy=True):
        if self.with_transit:
            pv = atleast_2d(pv)
            if self.chromatic_transit:
                mean_ar = pv[:, self._sl_k2].mean(1)
                pvv = zeros((pv.shape[0], pv.shape[1]-self.npb+1))
                pvv[:, :4] = pv[:, :4]
                pvv[:, 4] = mean_ar
                pvv[:, 5:] = pv[:, 4+self.npb:]

                pvp = map_pv(pvv)
                ldc = map_ldc(pvv[:, 5:5 + 2 * self.npb])
                flux = self.tm.evaluate_pv(pvp, ldc, copy)
                rel_ar = pv[:, self._sl_k2] / mean_ar[:,newaxis]
                flux = change_depth(rel_ar, flux, self.lcids, self.pbids)
            else:
                flux = super().transit_model(pv, copy)

            if self.with_contamination:
                p = pv[:, self._sl_cn]
                pv_cnt = zeros((pv.shape[0], self.npb))
                pv_cnt[:,0] = p[:,0]
                pv_cnt[:,1:] = p[:,1:] + p[:,0][:, newaxis]
                bad = any(pv_cnt < 0.0, 1)
                flux = contaminate(flux, pv_cnt, self.lcids, self.pbids)
                flux[bad,0] = inf
            return flux
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
            off[:, sl] = self.ofluxes[i][:, p[:, i]].T
        return squeeze(off)

    def reference_flux(self, pv):
        if self.cids.size > 0:
            pv = atleast_2d(pv)
            p = floor(clip(pv[:, self._sl_ref], 0., 0.999) * self.napt+1).astype('int')
            r = zeros((pv.shape[0], self.ofluxa.size))
            nref = self.cids.shape[1]
            for ipb, sl in enumerate(self.lcslices):
                for i in range(nref):
                    r[:, sl] += self.refs[ipb][i][:, p[:, ipb * nref + i]].T
                r[:, sl] = r[:, sl] / median(r[:, sl], 1)[:, newaxis]
            return squeeze(where(isfinite(r), r, inf))
        else:
            return 1.

    def extinction(self, pv):
        pv = atleast_2d(pv)
        ext = zeros((pv.shape[0], self.timea.size))
        for i, sl in enumerate(self.lcslices):
            st = self._start_ccoef + i * 6
            ext[:, sl] = exp(- pv[:, st + 2] * self.airmasses[i][:, newaxis]).T
            ext[:, sl] /= mean(ext[:, sl], 1)[:, newaxis]
        return squeeze(ext)

    def baseline(self, pv):
        pv = atleast_2d(pv)
        bl = zeros((pv.shape[0], self.timea.size))
        for i,sl in enumerate(self.lcslices):
            st = self._start_ccoef + i*6
            p = pv[:, st:st+6]
            #bl[:, sl] = (self.covariates[i] @ p[:,[0,1,3,4,5]].T).T
            bl[:, sl] = (self.covariates[i][:,[0,2,3]] @ p[:,[0,3,4]].T).T

        if self.n_legendre > 0:
            for i, sl in enumerate(self.lcslices):
                st = self._start_leg + i * self.n_legendre
                p = pv[:, st:st + self.n_legendre]
                bl[:, sl] += (self.legendre[i] @ p.T).T
        bl = bl * self.extinction(pv)
        return bl

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
            if not self.photometry_frozen:
                self.set_ofluxa(pv)

        elif model == 'mc':
            fc = array(self.posterior_samples(derived_parameters=False))
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
        t0 = self.tref

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
        setp(axs[3, :], xlabel=f'Time - {self.tref:9.0f} [BJD]')
        setp(axs[0, :], xlabel='Residual [ppt]', yticks=[])
        [sb.despine(ax=ax, offset=5, left=True) for ax in axs[0]]
        return fig, axs


    def plot_posteriors(self, figsize: tuple = (13, 5), fig=None, gridspec=None):
        if fig is None:
            fig = figure(figsize=figsize, constrained_layout=True)
        axs = fig.subplots(2, 3, gridspec_kw=gridspec)

        df = self.posterior_samples()
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

        setp(axs, yticks=[])
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