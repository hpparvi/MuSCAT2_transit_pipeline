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

import math as mt
import warnings
from os.path import join, split

from emcee import EnsembleSampler
from george import GP
from george.kernels import ExpSquaredKernel as ESK, ConstantKernel as CK
from numba import njit
from numpy import (ones, zeros_like, median, concatenate, dot, ones_like,
                   clip, argsort, any, s_, arccos, nan, isnan, full, sum, repeat, arange)
from numpy.linalg import lstsq, LinAlgError
from pyde import DiffEvol
from pytransit import MandelAgol as MA
from pytransit.contamination import SMContamination
from pytransit.contamination.filter import sdss_g, sdss_r, sdss_i, sdss_z
from pytransit.contamination.instrument import Instrument
from pytransit.ma_quadratic_nb import eval_quad_ip_mp
from pytransit.orbits_py import z_circular, duration_eccentric, as_from_rhop
from pytransit.param.parameter import *
from pytransit.param.parameter import PParameter, GParameter, LParameter
from pytransit.param.parameter import UniformPrior as U, NormalPrior as N, GammaPrior as GM
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

from scipy.stats import t as tdist

@njit(cache=False)
def lnlike_normal(o, m, e):
    return -sum(log(e)) -0.5*p.size*log(2.*pi) - 0.5*sum((o-m)**2/e**2)

@njit("f8(f8[:], f8[:], f8)", cache=False)
def lnlike_normal_s(o, m, e):
    return -o.size*log(e) -0.5*o.size*log(2.*pi) - 0.5*sum((o-m)**2)/e**2

@njit("f8[:](f8[:], f8[:])", fastmath=False, cache=False)
def unpack_orbit(pv, zpv):
    zpv[:2] = pv[:2]
    zpv[2] = as_from_rhop(pv[2], pv[1])
    if zpv[2] <= 1.:
        zpv[2] = nan
    else:
        zpv[3] = arccos(pv[3] / zpv[2])
    return zpv

@njit("f8[:,:](f8[:], f8[:,:])", fastmath=True, cache=False)
def qq_to_uv(pv, uv):
    a, b = sqrt(pv[::2]), 2.*pv[1::2]
    uv[:,0] = a * b
    uv[:,1] = a * (1. - b)
    return uv


class BaseLPF:

    models = "pb_independent_k pb_dependent_k pb_dependent_contamination physical_contamination".split()

    def __init__(self, target, datasets, filters, model='pb_independent_k', **kwargs):
        assert (model in self.models), 'Model must be one of:\n\t' + ', '.join(self.models)
        if 'free_k' in kwargs.keys() or 'contamination' in kwargs.keys():
            warnings.warn(
                DeprecationWarning('free_k and contamination are deprecated. Please set the model instead.'))
        self.model = model
        self.tm = MA(interpolate=True, klims=(0.01, 0.75), nk=512, nz=512)
        self.target = target
        self.datasets = datasets
        self.filters = filters
        self.npb = npb = len(filters)
        self.use_oec = kwargs.get('use_oec', True)
        self.planet = None

        self.de = None
        self.sampler = None
        self.ldsc = None
        self.ldps = None

        if datasets is not None:
            self.nlc = datasets.size
            self.datasets.mask_outliers(5)
            self.times = self.datasets.btimes
            self.fluxes = self.datasets.bfluxes
            self.covariates = self.datasets.bcovariates
            self.wn = [median(wn) for wn in self.datasets.bwn]
            self.timea = concatenate(self.times)
            self.ofluxa = concatenate(self.fluxes)
            self.mfluxa = zeros_like(self.ofluxa)
            self.pbida = concatenate([full(t.size, ds.pbid) for t, ds in zip(self.times, self.datasets)])
            self.lcida = concatenate([full(t.size, i) for i, t in enumerate(self.times)])

            self.lcslices = []
            sstart = 0
            for i in range(self.nlc):
                s = self.times[i].size
                self.lcslices.append(s_[sstart:sstart + s])
                sstart += s

        # Read the planet parameters from the OEC
        # ---------------------------------------
        if self.use_oec:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                import exodata
                exocat = exodata.OECDatabase(join(split(__file__)[0], '../ext/oec/systems/'))
            self.planet = exocat.searchPlanet(target)

        p = self.planet.P if self.planet else kwargs.get('period', 5.)
        t0 = datasets[0].time.mean() if datasets is not None else 0.0
        tce = datasets[0].time.ptp() / 10 if datasets is not None else 1e-5

        # Set up the instrument and contamination model
        # --------------------------------------------
        self.instrument = Instrument('MuSCAT2', [sdss_g, sdss_r, sdss_i, sdss_z])
        self.cm = SMContamination(self.instrument, "i'")

        # Initialise the lnprior hook list
        # --------------------------------
        self.lnpriors = []

        # Setup parametrisation
        # =====================

        # Basic system parameters
        # -----------------------
        self.ps = ps = ParameterSet()
        psystem = [
            GParameter('tc', 'zero_epoch', 'd', N(t0, tce), (-inf, inf)),
            GParameter('pr', 'period', 'd', N(p, 1e-5), (0, inf)),
            GParameter('rho', 'stellar_density', 'g/cm^3', U(0.1, 25.0), (0, inf)),
            GParameter('b', 'impact_parameter', 'R_s', U(0.0, 1.0), (0, 1))]
        ps.add_global_block('system', psystem)

        # Radius ratio and contamination
        # ------------------------------
        # We have four scenarios for the radius ratio and contamination
        #
        #  1. Separate radius ratio for each passband (nongray atmosphere)
        #  2. Common radius ratio for each passband (gray atmosphere)
        #  3. Common radius ratio with possible contamination
        #  4. Common radius ratio with physically based contamination
        #
        if model == 'pb_dependent_k':
            pk2 = [PParameter('k2_{}'.format(pb), 'area_ratio', 'A_s', GM(0.1), (0.01 ** 2, 0.55 ** 2)) for pb in
                   filters]
            ps.add_passband_block('k2', 1, npb, pk2)
            self._pid_k2 = arange(ps.blocks[-1].start, ps.blocks[-1].stop)
            self._start_k2 = ps.blocks[-1].start
            self._sl_k2 = ps.blocks[-1].slice
            self._pid_cn = None
        elif model == 'pb_independent_k':
            pk2 = [PParameter('k2', 'area_ratio', 'A_s', GM(0.1), (0.01 ** 2, 0.55 ** 2))]
            ps.add_passband_block('k2', 1, 1, pk2)
            self._pid_k2 = repeat(ps.blocks[-1].start, npb)
            self._start_k2 = ps.blocks[-1].start
            self._sl_k2 = ps.blocks[-1].slice
            self._pid_cn = None
        elif model == 'pb_dependent_contamination':
            pk2 = [PParameter('k2', 'area_ratio', 'A_s', GM(0.1), (0.01 ** 2, 0.55 ** 2))]
            pcn = [PParameter('cnt_{}'.format(pb), 'contamination', '', U(0., 1.), (0., 1.)) for pb in filters]
            ps.add_passband_block('k2', 1, 1, pk2)
            self._pid_k2 = repeat(ps.blocks[-1].start, npb)
            self._start_k2 = ps.blocks[-1].start
            self._sl_k2 = ps.blocks[-1].slice
            ps.add_passband_block('contamination', 1, npb, pcn)
            self._pid_cn = arange(ps.blocks[-1].start, ps.blocks[-1].stop)
        elif model == 'physical_contamination':
            pk2 = [PParameter('k2_app', 'apparent_area_ratio', 'A_s', GM(0.1), (0.01 ** 2, 0.55 ** 2))]
            pcn = [GParameter('k2_true', 'true_area_ratio', 'As', GM(0.1), bounds=(1e-8, inf)),
                   GParameter('teff_h', 'host_teff', 'K', U(2500, 12000), bounds=(2500, 12000)),
                   GParameter('teff_c', 'contaminant_teff', 'K', U(2500, 12000), bounds=(2500, 12000))]
            ps.add_passband_block('k2', 1, 1, pk2)
            self._pid_k2 = repeat(ps.blocks[-1].start, npb)
            self._start_k2 = ps.blocks[-1].start
            self._sl_k2 = ps.blocks[-1].slice
            ps.add_global_block('contamination', pcn)
            self._pid_cn = arange(ps.blocks[-1].start, ps.blocks[-1].stop)
            self.lnpriors.append(lambda pv: 0.0 if pv[4] < pv[5] else -inf)

        # Limb darkening
        # --------------
        pld = concatenate([
            [PParameter('q1_{:d}'.format(i), 'q1_coefficient', '', U(0, 1), bounds=(0, 1)),
             PParameter('q2_{:d}'.format(i), 'q2_coefficient', '', U(0, 1), bounds=(0, 1))]
            for i in range(npb)])
        ps.add_passband_block('ldc', 2, npb, pld)
        self._sl_ld = ps.blocks[-1].slice
        self._start_ld = ps.blocks[-1].start
        ps.freeze()

        # Set the radius ratio and contamination model
        # --------------------------------------------
        if model in self.models[:2]:
            self.transit_model = self.uncontaminated_transit_model
        elif model == 'pb_dependent_contamination':
            self.transit_model = self.contaminated_transit_model_free
        elif model == 'physical_contamination':
            self.transit_model = self.contaminated_transit_model_phys

        # Initialise the temporary arrays
        # -------------------------------
        self._zpv = zeros(6)
        self._tuv = zeros((self.npb, 2))
        self._zeros = zeros(self.npb)
        self._ones = ones(self.npb)

        if datasets is not None:
            self._bad_fluxes = [ones_like(t) for t in self.times]
        else:
            self._bad_fluxes = None


    def mask_outliers(self, sigma=5, mf_width=10, means=None):
        self.datasets.mask_outliers(sigma=sigma, mf_width=mf_width, means=means)
        self.times = self.datasets.times
        self.fluxes = self.datasets.fluxes
        self.covariates = self.datasets.covariates
        self.wn = [median(wn) for wn in self.datasets.wn]


    def create_pv_population(self, npop=50):
        pvp = self.ps.sample_from_prior(npop)
        for sl in self.ps.blocks[1].slices:
            pvp[:,sl] = uniform(0.01**2, 0.25**2, size=(npop, 1))
        if self.ldps:
            istart = self._start_ld
            cms, ces = self.ldps.coeffs_tq()
            for i, (cm, ce) in enumerate(zip(cms.flat, ces.flat)):
                pvp[:, i + istart] = normal(cm, ce, size=pvp.shape[0])
        else:
            ldsl = self._sl_ld
            for i in range(pvp.shape[0]):
                pid = argsort(pvp[i, ldsl][::2])[::-1]
                pvp[i, ldsl][::2] = pvp[i, ldsl][::2][pid]
                pvp[i, ldsl][1::2] = pvp[i, ldsl][1::2][pid]
        if self.model == 'pb_dependent_contamination':
            pvp[:,5] = pvp[:,4]
            cref = uniform(0, 0.99, size=npop)
            pvp[:,5] = pvp[:,4] / (1. - cref)
        return pvp

    def baseline(self, pv):
        """Flux baseline (multiplicative)"""
        return ones(self.nlc)

    def trends(self, pv):
        """Systematic trends (additive)"""
        return zeros(self.nlc)

    def uncontaminated_transit_model(self, pv):
        zpv = unpack_orbit(pv, self._zpv)
        if isnan(zpv[2]):
            fluxes = self._bad_fluxes
        else:
            _k = sqrt(pv[self._pid_k2])
            uv = qq_to_uv(pv[self._sl_ld], self._tuv)
            z = z_circular(self.timea, zpv)
            fluxes = eval_quad_ip_mp(z, self.pbida, _k, uv, self._zeros, self.tm.ed, self.tm.ld, self.tm.le, self.tm.kt,
                                     self.tm.zt)
            fluxes = [fluxes[sl] for sl in self.lcslices]
        return fluxes

    def contaminated_transit_model_free(self, pv):
        cnt = pv[self._pid_cn]
        zpv = unpack_orbit(pv, self._zpv)
        if isnan(zpv[2]):
            fluxes = self._bad_fluxes
        else:
            _k = full(self.npb, sqrt(pv[4]))
            uv = qq_to_uv(pv[self._sl_ld], self._tuv)
            z = z_circular(self.timea, zpv)
            fluxes = eval_quad_ip_mp(z, self.pbida, _k, uv, cnt, self.tm.ed, self.tm.ld, self.tm.le, self.tm.kt,
                                     self.tm.zt)
            fluxes = [fluxes[sl] for sl in self.lcslices]
        return fluxes

    def contaminated_transit_model_phys(self, pv):
        cnref = 1. - pv[4] / pv[5]
        cnt = self.cm.contamination(cnref, pv[6], pv[7])
        zpv = unpack_orbit(pv, self._zpv)
        if isnan(zpv[2]):
            fluxes = self._bad_fluxes
        else:
            _k = full(self.npb, sqrt(pv[5]))
            uv = qq_to_uv(pv[self._sl_ld], self._tuv)
            z = z_circular(self.timea, zpv)
            fluxes = eval_quad_ip_mp(z, self.pbida, _k, uv, cnt, self.tm.ed, self.tm.ld, self.tm.le, self.tm.kt,
                                     self.tm.zt)
            fluxes = [fluxes[sl] for sl in self.lcslices]
        return fluxes

    def flux_model(self, pv):
        bls = self.baseline(pv)
        trs = self.trends(pv)
        tms = self.transit_model(pv)
        return [tm*bl + tr for tm,bl,tr in zip(tms, bls, trs)]

    def residuals(self, pv):
        return [fo - fm for fo, fm in zip(self.fluxes, self.flux_model(pv))]

    def add_t14_prior(self, m, s):
        def T14(pv):
            a = as_from_rhop(pv[2], pv[1])
            t14 = duration_eccentric(pv[1], sqrt(pv[4]), a, mt.acos(pv[3] / a), 0, 0, 1)
            return norm.logpdf(t14, m, s)
        self.lnpriors.append(T14)

    def add_as_prior(self, m, s):
        def as_prior(pv):
            a = as_from_rhop(pv[2], pv[1])
            return norm.logpdf(a, m, s)
        self.lnpriors.append(as_prior)

    def add_ldtk_prior(self, teff, logg, z, uncertainty_multiplier=3, pbs=('g', 'r', 'i', 'z')):
        from ldtk import LDPSetCreator
        from ldtk.filters import sdss_g, sdss_r, sdss_i, sdss_z
        fs = {n: f for n, f in zip('g r i z'.split(), (sdss_g, sdss_r, sdss_i, sdss_z))}
        filters = [fs[k] for k in pbs]
        self.ldsc = LDPSetCreator(teff, logg, z, filters)
        self.ldps = self.ldsc.create_profiles(1000)
        self.ldps.resample_linear_z()
        self.ldps.set_uncertainty_multiplier(uncertainty_multiplier)
        def ldprior(pv):
            return self.ldps.lnlike_tq(pv[self._sl_ld])
        self.lnpriors.append(ldprior)


    def lnprior(self, pv):
        return self.ps.lnprior(pv)

    def lnprior_hooks(self, pv):
        """Additional constraints."""
        return sum([f(pv) for f in self.lnpriors])

    def lnlikelihood(self, pv):
        flux_m = self.flux_model(pv)
        lnlike = 0.0
        for i in range(self.nlc):
            lnlike += lnlike_normal_s(self.fluxes[i], flux_m[i], self.wn[i])
        return lnlike

    def lnposterior(self, pv):
        if any(pv < self.ps.bounds[:, 0]) or any(pv > self.ps.bounds[:, 1]):
            return -inf
        else:
            return self.lnprior(pv) + self.lnlikelihood(pv) + self.lnprior_hooks(pv)

    def __call__(self, pv):
        return self.lnposterior(pv)

    def optimize_global(self, niter=200, npop=50, population=None, label='Global optimisation'):
        if self.de is None:
            self.de = DiffEvol(self.lnposterior, clip(self.ps.bounds, -1, 1), npop, maximize=True)
            if population is None:
                self.de._population[:, :] = self.create_pv_population(npop)
            else:
                self.de._population[:,:] = population
        for _ in tqdm(self.de(niter), total=niter, desc=label):
            pass

    def sample_mcmc(self, niter=500, thin=5, label='MCMC sampling', reset=False):
        if self.sampler is None:
            self.sampler = EnsembleSampler(self.de.n_pop, self.de.n_par, self.lnposterior)
            pop0 = self.de.population
        else:
            pop0 = self.sampler.chain[:,-1,:].copy()
        if reset:
            self.sampler.reset()
        for _ in tqdm(self.sampler.sample(pop0, iterations=niter, thin=thin), total=niter, desc=label):
            pass


class NormalLSqLPF(BaseLPF):
    """Log posterior function with least-squares baseline and normal likelihood"""

    def trends(self, pv):
        trends = []
        for i, (tm,bl) in enumerate(zip(self.transit_model(pv), self.baseline(pv))):
            coefs = lstsq(self.covariates[i], self.fluxes[i] - tm*bl, rcond=None)[0]
            trends.append(dot(self.covariates[i], coefs))
        return trends

    def flux_model(self, pv):
        model_fluxes = self.transit_model(pv)
        baselines = self.baseline(pv)
        for i, (tm,bl) in enumerate(zip(model_fluxes, baselines)):
            coefs = lstsq(self.covariates[i], self.fluxes[i] - tm*bl, rcond=None)[0]
            fbl = dot(self.covariates[i], coefs)
            model_fluxes[i] = tm*bl + fbl
        return model_fluxes


    def lnlikelihood(self, pv):
        try:
            return super(NormalLSqLPF, self).lnlikelihood(pv)
        except LinAlgError:
            return -inf

class StudentLSqLPF(NormalLSqLPF):
    """Log posterior function with least-squares baseline and Student T likelihood"""

    def __init__(self, target, datasets, filters, model='pb_independent_k', **kwargs):
        super().__init__(target, datasets, filters, model, **kwargs)
        self.ps.thaw()
        perr = [LParameter('et_{:d}'.format(i), 'error_df', '', U(1e-6, 1), bounds=(1e-6, 1)) for i in range(self.npb)]
        self.ps.add_lightcurve_block('error', 1, self.npb, perr)
        self.ps.freeze()
        self._sler = self.ps.blocks[-1].slices

    def lnlikelihood(self, pv):
        flux_m = self.flux_model(pv)
        lnlike = 0.0
        for i, sl in enumerate(self._sler):
            sigma, df = self.wn[i], 1. / pv[sl]
            lnlike += tdist.logpdf(self.fluxes[i] - flux_m[i], df, 0., sigma).sum()
        return lnlike


class GPLPF(BaseLPF):
    def __init__(self, target, datasets, filters, model='pb_independent_k', fit_wn=True, **kwargs):
        super().__init__(target, datasets, filters, model, **kwargs)

        pbl = [LParameter('bl_{:d}'.format(i), 'baseline', '', N(1, 1e-2), bounds=(-inf,inf)) for i in range(self.nlc)]
        self.ps.thaw()
        self.ps.add_lightcurve_block('baseline', 1, self.nlc, pbl)
        self.ps.freeze()

        self._slbl = self.ps.blocks[-1].slice
        self._stbl = self.ps.blocks[-1].start

        self.logwnvar = log(array(self.wn) ** 2)
        self._create_kernel()
        self.covariates = [cv[:, self.covids] for cv in self.covariates]
        self.freeze = []
        self.standardize = []

        self.gps = [GP(self._create_kernel(),
                       mean=0., fit_mean=False,
                       white_noise=0.8*wn, fit_white_noise=fit_wn) for wn in self.logwnvar]

        # Freeze the GP hyperparameters marked as frozen in _create_kernel
        for gp in self.gps:
            pars = gp.get_parameter_names()
            [gp.freeze_parameter(pars[i]) for i in self.freeze]

        # Standardize the covariates marked to be standardized in _create_kernel
        for c in self.covariates:
            for i in self.standardize:
                c[:,i] = (c[:,i] - c[:,i].min()) / c[:,i].ptp()

        self.compute_always = False
        self.compute_gps()
        self.de = None
        self.gphpres = None
        self.gphps = None

    def compute_gps(self, hps=None):
        if hps is not None:
            [gp.set_parameter_vector(hp) for gp, hp in zip(self.gps, hps)]
        [gp.compute(cv) for gp, cv, wn in zip(self.gps, self.covariates, self.wn)]

    def optimize_hps(self, pv, method='L-BFGS-B'):
        self.gphpres = []
        self.gphps = []
        trends = [t-t.mean() for t in [o-m for o,m in zip(self.fluxes, self.transit_model(pv))]]

        for gp, residual in tqdm(zip(self.gps, trends), total=self.nlc, desc='Optimizing GP hyperparameters'):
            def nll(hp):
                gp.set_parameter_vector(hp)
                l = gp.log_likelihood(residual, quiet=True)
                g = gp.grad_log_likelihood(residual, quiet=True)
                return -l, -g
            res = minimize(nll, gp.get_parameter_vector(), jac=True, bounds=gp.get_parameter_bounds(), method=method)
            self.gphpres.append(res)
            self.gphps.append(res.x)

    def optimize_hps_jointly(self, pv, method='L-BFGS-B'):
        self.gphpres = []
        self.gphps = []
        trends = [t-t.mean() for t in [o-m for o,m in zip(self.fluxes, self.transit_model(pv))]]

        def nll(hp):
            [gp.set_parameter_vector(hp) for gp in self.gps]
            l = sum([gp.log_likelihood(res, quiet=True) for res,gp in zip(trends, self.gps)])
            g = sum([gp.grad_log_likelihood(res, quiet=True) for res,gp in zip(trends, self.gps)])
            return -l, -g

        res = minimize(nll, self.gps[0].get_parameter_vector(), jac=True, bounds=self.gps[0].get_parameter_bounds(),
                       method=method)
        self.gphpres.append(res)
        self.gphps.append(res.x)
        self.gphpres = 4*self.gphpres
        self.gphps = 4*self.gphps


    def _create_kernel(self):
        self.covids = (2, 3, 4, 5, 6)
        self.standardize = []
        nd = len(self.covids)
        logv = self.logwnvar.mean()
        kernel_sk = ESK(10, bounds=((None, None),), ndim=nd, axes=0)
        kernel_am = ESK(1, bounds=((None, None),), ndim=nd, axes=1)
        kernel_xy = ESK(2, metric_bounds=((0, 20),), ndim=nd, axes=[2, 3])
        kernel_en = ESK(1, bounds=((None, None),), ndim=nd, axes=4)
        self.kernels = kernel_sk, kernel_am, kernel_xy, kernel_en
        return CK(logv, bounds=((-18, -7),), ndim=nd) * kernel_sk * kernel_am * kernel_xy * kernel_en


    def predict(self, pv, return_var=False):
        self.compute_gps(self.gphps)
        return [gp.predict(residuals, cv, return_cov=False, return_var=return_var)
                for gp, residuals, cv in zip(self.gps, self.residuals(pv), self.covariates)]


    def baseline(self, pv):
        """Flux baseline (multiplicative)"""
        return pv[self._slbl]

    def lnlikelihood(self, pv):
        if self.compute_always:
            self.compute_gps()
        lnlike = 0.0
        for gp, residuals in zip(self.gps, self.residuals(pv)):
            lnlike += gp.log_likelihood(residuals)
        return lnlike
