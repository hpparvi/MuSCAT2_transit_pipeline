import warnings
import math as mt
from os.path import join, split

from numpy import (inf, sqrt, ones, hstack, zeros_like, median, floor, concatenate, dot, diff, log, ones_like,
                   percentile, clip, argsort, any, array, s_, zeros)
from numpy.linalg import lstsq
from numpy.random import uniform, normal
from scipy.stats import norm as N, uniform as U, gamma as GM, scoreatpercentile as sap
from scipy.optimize import minimize
from tqdm import tqdm
from astropy.stats import sigma_clip, mad_std

from emcee import EnsembleSampler

from george import GP
from george.kernels import ExpKernel as EK, ExpSquaredKernel as ESK, ConstantKernel as CK, Matern32Kernel as M32
from george.kernels import LinearKernel as LK

from pytransit import MandelAgol as MA
from pytransit.param.basicparameterization import BasicCircularParameterization as BCP
from pytransit.orbits_f import orbits as of
from pytransit.contamination import SMContamination, apparent_radius_ratio, true_radius_ratio
from pytransit.contamination.filter import sdss_g, sdss_r, sdss_i, sdss_z
from pytransit.contamination.instrument import Instrument

from exotk.utils.orbits import as_from_rhop
from exotk.utils.likelihood import ll_normal_es
from exotk.utils.misc import fold

from pyde import DiffEvol

import exodata

warnings.filterwarnings("ignore", category=UserWarning)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    exocat = exodata.OECDatabase(join(split(__file__)[0], '../ext/oec/systems/'))

from scipy.stats import t as tdist

from muscat2ta.parameter import *
from muscat2ta.parameter import UniformPrior as U


class BaseLPF:
    def __init__(self, target, datasets, filters, nthreads=1, free_k=False, contamination=False, **kwargs):
        assert (not free_k) or (free_k != contamination), 'Cannot set both free_k and contamination on'
        self.tm = MA(interpolate=False, nthr=nthreads)
        self.target = target
        self.datasets = datasets
        self.filters = filters
        self.nthr = nthreads
        self.npb = npb = len(filters)
        self.free_k = free_k
        self.contamination = contamination

        self.datasets.mask_outliers(5)
        self.times = self.datasets.btimes
        self.fluxes = self.datasets.bfluxes
        self.covariates = self.datasets.bcovariates
        self.wn = [median(wn) for wn in self.datasets.bwn]

        self.de = None
        self.sampler = None

        # Read the planet parameters from the OEC
        # ---------------------------------------
        self.planet = exocat.searchPlanet(target)
        p = self.planet.P if self.planet else 5.
        t0 = datasets[0].time.mean()
        tce = datasets[0].time.ptp() / 10

        # Set up the instrument and contamination model
        # --------------------------------------------
        self.instrument = Instrument('MuSCAT2', [sdss_g, sdss_r, sdss_i, sdss_z])
        self.cm  = SMContamination(self.instrument, "i'")

        # Set up the parametrisation and priors
        # -------------------------------------
        psystem = [
            GParameter('tc', 'zero_epoch', 'd', N(t0, tce), (-inf, inf)),
            GParameter('pr', 'period', 'd', N(p, 1e-5), (0, inf)),
            GParameter('rho', 'stellar_density', 'g/cm^3', U(0.1, 5.0), (0, inf)),
            GParameter('b', 'impact_parameter', 'R_s', U(0.0, 1.0), (0, 1))]

        # We have three possible scenarios for the radius ratio parametrisation
        #
        #  1. Separate radius ratio for each passband (nongray atmosphere)
        #  2. Common radius ratio for each passband (gray atmosphere)
        #  3. Common radius ratio with possible contamination
        #
        if free_k:
            pk2 = [PParameter('k2_{}'.format(pb), 'area_ratio', 'A_s', GM(0.1), (1e-8, inf)) for pb in 'g r i z'.split()]
        else:
            psystem.append(GParameter('k2', 'area_ratio', 'A_s', GM(0.1), (1e-8, inf)))

        if contamination:
            psystem.extend([GParameter('k2_app', 'apparent_area_ratio', 'As', GM(0.1), bounds=(1e-8, inf)),
                            GParameter('teff_h', 'host_teff', 'K',        U(2500, 12000), bounds=(2500, 12000)),
                            GParameter('teff_c', 'contaminant_teff', 'K', U(2500, 12000), bounds=(2500, 12000))])

        pld = concatenate([
            [PParameter('q1_{:d}'.format(i), 'q1_coefficient', '', U(0, 1), bounds=(0, 1)),
             PParameter('q2_{:d}'.format(i), 'q2_coefficient', '', U(0, 1), bounds=(0, 1))]
            for i in range(npb)])


        self.ps = ps = ParameterSet()
        ps.add_global_block('system', psystem)
        if free_k:
            ps.add_passband_block('k2', 1, npb, pk2)
        ps.add_passband_block('ldc', 2, npb, pld)
        ps.freeze()

        self.teff_prior = U(2500, 12000)

        # Define the parameter slices
        # ---------------------------
        self._slk2 = ps.blocks[1].slices if free_k else 4*[s_[4:5]]
        self._slld = ps.blocks[-1].slices

        # Setup basic parametrisation
        # ---------------------------
        self.par = BCP()

        self.transit_model = self.contaminated_transit_model if contamination else self.uncontaminated_transit_model


    def mask_outliers(self, sigma=5, mf_width=10, means=None):
        self.datasets.mask_outliers(sigma=sigma, mf_width=mf_width, means=means)
        self.times = self.datasets.times
        self.fluxes = self.datasets.fluxes
        self.covariates = self.datasets.covariates
        self.wn = [median(wn) for wn in self.datasets.wn]

    def create_pv_population(self, npop=50):
        pvp = self.ps.sample_from_prior(npop)
        for sl in self._slk2:
            pvp[:,sl] = uniform(1e-8, 0.2**2, size=(npop, 1))
        #for i in range(self.npb):
        #    pvp[:,self._stbl+i] = normal(sap(self.fluxes[i], 95), 2e-3, size=npop)
        ldsl = self.ps.blocks[1].slice
        for i in range(pvp.shape[0]):
            pid = argsort(pvp[i, ldsl][::2])[::-1]
            pvp[i, ldsl][::2] = pvp[i, ldsl][::2][pid]
            pvp[i, ldsl][1::2] = pvp[i, ldsl][1::2][pid]
        if self.contamination:
            pvp[:,5] = pvp[:,4]
            cref = uniform(0, 1, size=npop)
            pvp[:,4] = pvp[:,5] / (1. - cref)
        return pvp

    def baseline(self, pv):
        """Flux baseline (multiplicative)"""
        return ones(self.npb)

    def trends(self, pv):
        """Systematic trends (additive)"""
        return zeros(self.npb)

    def uncontaminated_transit_model(self, pv):
        """Base transit shape model"""
        _a = as_from_rhop(pv[2], pv[1])
        if _a <= 1.:
            return [ones_like(f) for f in self.times]
        _i = mt.acos(pv[3] / _a)
        fluxes = []
        for i, (t, s) in enumerate(zip(self.times, self._slld)):
            _k = mt.sqrt(pv[self._slk2[i]])
            q1, q2 = pv[s]
            a, b = sqrt(q1), 2 * q2
            _uv = array([a * b, a * (1. - b)]).T
            fluxes.append(self.tm.evaluate(t, _k, _uv, pv[0], pv[1], _a, _i))
        return fluxes

    def contaminated_transit_model(self, pv):
        cnref = 1. - pv[5]/pv[4]
        cnt = self.cm.contamination(cnref, pv[6], pv[7])
        _a = as_from_rhop(pv[2], pv[1])
        if _a <= 1.:
            return [ones_like(f) for f in self.times]
        _i = mt.acos(pv[3] / _a)
        _k = sqrt(pv[4])
        fluxes = []
        for i, (t, s, c) in enumerate(zip(self.times, self._slld, cnt)):
            q1, q2 = pv[s]
            a, b = sqrt(q1), 2 * q2
            _uv = array([a * b, a * (1. - b)]).T
            fluxes.append(self.tm.evaluate(t, _k, _uv, pv[0], pv[1], _a, _i, c=c))
        return fluxes

    def flux_model(self, pv):
        bls = self.baseline(pv)
        trs = self.trends(pv)
        tms = self.transit_model(pv)
        return [tm*bl + tr for tm,bl,tr in zip(tms, bls, trs)]

    def residuals(self, pv):
        return [fo - fm for fo, fm in zip(self.fluxes, self.flux_model(pv))]

    def lnprior(self, pv):
        return self.ps.lnprior(pv)

    def lnprior_ext(self, pv):
        """Additional constraints."""
        if self.contamination:
            c = 1. - pv[5]/pv[4]
            return (1-c) * self.teff_prior.logpdf(pv[6]) + c * self.teff_prior.logpdf(pv[7])
        else:
            return 0.

    def lnlikelihood(self, pv):
        flux_m = self.flux_model(pv)
        lnlike = 0.0
        for i in range(self.npb):
            lnlike += ll_normal_es(self.fluxes[i], flux_m[i], self.wn[i])
        return lnlike

    def lnposterior(self, pv):
        if any(pv < self.ps.bounds[:, 0]) or any(pv > self.ps.bounds[:, 1]):
            return -inf
        elif any(diff(sqrt(pv[self.ps.blocks[1].slice][::2])) > 0.):
            return -inf
        elif self.contamination and (pv[5] > pv[4]):
            return -inf
        else:
            return self.lnprior(pv) + self.lnprior_ext(pv) + self.lnlikelihood(pv)

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

    def sample_mcmc(self, niter=500, thin=5, label='MCMC sampling'):
        if self.sampler is None:
            self.sampler = EnsembleSampler(self.de.n_pop, self.de.n_par, self.lnposterior)
            pop0 = self.de.population
        else:
            pop0 = self.sampler.chain[:,-1,:]
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


class StudentLSqLPF(NormalLSqLPF):
    """Log posterior function with least-squares baseline and Student T likelihood"""

    def __init__(self, target, datasets, filters, nthreads=1, free_k=False, contamination=False, **kwargs):
        super().__init__(target, datasets, filters, nthreads, free_k, contamination, **kwargs)
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
    def __init__(self, target, datasets, filters, nthreads=1, free_k=False, **kwargs):
        super().__init__(target, datasets, filters, nthreads, free_k, **kwargs)

        pbl = [LParameter('bl_{:d}'.format(i), 'baseline', '', N(1, 1e-2), bounds=(-inf,inf)) for i in range(self.npb)]
        self.ps.thaw()
        self.ps.add_lightcurve_block('baseline', 1, self.npb, pbl)
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
                       white_noise=wn, fit_white_noise=False) for wn in self.logwnvar]

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
        for gp, residual in tqdm(zip(self.gps, self.residuals(pv)), total=self.npb, desc='Optimizing GP hyperparameters'):
            residual -= residual.mean()
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
        residuals = [r - r.mean() for r in self.residuals(pv)]

        def nll(hp):
            [gp.set_parameter_vector(hp) for gp in self.gps]
            l = sum([gp.log_likelihood(res, quiet=True) for res,gp in zip(residuals, self.gps)])
            g = sum([gp.grad_log_likelihood(res, quiet=True) for res,gp in zip(residuals, self.gps)])
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
        kernel_sk = LK(0, order=1, bounds=((None, None),), ndim=nd, axes=0)
        kernel_am = LK(0, order=1, bounds=((None, None),), ndim=nd, axes=1)
        kernel_xy = ESK(2, metric_bounds=((0, 20),), ndim=nd, axes=[2, 3])
        kernel_en = LK(0, order=1, bounds=((None, None),), ndim=nd, axes=4)
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
            lnlike += gp.lnlikelihood(residuals)
        return lnlike
