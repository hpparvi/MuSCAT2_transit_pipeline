import warnings
import math as mt
from os.path import join, split

from numpy import (inf, sqrt, ones, hstack, zeros_like, median, floor, concatenate, dot, diff, log, ones_like,
                   percentile, clip, argsort, any, array)
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
    def __init__(self, target, datasets, filters, nthreads=1):
        self.tm = MA(interpolate=False, nthr=nthreads)
        self.target = target
        self.datasets = datasets
        self.filters = filters
        self.nthr = nthreads
        self.npb = npb = len(filters)

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
        k2lims = array([0.1, 1.2]) * diff(percentile(self.fluxes[0], [1, 99]))
        tce = datasets[0].time.ptp() / 10

        psystem = [
            GParameter('tc', 'zero_epoch', 'd', N(t0, tce), (-inf, inf)),
            GParameter('pr', 'period', 'd', N(p, 1e-5), (0, inf)),
            GParameter('k2', 'area_ratio', 'A_s', GM(0.1), (1e-8, inf)),
            GParameter('rho', 'stellar_density', 'g/cm^3', U(0.1, 5.0), (0, inf)),
            GParameter('b', 'impact_parameter', 'R_s', U(0.0, 1.0), (0, 1))]

        pld = concatenate([
            [PParameter('q1_{:d}'.format(i), 'q1_coefficient', '', U(0, 1), bounds=(0, 1)),
             PParameter('q2_{:d}'.format(i), 'q2_coefficient', '', U(0, 1), bounds=(0, 1))]
            for i in range(npb)])

        pbl = [LParameter('bl_{:d}'.format(i), 'baseline', '', N(1, 1e-3), bounds=(-inf,inf)) for i in range(self.npb)]

        self.ps = ps = ParameterSet()
        ps.add_global_block('system', psystem)
        ps.add_passband_block('ldc', 2, npb, pld)
        ps.add_lightcurve_block('baseline', 1, npb, pbl)
        ps.freeze()

        # Define the parameter slices
        # ---------------------------
        self._slld = ps.blocks[1].slices
        self._slbl = ps.blocks[2].slice

        # Setup basic parametrisation
        # ---------------------------
        self.par = BCP()

    def mask_outliers(self, sigma=5, mf_width=10, means=None):
        self.datasets.mask_outliers(sigma=sigma, mf_width=mf_width, means=means)
        self.times = self.datasets.times
        self.fluxes = self.datasets.fluxes
        self.covariates = self.datasets.covariates
        self.wn = [median(wn) for wn in self.datasets.wn]

    def create_pv_population(self, npop=50):
        pvp = self.ps.sample_from_prior(npop)
        pvp[:,2] = uniform(1e-8, 0.2**2, size=npop)
        ldsl = self.ps.blocks[1].slice
        for i in range(pvp.shape[0]):
            pid = argsort(pvp[i, ldsl][::2])[::-1]
            pvp[i, ldsl][::2] = pvp[i, ldsl][::2][pid]
            pvp[i, ldsl][1::2] = pvp[i, ldsl][1::2][pid]
        return pvp

    def baseline(self, pv):
        return pv[self._slbl]

    def transit_model(self, pv):
        _a = as_from_rhop(pv[3], pv[1])
        if _a <= 1.:
            return [ones_like(f) for f in self.times]
        _i = mt.acos(pv[4] / _a)
        _k = mt.sqrt(pv[2])
        fluxes = []
        for t, s in zip(self.times, self._slld):
            q1, q2 = pv[s]
            a, b = sqrt(q1), 2 * q2
            _uv = array([a * b, a * (1. - b)]).T
            fluxes.append(self.tm.evaluate(t, _k, _uv, pv[0], pv[1], _a, _i))
        return fluxes

    def flux_model(self, pv):
        bl = self.baseline(pv)
        tm = self.transit_model(pv)
        return [b * t for b, t in zip(bl, tm)]

    def residuals(self, pv):
        return [fo - fm for fo, fm in zip(self.fluxes, self.flux_model(pv))]

    def lnprior(self, pv):
        return self.ps.lnprior(pv)

    def lnlikelihood(self, pv):
        raise NotImplementedError

    def lnposterior(self, pv):
        if any(pv < self.ps.bounds[:, 0]) or any(pv > self.ps.bounds[:, 1]):
            return -inf
        elif any(diff(sqrt(pv[self.ps.blocks[1].slice][::2])) > 0.):
            return -inf
        else:
            return self.lnprior(pv) + self.lnlikelihood(pv)

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
            pop0 = self.sampler.chains[:,-1,:]
        for _ in tqdm(self.sampler.sample(pop0, iterations=niter, thin=thin), total=niter, desc=label):
            pass


class SimpleLPF(BaseLPF):
    def __init__(self, target, datasets, filters, nthreads=1):
        super().__init__(target, datasets, filters, nthreads)

        pbl = [LParameter('bl_{:d}'.format(i), 'baseline', '', N(1, 1e-3), bounds=(-inf,inf)) for i in range(self.npb)]
        per = [LParameter('er_{:d}'.format(i), 'sigma', '', U(1e-5, 1e-2), bounds=(-inf,inf)) for i in range(self.npb)]
        ps = self.ps
        ps.thaw()
        ps.add_lightcurve_block('baseline', 1, self.npb, pbl)
        ps.add_lightcurve_block('sigma', 1, self.npb, per)
        ps.freeze()

        # Define the parameter slices
        # ---------------------------
        self._slbl = ps.blocks[2].slice
        self._sler = ps.blocks[3].slices

    def baseline(self, pv):
        return pv[self._slbl]

    def lnlikelihood(self, pv):
        flux_m = self.flux_model(pv)
        return sum([ll_normal_es(self.fluxes[i], flux_m[i], pv[self._sler[i]]) for i in range(self.npb)])


class NormalLSqLPF(BaseLPF):
    """Log posterior function with least-squares baseline and normal likelihood"""
    def baseline(self, pv):
        model_fluxes = self.transit_model(pv)
        for i, fm in enumerate(model_fluxes):
            coefs = lstsq(self.covariates[i][:,1:], self.fluxes[i] / fm)[0]
            fbl = dot(self.covariates[i][:,1:], coefs)
            model_fluxes[i] = fbl
        return model_fluxes

    def flux_model(self, pv):
        model_fluxes = self.transit_model(pv)
        for i, fm in enumerate(model_fluxes):
            coefs = lstsq(self.covariates[i][:,1:], self.fluxes[i] / fm)[0]
            fbl = dot(self.covariates[i][:,1:], coefs)
            model_fluxes[i] *= fbl
        return model_fluxes

    def lnlikelihood(self, pv):
        flux_m = self.flux_model(pv)
        lnlike = 0.0
        for i in range(self.npb):
            lnlike += ll_normal_es(self.fluxes[i], flux_m[i], self.wn[i])
        return lnlike


class StudentLSqLPF(NormalLSqLPF):
    """Log posterior function with least-squares baseline and Student T likelihood"""

    def __init__(self, target, datasets, filters, nthreads=1, **kwargs):
        super().__init__(target, datasets, filters, nthreads, **kwargs)
        self.ps.thaw()
        perr = [LParameter('et_{:d}'.format(i), 'error_df', '', U(1e-6, 1), bounds=(1e-6, 1)) for i in range(self.npb)]
        self.ps.add_lightcurve_block('error', 1, self.npb, perr)
        self.ps.freeze()
        self._sler = self.ps.blocks[3].slices

    def lnlikelihood(self, pv):
        flux_m = self.flux_model(pv)
        lnlike = 0.0
        for i, sl in enumerate(self._sler):
            sigma, df = self.wn[i], 1. / pv[sl]
            lnlike += tdist.logpdf(self.fluxes[i] - flux_m[i], df, 0., sigma).sum()
        return lnlike


class GPLPF(BaseLPF):
    def __init__(self, target, datasets, filters, nthreads=1):
        super().__init__(target, datasets, filters, nthreads)
        self.logwnvar = log(array(self.wn) ** 2)
        self._create_kernel()
        self.covariates = [cv[:, self.covids] for cv in self.covariates]
        self.gps = [GP(self._create_kernel(),
                       mean=1., fit_mean=True,
                       white_noise=wn, fit_white_noise=False) for wn in self.logwnvar]
        self.compute_gps()
        self.de = None

    def compute_gps(self, hps=None):
        if hps is not None:
            [gp.set_parameter_vector(hp) for gp, hp in zip(self.gps, hps)]
        [gp.compute(cv) for gp, cv in zip(self.gps, self.covariates)]

    def optimize_hps(self, pv, i=0):
        self.gphpres = []
        self.gphps = []
        for gp, residual in tqdm(zip(self.gps, self.residuals(pv)), total=self.npb,
                                 desc='Optimizing GP hyperparameters'):
            res = minimize(gp.nll, gp.get_parameter_vector(), (residual,),
                           jac=gp.grad_nll, bounds=gp.get_parameter_bounds())
            self.gphpres.append(res)
            self.gphps.append(res.x)

    def _create_kernel(self):
        self.covids = (2, 3, 4, 5, 6)
        logv = self.logwnvar.mean()
        kernel_sk = LK(0, order=1, bounds=((None, None),), ndim=5, axes=0)
        kernel_am = LK(0, order=1, bounds=((None, None),), ndim=5, axes=1)
        kernel_xy = (CK(logv, bounds=((-18, -10),), ndim=5, axes=[2, 3])
                     * ESK(10, metric_bounds=((0, 10),), ndim=5, axes=[2, 3]))
        kernel_en = LK(0, order=1, bounds=((None, None),), ndim=5, axes=4)
        self.kernels = kernel_sk, kernel_am, kernel_xy, kernel_en
        return kernel_sk + kernel_am + kernel_xy + kernel_en

    def predict(self, pv):
        self.compute_gps(self.gphps)
        return [gp.predict(residuals, cv, return_cov=False)
                for gp, residuals, cv in zip(self.gps, self.residuals(pv), self.covariates)]

    def lnlikelihood(self, pv):
        lnlike = 0.0
        for gp, residuals in zip(self.gps, self.residuals(pv)):
            lnlike += gp.lnlikelihood(residuals)
        return lnlike
