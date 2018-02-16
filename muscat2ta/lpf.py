import warnings
import math as mt
from os.path import join, split

from numpy import (inf, sqrt, ones, hstack, zeros_like, median, floor, concatenate, dot, diff, log, ones_like,
                   percentile, clip, argsort, any)
from numpy.linalg import lstsq
from numpy.random import uniform, normal
from scipy.stats import norm as N, uniform as U, scoreatpercentile as sap
from scipy.optimize import minimize
from tqdm import tqdm
from astropy.stats import sigma_clip, mad_std

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

from parameter import *


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
        self.wn = self.datasets.bwn

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
            GParameter('k2', 'area_ratio', 'A_s', U(*k2lims), (0, inf)),
            GParameter('rho', 'stellar_density', 'g/cm^3', U(0.1, 5.0), (0, inf)),
            GParameter('b', 'impact_parameter', 'R_s', U(0.0, 1.0), (0, 1))]

        pld = concatenate([
            [PParameter('q1_{:d}'.format(i), 'q1_coefficient', '', U(0, 1), bounds=(0, 1)),
             PParameter('q2_{:d}'.format(i), 'q2_coefficient', '', U(0, 1), bounds=(0, 1))]
            for i in range(npb)])

        self.ps = ps = ParameterSet()
        ps.add_global_block('system', psystem)
        ps.add_passband_block('ldc', 2, npb, pld)
        ps.freeze()

        # Define the parameter slices
        # ---------------------------
        self._slld = ps.blocks[1].slices

        # Setup basic parametrisation
        # ---------------------------
        self.par = BCP()

    def mask_outliers(self, sigma=5, mf_width=10, means=None):
        self.datasets.mask_outliers(sigma=sigma, mf_width=mf_width, means=means)
        self.times = self.datasets.times
        self.fluxes = self.datasets.fluxes
        self.covariates = self.datasets.covariates
        self.wn = self.datasets.wn

    def create_pv_population(self, npop=50):
        pvp = self.ps.sample_from_prior(npop)
        ldsl = self.ps.blocks[1].slice
        for i in range(pvp.shape[0]):
            pid = argsort(pvp[i, ldsl][::2])[::-1]
            pvp[i, ldsl][::2] = pvp[i, ldsl][::2][pid]
            pvp[i, ldsl][1::2] = pvp[i, ldsl][1::2][pid]
        return pvp


    def compute_baseline(self, pv):
        raise NotImplementedError

    def compute_transit(self, pv):
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

    def compute_lc_model(self, pv):
        bl = self.compute_baseline(pv)
        tm = self.compute_transit(pv)
        return [b * t for b, t in zip(bl, tm)]

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

    def optimize_global(self, niter=200, npop=50):
        self.de = DiffEvol(self.lnposterior, clip(self.ps.bounds, -1, 1), npop, maximize=True)
        self.de._population[:, :] = self.create_pv_population(npop)
        for _ in tqdm(self.de(niter), total=niter, desc='Differential evolution'):
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

    def compute_baseline(self, pv):
        return pv[self._slbl]

    def lnlikelihood(self, pv):
        flux_m = self.compute_lc_model(pv)
        return sum([ll_normal_es(self.fluxes[i], flux_m[i], pv[self._sler[i]]) for i in range(self.npb)])


class NormalLSqLPF(BaseLPF):
    """Log posterior function with least-squares baseline and normal likelihood"""
    def compute_baseline(self, pv):
        model_fluxes = self.compute_transit(pv)
        for i, fm in enumerate(model_fluxes):
            coefs = lstsq(self.covariates[i], self.fluxes[i] / fm)[0]
            fbl = dot(self.covariates[i], coefs)
            model_fluxes[i] = fbl
        return model_fluxes

    def compute_lc_model(self, pv):
        model_fluxes = self.compute_transit(pv)
        for i, fm in enumerate(model_fluxes):
            coefs = lstsq(self.covariates[i], self.fluxes[i] / fm)[0]
            fbl = dot(self.covariates[i], coefs)
            model_fluxes[i] *= fbl
        return model_fluxes

    def lnlikelihood(self, pv):
        flux_m = self.compute_lc_model(pv)
        lnlike = 0.0
        for i in range(self.npb):
            lnlike += ll_normal_es(self.fluxes[i], flux_m[i], self.wn[i])
        return lnlike


class StudentLSqLPF(NormalLSqLPF):
    """Log posterior function with least-squares baseline and Student T likelihood"""

    def __init__(self, target, datasets, filters, nthreads=1, **kwargs):
        super().__init__(target, datasets, filters, nthreads, **kwargs)
        #ps = self.ps
        self.ps.thaw()
        perr = [LParameter('et_{:d}'.format(i), 'error_df', '', U(1e-6, 1), bounds=(1e-6, 1)) for i in range(self.npb)]
        self.ps.add_lightcurve_block('error', 1, self.npb, perr)
        self.ps.freeze()
        #self.ps = ps
        self._sler = self.ps.blocks[2].slices

    def lnlikelihood(self, pv):
        flux_m = self.compute_lc_model(pv)
        lnlike = 0.0
        for i, sl in enumerate(self._sler):
            sigma, df = self.wn[i], 1. / pv[sl]
            lnlike += tdist.logpdf(self.fluxes[i] - flux_m[i], df, 0., sigma).sum()
        return lnlike


class LinearLPF(BaseLPF):
    def __init__(self, target, datasets, filters, nthreads=1, **kwargs):
        super().__init__(target, datasets, filters, nthreads, **kwargs)

        pbl = concatenate([(LParameter('bc_{:d}'.format(i), 'baseline_constant', '', N(1, 1e-3), bounds=(-inf, inf)),
                            LParameter('bt_{:d}'.format(i), 'l_time_coefficient', '', N(0, 1e-1), bounds=(-inf, inf)),
                            LParameter('bx_{:d}'.format(i), 'l_airmass_coefficient', '', N(0, 1e-1),
                                       bounds=(-inf, inf)))
                           for i in range(self.npb)])
        ps = self.ps
        ps.thaw()
        ps.add_lightcurve_block('baseline', 3, self.npb, pbl)
        ps.freeze()

        # Define the parameter slices
        # ---------------------------
        self._slbl = ps.blocks[2].slice

    def deterministic_baseline(self, pv):
        fm = []
        for i, (bc, bt, bx) in enumerate(pv[self._slbl].reshape([3, -1])):
            fm.append(bc + bt * self.auxs[i][:, 1] + bx * self.auxs[i][:, 2])
        return fm

    def compute_baseline(self, pv):
        model_fluxes = self.compute_transit(pv)
        base_fluxes = self.deterministic_baseline(pv)
        for i, (fm, fbd) in enumerate(zip(model_fluxes, base_fluxes)):
            coefs = lstsq(self.auxs[i][:, [0, 3, 4, 5]], self.fluxes[i] - (fm * fbd))[0]
            fbl = dot(self.auxs[i][:, [0, 3, 4, 5]], coefs)
            model_fluxes[i] = fbl + fbd
        return model_fluxes

    def compute_lc_model(self, pv):
        model_fluxes = self.compute_transit(pv)
        base_fluxes = self.deterministic_baseline(pv)
        for i, (fm, fbd) in enumerate(zip(model_fluxes, base_fluxes)):
            coefs = lstsq(self.auxs[i][:, [0, 3, 4, 5]], self.fluxes[i] - (fm * fbd))[0]
            fbl = dot(self.auxs[i][:, [0, 3, 4, 5]], coefs)
            model_fluxes[i] = fm * fbd + fbl
        return model_fluxes

    def lnlikelihood(self, pv):
        flux_m = self.compute_lc_model(pv)
        lnlike = 0.0
        for i in range(self.npb):
            lnlike += ll_normal_es(self.fluxes[i], flux_m[i], self.wn[i])
        return lnlike


class GPLPF(LinearLPF):
    def __init__(self, target, datasets, filters, nthreads=1, **kwargs):
        self.gps = None
        super().__init__(target, datasets, filters, nthreads, **kwargs)

        self.kernels = [
            CK(log(1e-6), log([[1e-14, 1e-4]]), ndim=4, axes=[0]) * EK( 1.0, log([[0.01, 1e5]]), ndim=4, axes=[0]) +
            CK(log(1e-6), log([[1e-14, 1e-4]]), ndim=4, axes=[1]) * ESK(1.0, log([[0.50, 1e5]]), ndim=4, axes=[1]) +
            CK(log(1e-6), log([[1e-14, 1e-4]]), ndim=4, axes=[2]) * ESK(1.0, log([[0.50, 1e5]]), ndim=4, axes=[2]) +
            CK(log(1e-6), log([[1e-14, 1e-4]]), ndim=4, axes=[3]) * ESK(1.0, log([[0.50, 1e5]]), ndim=4, axes=[3])
            for i in range(self.npb)]
        self.gps = [GP(self.kernels[i]) for i in range(len(filters))]

    def _setup_arrays(self, masks=None):
        super()._setup_arrays(masks)
        if self.gps:
            [gp.compute(aux[:, [1, 3, 4, 5]], wn) for gp, aux, wn in zip(self.gps, self.auxs, self.wn)]

    def optimize_hps(self, pv):
        fms = self.compute_transit(pv)
        res = [fo - fm for fo, fm in zip(self.fluxes, fms)]
        self.hps = hps = []
        self.lsq = []
        for i, gp in enumerate(self.gps):
            b = lstsq(self.auxs[i], 1 + res[i])[0]
            b[3:] = 0
            self.lsq.append(b[:3])
            r = res[i] - dot(self.auxs[i], b)
            r -= r.mean()
            hps.append(minimize(lambda pv: gp.nll(pv, r), gp.get_parameter_vector(),
                                jac=lambda pv: gp.grad_nll(pv, r), bounds=gp.get_parameter_bounds()))
            gp.set_parameter_vector(self.hps[-1].x)
            gp.compute(self.auxs[i][:, [1, 3, 4, 5]], self.wn[i])

    def compute_baseline(self, pv):
        return self.deterministic_baseline(pv)

    def compute_lc_model(self, pv):
        bl = self.compute_baseline(pv)
        tm = self.compute_transit(pv)
        return [b * t for b, t in zip(bl, tm)]

    def lnlikelihood(self, pv):
        flux_m = self.compute_lc_model(pv)
        lnlike = 0.0
        for i, (fo, fm, gp) in enumerate(zip(self.fluxes, flux_m, self.gps)):
            lnlike += gp.lnlikelihood(fo - fm)
        return lnlike

    def create_pv_population(self, npop=50):
        pvp = self.ps.sample_from_prior(npop)
        k2est = (1 - sap(concatenate(self.fluxes), 10))
        pvp[:, 2] = uniform(0.95 * k2est, 1.05 * k2est, pvp.shape[0])
        return pvp
