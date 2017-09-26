import warnings
import math as mt
from os.path import join, split

from numpy import inf, sqrt, ones, hstack, zeros_like, median, floor, concatenate, dot, diff, log
from numpy.linalg import lstsq
from numpy.random import uniform, normal
from scipy.stats import norm as N, uniform as U, scoreatpercentile as sap
from scipy.optimize import minimize

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

import exodata

warnings.filterwarnings("ignore", category=UserWarning)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    exocat = exodata.OECDatabase(join(split(__file__)[0], '../ext/oec/systems/'))

from scipy.stats import t as tdist

from .parameter import *

class BaseLPF:
    def __init__(self, target, datasets, filters, nthreads=1, tce=2e-3, pre=1e-5, shift_tc=True):
        self.tm = MA(interpolate=False, klims=(sqrt(0.007), sqrt(0.02)), nthr=nthreads)
        self.target = target
        self.datasets = datasets
        self.filters = filters
        self.nthr = nthreads
        self.npb = len(filters)

        # Read the planet parameters from the OEC
        # ---------------------------------------
        self.planet = p  = exocat.searchPlanet(target)
        self.star   = s  = p.star
        self.t0_bjd = t0 = float(p.transittime)
        self.period = pr = float(p.P)
        self.sma         = float(p.a.rescale('R_s') / s.R)
        self.inc         = float(p.i.rescale('rad'))
        self.k           = float(p.R.rescale('R_s') / s.R)

        # Extract the data
        # ----------------
        # Extract the data arrays from the dataframes and remove
        # any outrageously clear outliers using sigma clipping
        self._setup_arrays()
        self.masks = outlier_masks = self.sigma_clip(sigma=3)
        self._setup_arrays(outlier_masks)
        self.npb = npb = len(self.fluxes)
        self.wn = [mad_std(diff(f)) / sqrt(2) for f in self.fluxes]

        t0 = self.t0
        if shift_tc:
            self.t0 = t0 = self.t0 + self.period * self.tno

        self.compute_ot_masks()

        # Define the parameters and their priors
        # --------------------------------------
        k2lims = array([0.8,1.2])*self.k**2

        psystem = [
            GParameter('tc',  'zero_epoch',       'd',      N( t0,   tce), (-inf, inf)),
            GParameter('pr',  'period',           'd',      N( pr,   pre), (   0, inf)),
            GParameter('k2',  'area_ratio',       'A_s',    U(   *k2lims), (   0, inf)),
            GParameter('rho', 'stellar_density',  'g/cm^3', U(1.0,   3.0), (   0, inf)),
            GParameter('b',   'impact_parameter', 'R_s',    U(0.0,   1.0), (   0,   1))]

        pld = concatenate([
            [PParameter('q1_{:d}'.format(i), 'q1_coefficient', '', U(0,1), bounds=(0,1)),
             PParameter('q2_{:d}'.format(i), 'q2_coefficient', '', U(0,1), bounds=(0,1))]
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


    def _setup_arrays(self, masks=None):
        self.times_bjd, self.fluxes, self.ferrs, self.auxs, self.npt = self._extract_arrays(self.datasets, masks)

        # Apply the time offset
        # ---------------------
        self.time_offset = floor(self.times_bjd[0].min())
        self.t0 = self.t0_bjd - self.time_offset
        self.times       = [t - self.time_offset for t in self.times_bjd]
        self.tno = int(round((median(self.times[0]) - self.t0)/self.period))


    def _extract_arrays(self, datasets, masks=None):
        times, fluxes, ferrs, auxs, npts = [], [], [], [], []
        for i, f in enumerate(self.filters):
            ds = datasets[f]
            mask = ones(ds.bjd.size, bool) if masks is None else masks[i]
            npt = int(mask.sum())
            times.append(ds.bjd.values[mask])
            fluxes.append(ds.flux.values[mask])
            ferrs.append(ds.ferr.values[mask])
            auxs.append(hstack([ones((npt, 1)), ds['bjd airmass dx dy fwhm'.split()].values[mask]]))
            auxs[-1][:, 1:] -= auxs[-1][:, 1:].mean(0)
            npts.append(npt)
        return times, fluxes, ferrs, auxs, npt


    def sigma_clip(self, pv=None, mf_width=10, sigma=6):
        models = None if pv is None else self.compute_lc_model(pv)
        masks = []
        for i, flux in enumerate(self.fluxes):
            if models is None:
                d = zeros_like(flux)
                d[1:] = flux[1:] - flux[:-1]
            else:
                d = flux - models[i]
            masks.append(~sigma_clip(d, sigma=sigma, iters=10).mask)
        return masks

    def create_pv_population(self, npop=50):
        pvp = self.ps.sample_from_prior(npop)
        k2est = (1 - sap(concatenate(self.fluxes), 10))
        pvp[:,2] = uniform(0.95*k2est, 1.05*k2est, pvp.shape[0])
        #for i, p in enumerate(concatenate(self.compute_baseline_coeffs())):
        #    pvp[:, self._sbl + i] = normal(p, 1e-3, pvp.shape[0])
        return pvp

    def compute_ot_masks(self):
        self.masks_ot = []
        for time in self.times:
            phase = (fold(time, self.period, self.t0, shift=0.5) - 0.5) * self.period
            tdur = of.duration_eccentric_f(self.period, self.k, self.sma, self.inc, 0., 0., 1)
            self.masks_ot.append(abs(phase) > 0.6 * tdur)

    def compute_baseline(self, pv):
        raise NotImplementedError

    def compute_transit(self, pv):
        _a = as_from_rhop(pv[3], pv[1])
        _i = mt.acos(pv[4] / _a)
        _k = mt.sqrt(pv[2])
        fluxes = []
        for t,s in zip(self.times, self._slld):
            q1, q2 = pv[s]
            a, b = sqrt(q1), 2*q2
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
        if any(pv < self.ps.bounds[:,0]) or any(pv > self.ps.bounds[:,1]):
            return -inf
        else:
            return self.lnprior(pv) + self.lnlikelihood(pv)

    def __call__(self, pv):
        return self.lnposterior(pv)



class SimpleLPF(BaseLPF):
    def __init__(self, target, datasets, filters, nthreads=1):
        super().__init__(target, datasets, filters, nthreads)

        pbl = [LParameter('bl_{:d}'.format(i), 'baseline', '', N(1, 1e-3), bounds=(-inf,inf)) for i in range(npb)]
        per = [LParameter('er_{:d}'.format(i), 'sigma', '', U(1e-5, 1e-2), bounds=(-inf,inf)) for i in range(npb)]
        ps = self.ps
        ps.thaw()
        ps.add_lightcurve_block('baseline', 1, npb, pbl)
        ps.add_lightcurve_block('sigma', 1, npb, per)
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


class LSqLPF(BaseLPF):
    def compute_baseline(self, pv):
        model_fluxes = self.compute_transit(pv)
        for i, fm in enumerate(model_fluxes):
            coefs = lstsq(self.auxs[i], self.fluxes[i] / fm)[0]
            fbl = dot(self.auxs[i], coefs)
            model_fluxes[i] = fbl
        return model_fluxes

    def compute_lc_model(self, pv):
        model_fluxes = self.compute_transit(pv)
        for i, fm in enumerate(model_fluxes):
            coefs = lstsq(self.auxs[i], self.fluxes[i] / fm)[0]
            fbl = dot(self.auxs[i], coefs)
            model_fluxes[i] *= fbl
        return model_fluxes


class NormalLSqLPF(LSqLPF):
    """Log posterior function with least-squares baseline and normal likelihood"""
    def lnlikelihood(self, pv):
        flux_m = self.compute_lc_model(pv)
        lnlike = 0.0
        for i in range(self.npb):
            lnlike += ll_normal_es(self.fluxes[i], flux_m[i], self.wn[i])
        return lnlike


class StudentLSqLPF(LSqLPF):
    """Log posterior function with least-squares baseline and Student T likelihood"""

    def __init__(self, target, datasets, filters, nthreads=1, **kwargs):
        super().__init__(target, datasets, filters, nthreads, **kwargs)
        ps = self.ps
        ps.thaw()
        perr = [LParameter('et_{:d}'.format(i), 'error_df', '', U(1e-6, 1), bounds=(1e-6, 1)) for i in range(self.npb)]
        ps.add_lightcurve_block('error', 1, self.npb, perr)
        ps.freeze()
        self.ps = ps
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
            CK(log(1e-6), (log(1e-14), log(1e-4)), ndim=4, axes=[0]) * EK(1.0, (0.01, 1e5), ndim=4, axes=[0]) +
            CK(log(1e-6), (log(1e-14), log(1e-4)), ndim=4, axes=[1]) * ESK(1.0, (0.50, 1e5), ndim=4, axes=[1]) +
            CK(log(1e-6), (log(1e-14), log(1e-4)), ndim=4, axes=[2]) * ESK(1.0, (0.50, 1e5), ndim=4, axes=[2]) +
            CK(log(1e-6), (log(1e-14), log(1e-4)), ndim=4, axes=[3]) * ESK(1.0, (0.50, 1e5), ndim=4, axes=[3])
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
            hps.append(minimize(lambda pv: gp.nll(pv, r), gp.get_vector(),
                                jac=lambda pv: gp.grad_nll(pv, r), bounds=gp.get_bounds()))
            gp.set_vector(self.hps[-1].x)
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
