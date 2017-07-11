import warnings
import math as mt

from numpy import inf, sqrt, ones, hstack, zeros_like, median, floor, concatenate
from scipy.stats import norm as N, uniform as U

from astropy.stats import sigma_clip

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
    exocat = exodata.OECDatabase('../ext/oec/systems/')

from scipy.stats import t as tdist

from parameter import *

class LPFunction:
    def __init__(self, target, datasets, filters, nthreads=1):
        self.tm = MA(interpolate=False, klims=(sqrt(0.007), sqrt(0.02)), nthr=nthreads)
        self.target = target
        self.datasets = datasets
        self.filters = filters
        self.nthr = nthreads
        self.npb = len(filters)

        # Extract the data
        # ----------------
        # Extract the data arrays from the dataframes and remove
        # any outrageously clear outliers using sigma clipping
        self._setup_arrays()
        self.masks = outlier_masks = self.sigma_clip(sigma=3)
        self._setup_arrays(outlier_masks)
        self.npb = npb = len(self.fluxes)

        # Read the planet parameters from the OEC
        # ---------------------------------------
        self.planet = p  = exocat.searchPlanet(target)
        self.star   = s  = p.star
        self.t0_bjd = t0 = float(p.transittime)
        self.period = pr = float(p.P)
        self.sma         = float(p.a.rescale('R_s') / s.R)
        self.inc         = float(p.i.rescale('rad'))
        self.k           = float(p.R.rescale('R_s') / s.R)

        # Apply the time offset
        # ---------------------
        self.time_offset = floor(self.times[0].min())
        self.t0 = t0     = self.t0_bjd - self.time_offset
        self.times       = [t - self.time_offset for t in self.times]

        self.compute_ot_masks()

        # Define the parameters and their priors
        # --------------------------------------
        k2lims = array([0.8,1.2])*self.k**2

        psystem = [
            GParameter('tc',  'zero_epoch',       'd',      N( t0, 0.020), (-inf, inf)),
            GParameter('pr',  'period',           'd',      N( pr, 0.001), (   0, inf)),
            GParameter('k2',  'area_ratio',       'A_s',    U(   *k2lims), (   0, inf)),
            GParameter('rho', 'stellar_density',  'g/cm^3', U(1.0,   3.0), (   0, inf)),
            GParameter('b',   'impact_parameter', 'R_s',    U(0.0,   1.0), (   0,   1))]

        pld = concatenate([
            [PParameter('q1_{:d}'.format(i), 'q1_coefficient', '', U(0,1), bounds=(0,1)),
             PParameter('q2_{:d}'.format(i), 'q2_coefficient', '', U(0,1), bounds=(0,1))]
            for i in range(npb)])

        pbl = [
            LParameter('bl_{:d}'.format(i), 'baseline', '', N(1, 1e-3), bounds=(-inf,inf))
            for i in range(npb)]

        per = [
            LParameter('er_{:d}'.format(i), 'sigma', '', U(1e-5, 1e-2), bounds=(-inf,inf))
            for i in range(npb)]

        self.ps = ps = ParameterSet()
        ps.add_global_block('system', psystem)
        ps.add_passband_block('ldc', 2, npb, pld)
        ps.add_lightcurve_block('baseline', 1, npb, pbl)
        ps.add_lightcurve_block('sigma', 1, npb, per)
        ps.freeze()

        # Define the parameter slices
        # ---------------------------
        self._slld = ps.blocks[1].slices
        self._slbl = ps.blocks[2].slice
        self._sler = ps.blocks[3].slices

        # Setup basic parametrisation
        # ---------------------------
        self.par = BCP()


    def _setup_arrays(self, masks=None):
        self.times, self.fluxes, self.ferrs, self.auxs, self.npt = self._extract_arrays(self.datasets, masks)

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
            auxs[-1][:, 1] -= auxs[-1][:, 1].mean()
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
        pvp = self.ps.generate_pv_population(npop)
        for i, p in enumerate(concatenate(self.compute_baseline_coeffs())):
            pvp[:, self._sbl + i] = normal(p, 1e-3, pvp.shape[0])
        return pvp

    def compute_ot_masks(self):
        self.masks_ot = []
        for time in self.times:
            phase = (fold(time, self.period, self.t0, shift=0.5) - 0.5) * self.period
            tdur = of.duration_eccentric_f(self.period, self.k, self.sma, self.inc, 0., 0., 1)
            self.masks_ot.append(abs(phase) > 0.6 * tdur)

    def compute_baseline(self, pv):
        return pv[self._slbl]

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
        flux_m = self.compute_lc_model(pv)
        return sum([ll_normal_es(self.fluxes[i], flux_m[i], pv[self._sler[i]]) for i in range(self.npb)])

    def lnposterior(self, pv):
        if any(pv < self.ps.bounds[:,0]) or any(pv > self.ps.bounds[:,1]):
            return -inf
        else:
            return self.lnprior(pv) + self.lnlikelihood(pv)


    def __call__(self, pv):
        return self.lnposterior(pv)

