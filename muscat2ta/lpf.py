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
from os.path import join, split

from george import GP
from george.kernels import ExpSquaredKernel as ESK, ConstantKernel as CK
from muscat2ta.lc import M2LCSet
from numba import njit
from numpy import (zeros_like, median, dot, sum, inf, zeros, atleast_2d, sqrt, log, pi)
from numpy.linalg import lstsq, LinAlgError
from pytransit import QuadraticModel
from pytransit.contamination import SMContamination
from pytransit.contamination.filter import sdss_g, sdss_r, sdss_i, sdss_z
from pytransit.contamination.instrument import Instrument
from pytransit.lpf.lpf import BaseLPF
from pytransit.param.parameter import LParameter, UniformPrior as U, NormalPrior as N
from scipy.optimize import minimize
from scipy.stats import t as tdist
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

@njit(cache=False)
def lnlike_normal(o, m, e):
    return -sum(log(e)) -0.5*p.size*log(2.*pi) - 0.5*sum((o-m)**2/e**2)

@njit("f8(f8[:], f8[:], f8)", cache=False)
def lnlike_normal_s(o, m, e):
    return -o.size*log(e) -0.5*o.size*log(2.*pi) - 0.5*sum((o-m)**2)/e**2


class M2BaseLPF(BaseLPF):

    models = "pb_independent_k pb_dependent_k pb_dependent_contamination physical_contamination".split()

    def __init__(self, target: str, datasets: M2LCSet, filters: tuple, model='pb_independent_k',
                 use_oec: bool = False, period: float = 5.):
        assert (model in self.models), 'Model must be one of:\n\t' + ', '.join(self.models)
        self.model = model
        self.datasets = datasets
        self.use_oec = use_oec
        self.planet = None

        if datasets is not None:
            super().__init__(target, filters, datasets.btimes, datasets.bfluxes,
                             datasets.bwn, datasets.pbids, datasets.bcovariates,
                             tm = QuadraticModel(interpolate=True, klims=(0.01, 0.75), nk=512, nz=512))

        # Read the planet parameters from the OEC
        # ---------------------------------------
        if self.use_oec:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                import exodata
                exocat = exodata.OECDatabase(join(split(__file__)[0], '../ext/oec/systems/'))
            self.planet = exocat.searchPlanet(target)

        p = self.planet.P if self.planet else period
        t0 = datasets[0].time.mean() if datasets is not None else 0.0
        tce = datasets[0].time.ptp() / 10 if datasets is not None else 1e-5
        self.set_prior(0, N(t0, tce))
        self.set_prior(1, N( p, 1e-5))

    def _init_instrument(self):
        self.instrument = Instrument('MuSCAT2', [sdss_g, sdss_r, sdss_i, sdss_z])
        self.cm = SMContamination(self.instrument, "i'")

    def mask_outliers(self, sigma=5, mf_width=10, means=None):
        self.datasets.mask_outliers(sigma=sigma, mf_width=mf_width, means=means)
        self.times = self.datasets.times
        self.fluxes = self.datasets.fluxes
        self.covariates = self.datasets.covariates
        self.wn = [median(wn) for wn in self.datasets.wn]


class NormalLSqLPF(M2BaseLPF):
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

    def __init__(self, target, datasets, filters, model='pb_independent_k', use_oec: bool = False, period: float = 5.):
        super().__init__(target, datasets, filters, model, use_oec=use_oec, period=period)
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
