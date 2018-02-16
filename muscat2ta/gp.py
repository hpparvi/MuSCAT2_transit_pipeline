import numpy as np

from numpy import linspace, argmax, sort, log
from numpy.random import permutation

from scipy.optimize import minimize

from george import GP
from george.kernels import (ConstantKernel as CK, ExpSine2Kernel as ES2K,
                            ExpSquaredKernel as ESK, LinearKernel as LK, ExpKernel as EK)
from astropy.stats.lombscargle import LombScargle

def find_period(time, flux, minp=1, maxp=10):
    min2day = 1 / 60 / 24
    ls = LombScargle(time, flux - flux.mean())
    freq = linspace(1 / (maxp * min2day), 1 / (minp * min2day), 2500)
    power = ls.power(freq)
    return 1 / freq[argmax(power)], freq, power


def create_m2kernel(logp, logv):
    kernel_qp = (CK(logv, bounds=((-18, -10),), ndim=6, axes=0)
                 * ES2K(.001, logp, bounds=((0, 5), (-8, -4)), ndim=6, axes=0)
                 * ES2K(.001, logp, bounds=((0, 2), (-8, -4)), ndim=6, axes=0))
    kernel_sk = CK(logv, bounds=((-18, -10),), ndim=6, axes=1) * ESK(10, metric_bounds=((0, 10),), ndim=6, axes=1)
    kernel_am = LK(0, order=1, bounds=((None, None),), ndim=6, axes=2)
    kernel_xy = CK(logv, bounds=((-18, -10),), ndim=6, axes=3) * ESK(10, metric_bounds=((0, 10),), ndim=6, axes=[3, 4])
    kernel_en = CK(logv, bounds=((-18, -10),), ndim=6, axes=5) * ESK(10, metric_bounds=((0, 10),), ndim=6, axes=5)
    kernel = kernel_qp + kernel_sk + kernel_am + kernel_xy + kernel_en
    return kernel, (kernel_qp, kernel_sk, kernel_am, kernel_xy, kernel_en)

def create_m2kernel(logp, logv):
    kernel_sk = LK(0, order=1, bounds=((None, None),), ndim=5, axes=0)
    kernel_am = LK(0, order=1, bounds=((None, None),), ndim=5, axes=1)
    kernel_xy = CK(logv, bounds=((-18, -10),), ndim=5, axes=2) * ESK(10, metric_bounds=((0, 10),), ndim=5, axes=[2, 3])
    kernel_en = CK(logv, bounds=((-18, -10),), ndim=5, axes=4) * ESK(10, metric_bounds=((0, 10),), ndim=5, axes=4)
    kernel = kernel_sk + kernel_am + kernel_xy + kernel_en
    return kernel, (kernel_sk, kernel_am, kernel_xy, kernel_en)

class M2GP:
    def __init__(self, lpf, transits, pb=0, max_pts=500, kernelf=None, kernelfarg=(), covariates=(2,3,4,5,6)):
        self.lpf = lpf
        self.pb = pb

        npt = lpf.times[pb].size
        self.ids = pid = sort(permutation(npt)[:max_pts])

        self.transits = transits = transits[pb][pid]
        self.flux = flux = lpf.fluxes[pb][pid]
        self.time = lpf.covariates[pb][pid][:, 1]
        self.covariates = lpf.covariates[pb][pid][:, covariates]
        self._all_covariates = lpf.covariates[pb][:, covariates]

        self.residuals = residuals = flux - transits
        self.ls_period = find_period(lpf.times[pb][pid], residuals)[0]
        self.logp = logp = log(self.ls_period)
        self.logv = logv = log(residuals.var())
        self.logwn = log(lpf.wn[pb].mean() ** 2)

        if kernelf is None:
            kernel, self.kernels = create_m2kernel(logp, logv)
        else:
            kernel, self.kernels = kernelf(*kernelfarg)

        self.gp = GP(kernel, mean=residuals.mean(), fit_mean=True,
                     white_noise=self.logwn, fit_white_noise=False)
        self.gp.compute(self.covariates)


    def nll(self, pv):
        return self.gp.nll(pv, self.residuals)

    def grad_nll(self, pv):
        return self.gp.grad_nll(pv, self.residuals)

    def optimize_hps(self):
        self.res = minimize(self.nll, self.gp.get_parameter_vector(), jac=self.grad_nll,
                            bounds=self.gp.get_parameter_bounds())

    def predict(self):
        self.gp.set_parameter_vector(self.res.x)
        self.gp.compute(self.covariates)
        return self.gp.predict(self.residuals, self.covariates, return_cov=False)
