import math as m
import numpy as np
import statsmodels.api as sm

from numpy import (ones, full, sqrt, array, concatenate, diff, ones_like, floor, ceil, all, arange, digitize, zeros,
                   nan, linspace)
from numpy.polynomial.legendre import legvander
from scipy.ndimage import median_filter as mf
from astropy.stats import sigma_clip, mad_std

def find_period(time, flux, minp=1, maxp=10):
    min2day = 1 / 60 / 24
    ls = LombScargle(time, flux - flux.mean())
    freq = linspace(1 / (maxp * min2day), 1 / (minp * min2day), 2500)
    power = ls.power(freq)
    return 1 / freq[argmax(power)], freq, power

class M2LightCurve:
    def __init__(self, time, flux, covariates):
        self._time = array(time)
        self._flux = array(flux)
        self._covariates = concatenate([ones([self._time.size, 1]), array(covariates)], 1)
        self._mask = ones(time.size, np.bool)

        self.btime = self.time
        self.bflux = self.flux
        self.bcovariates = self.covariates
        self.bwn = self.wn

    def downsample_time(self, inttime=30.):
        duration = 24 * 60 * 60 * self.time.ptp()
        nbins = int(ceil(duration / inttime))
        bins = arange(nbins)
        edges = self.time[0] + bins * inttime / 24 / 60 / 60
        bids = digitize(self.time, edges) - 1
        bt, bf, be, bc = full(nbins, nan), zeros(nbins), zeros(nbins), zeros([nbins, self.covariates.shape[1]])
        for i, bid in enumerate(bins):
            bmask = bid == bids
            if bmask.sum() > 0:
                bt[i] = self.time[bmask].mean()
                bf[i] = self.flux[bmask].mean()
                if bmask.sum() > 2:
                    be[i] = self.flux[bmask].std() / sqrt(bmask.sum())
                else:
                    be[i] = self.wn
                bc[i] = self.covariates[bmask].mean(0)
        m = np.isfinite(bt)
        self.btime, self.bflux, self.bwn, self.bcovariates = bt[m], bf[m], be[m], bc[m]


    def downsample_nbins(self, nbins=500):
        pperbin = max(1, int(floor((self.flux.size / nbins))))
        nbins = max(1, int(floor(self.flux.size / pperbin)))
        iend =  pperbin * nbins

        self.btime = self.time[:iend].reshape([-1, pperbin]).mean(1)
        self.bflux = self.flux[:iend].reshape([-1, pperbin]).mean(1)
        self.bwn   = self.flux[:iend].reshape([-1, pperbin]).std(1) / sqrt(pperbin)
        self.bcovariates = self.covariates[:iend].reshape([-1, pperbin, 7]).mean(1)

        if all(self.bwn < 1e-8):
            self.bwn[:] = self.wn


    def detrend_poly(self, npol=20, istart=None, iend=None):
        flux = self.flux[istart:iend]
        covs = self.covariates[istart:iend, [2,4,5,6]].copy()
        covs -= covs.mean(0)
        pol = legvander(linspace(-1, 1, flux.size), npol)
        pol[:, 1:] /= pol[:, 1:].ptp(0)

        covs = concatenate([covs, pol], 1)
        rlm = sm.RLM(flux, covs, hasconst=True)
        res = rlm.fit()

        coefs = res.params.copy()
        res.params[4:] = 0
        systematics = res.predict()
        res.params[:] = coefs
        res.params[:4] = 0
        baseline = res.predict()
        flux_corr = flux - systematics + systematics.mean()
        return self.time[istart:iend], flux_corr, systematics, baseline


    @property
    def time(self):
        return self._time[self._mask]

    @property
    def flux(self):
        return self._flux[self._mask]

    @property
    def covariates(self):
        return self._covariates[self._mask]

    @property
    def wn(self):
        return mad_std(diff(self.flux)) / m.sqrt(2.)

    def mask_covariate_outliers(self, sigma=10, mf_width=15):
        cids = [2, 4, 5]
        mask = ones_like(self._mask)
        for i in cids:
            v = self._covariates[mask, i]
            mv = mf(v, mf_width)
            mmv = sigma_clip(v - mv, sigma, iters=10, stdfunc=mad_std)
            mask[mask] &= ~mmv.mask
        self._mask &= mask

    def mask_outliers(self, sigma=5, mf_width=15, mean=None):
        if mean is None:
            f = self.flux - mf(self.flux, mf_width)
        else:
            f = self.flux - mean
        self._mask[self._mask] = ~sigma_clip(f, sigma).mask


class M2LCSet:
    def __init__(self, lcs):
        self._lcs = tuple(lcs)

    def __getitem__(self, item):
        return self._lcs[item]

    def __iter__(self):
        return self._lcs.__iter__()

    @property
    def times(self):
        return [lc.time for lc in self._lcs]

    @property
    def btimes(self):
        return [lc.btime for lc in self._lcs]

    @property
    def fluxes(self):
        return [lc.flux for lc in self._lcs]

    @property
    def bfluxes(self):
        return [lc.bflux for lc in self._lcs]

    @property
    def covariates(self):
        return [lc.covariates for lc in self._lcs]

    @property
    def bcovariates(self):
        return [lc.bcovariates for lc in self._lcs]

    @property
    def wn(self):
        return [lc.wn for lc in self._lcs]

    @property
    def bwn(self):
        return [lc.bwn for lc in self._lcs]


    def mask_covariate_outliers(self, sigma=10, mf_width=15):
            [lc.mask_covariate_outliers(sigma, mf_width) for lc in self._lcs]

    def mask_outliers(self, sigma=5, mf_width=15, means=None):
        if means is None:
            [lc.mask_outliers(sigma, mf_width) for lc in self._lcs]
        else:
            [lc.mask_outliers(sigma, mean=mean) for lc, mean in zip(self._lcs, means)]

