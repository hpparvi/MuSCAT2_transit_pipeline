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

import math as m

import numpy as np
import statsmodels.api as sm
from astropy.stats import sigma_clip, mad_std
from numpy import (ones, full, sqrt, array, concatenate, diff, ones_like, floor, ceil, all, arange, digitize, zeros,
                   nan, linspace, isfinite, dot)
from numpy.linalg import lstsq
from numpy.polynomial.legendre import legvander
from scipy.ndimage import median_filter as mf


def find_period(time, flux, minp=1, maxp=10):
    min2day = 1 / 60 / 24
    ls = LombScargle(time, flux - flux.mean())
    freq = linspace(1 / (maxp * min2day), 1 / (minp * min2day), 2500)
    power = ls.power(freq)
    return 1 / freq[argmax(power)], freq, power


def downsample_time(time, vals, inttime=30., trange: tuple = None):
    duration = 24 * 60 * 60 * (time.ptp() if trange is None else trange[1] - trange[0])
    nbins = int(ceil(duration / inttime))
    bins = arange(nbins)
    edges = (time[0] if trange is None else trange[0]) + bins * inttime / 24 / 60 / 60
    bids = digitize(time, edges) - 1
    bt, bv, be = full(nbins, nan), zeros(nbins), zeros(nbins)
    for i, bid in enumerate(bins):
        bmask = bid == bids
        if bmask.sum() > 0:
            bt[i] = time[bmask].mean()
            bv[i] = vals[bmask].mean()
            if bmask.sum() > 2:
                be[i] = vals[bmask].std() / sqrt(bmask.sum())
            else:
                be[i] = nan
    m = isfinite(be)
    return bt[m], bv[m], be[m]


class M2LightCurve:
    def __init__(self, pbid, time, flux, error, covariates, covnames):
        self._time = array(time)
        self._flux = array(flux)
        self._error = array(error)
        self._covariates = concatenate([ones([self._time.size, 1]), array(covariates)], 1)
        self._covnames = ['intercept'] + list(covnames)
        self._mask = ones(time.size, np.bool)
        self.pbid = pbid

        self.btime = self.time
        self.bflux = self.flux
        self.bcovariates = self.covariates
        self.bwn = self.wn

    def downsample_time(self, inttime: float = 30., trange: tuple = None):
        duration = 24 * 60 * 60 * self.time.ptp() if trange is None else trange[1] - trange[0]
        nbins = int(ceil(duration / inttime))
        bins = arange(nbins)
        edges = (self.time[0] if trange is None else trange[0]) + bins * inttime / 24 / 60 / 60
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
        covs = self.covariates[istart:iend, [2, 4, 5, 6]].copy()
        covs -= covs.mean(0)
        covs /= covs.std(0)
        x = (self.time - self.time[0]) / diff(self.time[[0, -1]]) * 2 - 1
        pol = legvander(x, npol)
        pol[:, 1:] /= pol[:, 1:].ptp(0)
        covs = concatenate([covs, pol], 1)

        c, _, _, _ = lstsq(covs, flux, rcond=None)
        cbl = c.copy()
        cbl[:4] = 0.
        css = c.copy()
        css[4:] = 0.
        fbl = dot(covs, cbl)
        fsys = dot(covs, css)
        fall = dot(covs, c)
        return self.time[istart:iend], flux - fall + 1, fsys, fbl

    @property
    def time(self):
        return self._time[self._mask]

    @property
    def flux(self):
        return self._flux[self._mask]

    @property
    def error(self):
        return self._error[self._mask]

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
            mmv = sigma_clip(v - mv, sigma, maxiters=10)
            mask[mask] &= ~mmv.mask
        self._mask &= mask

    def mask_outliers(self, sigma=5, mf_width=15, mean=None):
        if mean is None:
            f = self.flux - mf(self.flux, mf_width)
        else:
            f = self.flux - mean
        self._mask[self._mask] = ~sigma_clip(f, sigma).mask

    def mask_limits(self, limits):
        mask = (self._flux > limits[0]) & (self._flux < limits[1])
        self._mask &= mask


class M2LCSet:
    def __init__(self, lcs):
        self._lcs = tuple(lcs)
        self.size = len(lcs)

    def __getitem__(self, item):
        return self._lcs[item]

    def __iter__(self):
        return self._lcs.__iter__()

    @property
    def pbids(self):
        return [lc.pbid for lc in self._lcs]

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

    def mask_limits(self, limits):
        [lc.mask_limits(limits) for lc in self._lcs]