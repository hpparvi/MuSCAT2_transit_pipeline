#  MuSCAT2 photometry and transit analysis pipeline
#  Copyright (C) 2020  Hannu Parviainen
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
from matplotlib.pyplot import subplots
from numba import njit
from numpy import ndarray, exp, inf, linspace, fabs, argmin, isfinite, newaxis, arange, mean, median, nan, ceil, any, \
    all, array, diff, concatenate, full, asarray, zeros, log, pi, sum, all, any
from photutils import CircularAperture
from scipy.optimize import minimize
from scipy.stats import norm

@njit(cache=False)
def lnlike_normal_s(o, m, e):
    return -o.size*log(e) -0.5*o.size*log(2.*pi) - 0.5*sum((o-m)**2)/e**2

@njit
def psf_model(x, center, separation, a1, a2, w1, w2, sky, sky_slope):
    y = sky + (x - x.mean()) * sky_slope
    y += a1 * exp(-(x - center + 0.5 * separation) ** 2 / (2 * w1 ** 2))
    y += a2 * exp(-(x - center - 0.5 * separation) ** 2 / (2 * w2 ** 2))
    return y


@njit
def model(pv, xs, nstars):
    i = 9
    cs = pv[0]  # component separation
    cax = pv[1:3]  # component amplitudes in x
    cay = pv[3:5]  # component amplitudes in y
    cwx = pv[5:7]  # component widhts in x
    cwy = pv[7:9]  # component widhts in x
    xc = pv[i:i + nstars]  # x centers for each star
    yc = pv[i + nstars:i + 2 * nstars]  # y centers for each star
    slx = pv[i + 2 * nstars:i + 3 * nstars]  # sky levels for each star in x
    sly = pv[i + 3 * nstars:i + 4 * nstars]  # sky levels for each star in y
    ssx = pv[i + 4 * nstars:i + 5 * nstars]  # sky slopes for each star in x
    ssy = pv[i + 5 * nstars:i + 6 * nstars]  # sky slopes for each star in y
    results = zeros((2 * nstars, xs.size))
    for i in range(nstars):
        results[i, :] = psf_model(xs, xc[i], cs, cax[0], cax[1], cwx[0], cwx[1], slx[i], ssx[i])
    for i in range(nstars):
        results[nstars + i, :] = psf_model(xs, yc[i], cs, cay[0], cay[1], cwy[0], cwy[1], sly[i], ssy[i])
    return results


def log_likelihood(pv, obs, nstars):
    pv = asarray(pv)
    if any(pv[:9] <= 0.0) or any(pv[-nstars:] <= 0.0):
        return inf
    if not all(0 < pv[9:9 + 2 * nstars]) and all(pv[9:9 + 2 * nstars] < obs.shape[1]):
        return inf

    mod = model(pv, arange(obs.shape[1]), nstars)
    lnl = 0.
    for i in range(nstars):
        lnl += lnlike_normal_s(obs[i], mod[i], pv[i - nstars])
        lnl += lnlike_normal_s(obs[i + nstars], mod[i + nstars], pv[i - nstars])
    return lnl


def log_posterior(pv, obs, nstars):
    return log_likelihood(pv, obs, nstars) + norm.logpdf(pv[-5 * nstars:-nstars], 0.0, 1.0).sum()


@njit
def final_centroid(rx, start, end):
    x = linspace(start, end, 300)
    m = model(x, rx[0], rx[1], rx[2], rx[3], rx[4], rx[5], rx[6], rx[7]) - rx[-2] - rx[-1] * x

    m = fabs(m / m.max() - 0.05)
    mask = x < rx[0]
    if any(mask):
        xleft = x[mask][argmin(m[mask])]
    else:
        xleft = 0

    if all(mask):
        xright = end
    else:
        xright = x[~mask][argmin(m[~mask])]
    xc = 0.5 * (xleft + xright)
    return xc, xright - xc


class Centroider:
    def __init__(self) -> None:
        self.image = None
        self.r = None

    def calculate_centroid(self, image: ndarray, centers: ndarray, aperture_radius: float = 20):
        raise NotImplementedError

    def plot_profiles(self):
        raise NotImplementedError

    def plot_images(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class DFCentroider(Centroider):
    def __init__(self):
        super().__init__()
        self._x0 = None

    def model(self, pv):
        return model(pv, arange(self.npt), self.nstars)

    def calculate_centroid(self, image: ndarray, centers: ndarray, aperture_radius: float = 20., x0: ndarray = None,
                           update: bool = True):
        self.image = image
        self.r = aperture_radius
        self.nstars = ns = len(centers)
        self.apertures = CircularAperture(centers, self.r)
        self.masks = self.apertures.to_mask()

        xarray, yarray = [], []
        for m in self.masks:
            c = m.cutout(image)
            x, y = c.mean(0), c.mean(1)
            xarray.append((x - x.min()) / x.ptp())
            yarray.append((y - y.min()) / y.ptp())
        self.profiles_obs = concatenate([xarray, yarray])
        self.npt = npt = self.profiles_obs.shape[1]

        if x0 is None:
            if update and self._x0 is not None:
                x0 = self._x0
            else:
                x0 = concatenate([[15, 1, 1, 1, 1, 5, 5, 5, 5],
                                  full(ns, npt / 2), full(ns, npt / 2),
                                  zeros(4 * ns),
                                  full(ns, 0.05)])
                self._x0 = x0

        self._res = minimize(lambda x: -log_likelihood(x, self.profiles_obs, self.nstars), x0, method='nelder-mead')
        self._x0 = self._res.x.copy()
        self.profiles_mod = model(self._res.x, arange(npt), self.nstars)

        self.centers, self.radius = self._centroid_from_model()

        for i, bb in enumerate(self.apertures.bbox):
            self.centers[i] += array([bb.ixmin, bb.iymin])
        return self.centers

    def _centroid_from_model(self, pv=None, res: int = 1000):
        pvo = pv if pv is not None else self._res.x
        pv = pvo.copy()
        pv[-5 * self.nstars:] = 0  # Set the baseline to zero
        pv[9:9 + 2 * self.nstars] = 0  # Set the centers to zero

        xs = linspace(-2 * self.r, 2 * self.r, res)
        models = model(pv, xs, self.nstars)
        models /= models.max(1)[:, newaxis]

        mx = models[0] >= 0.05
        my = models[self.nstars] >= 0.05

        xlimits = xs[mx][[0, -1]]
        ylimits = xs[my][[0, -1]]

        x_offset = xlimits.mean()
        y_offset = ylimits.mean()

        x_radius = 0.5 * diff(xlimits)[0]
        y_radius = 0.5 * diff(ylimits)[0]
        radius = max(x_radius, y_radius)

        centers = array([pvo[9:9 + self.nstars] + x_offset,
                         pvo[9 + self.nstars:9 + 2 * self.nstars] + y_offset]).T
        return centers, radius

    def plot_images(self, max_cols: int = 3):
        ncols = min(max_cols, self.nstars)
        nrows = int(ceil(self.nstars / ncols))

        aps = CircularAperture(self.centers, self.r)
        masks = aps.to_mask()

        fig, axs = subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

        for i in range(self.nstars):
            ax = axs.flat[i]
            dd = masks[i].cutout(self.image, fill_value=nan)
            ax.imshow(dd, extent=aps[i].bbox.extent)
            ax.plot(*self.centers[i], marker='x', c='w', ms=25)
            a = CircularAperture(self.centers[i], r=self.radius)
            a.plot(edgecolor='w', linestyle='--', axes=ax)
            aps[i].plot(edgecolor='w', axes=ax)
        fig.tight_layout()
        return fig

    def plot_profiles(self, axs=None, figsize=None):
        if axs is None:
            fig, axs = subplots(2, self.nstars, figsize=figsize, sharey='all')
        else:
            fig, axs = None, axs

        st = array([[bb.ixmin for bb in self.apertures.bbox],
                    [bb.iymin for bb in self.apertures.bbox]]).T

        x = arange(self.profiles_obs.shape[1])
        for ist in range(self.nstars):
            for ic in range(2):
                ax = axs[ic, ist]
                ax.plot(x + st[ist, ic], self.profiles_obs[ist + self.nstars * ic])
                ax.plot(x + st[ist, ic], self.profiles_mod[ist + self.nstars * ic])
                ax.axvline(self._res.x[9 + ist + self.nstars * ic] + st[ist, ic], ls='--')
                ax.axvline(self.centers[ist, ic])
                [ax.axvspan(self.centers[ist, ic] - r * self.radius, self.centers[ist, ic] + r * self.radius,
                            zorder=-100, alpha=0.1) for r in (0.5, 1.0, 1.5)]
        if fig is not None:
            fig.tight_layout()
        return fig
