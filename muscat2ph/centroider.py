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

import warnings

from astropy.stats import sigma_clipped_stats
from astropy.utils.exceptions import AstropyUserWarning
from matplotlib.pyplot import subplots
from numpy import inf, amin, nan, argsort, meshgrid, clip, percentile, flip, nanmean, log
from numpy import arange, zeros, sqrt, isfinite, array, all, squeeze, ndarray, zeros_like, ceil, argmin, ones, \
    full_like, atleast_2d
from photutils import CircularAperture
from scipy.ndimage import median_filter as mf, center_of_mass as com, median_filter, center_of_mass, label, \
    binary_dilation
from scipy.optimize import minimize

import nudged

def is_inside_frame(p, width: float = 1024, height: float = 1024, xpad: float = 15, ypad: float = 15):
    return (p[:, 0] > ypad) & (p[:, 0] < height - ypad) & (p[:, 1] > xpad) & (p[:, 1] < width - xpad)

def entropy(a):
    a = a - a.min() + 1e-10
    a = a / a.sum()
    return -(a*log(a)).sum()

class Centroider:
    def __init__(self, nstars: int, apt_radius: float = 20.0) -> None:
        self.r = apt_radius
        self.nstars = nstars

        self.image: ndarray = None
        self.apertures = None

        self.new_centers: ndarray = zeros((self.nstars, 2))
        self.ref_centers: ndarray = zeros((self.nstars, 2))
        self.radii: ndarray = zeros(self.nstars)

        self._transform = None
        #self.ref_relative_fluxes = self.calculate_relative_fluxes()

    def update(self, centers, mask=None):
        if mask is None:
            mask = ones(self.nstars, bool)
        self.new_centers = centers.copy()
        self.apertures = CircularAperture(self.new_centers, self.r)

        if self.old_centers is not None:
            self._transform = nudged.estimate(self.old_centers[mask], self.new_centers[mask])

    def estimate_radii(self, mask=None):
        if mask is None:
            mask = ones(self.nstars, bool)

        for i, apt in enumerate(self.apertures):
            if mask[i]:
                center = self.new_centers[i]
                d = apt.to_mask().cutout(self.image).copy()
                x, y = meshgrid(arange(d.shape[1]), arange(d.shape[0]))
                cx, cy = center[0] - apt.bbox.ixmin, center[1] - apt.bbox.iymin
                r = sqrt((x - cx) ** 2 + (y - cy) ** 2).ravel()
                sids = argsort(r)

                # Remove the background
                _, med, istd = sigma_clipped_stats(d)
                d -= med

                # Isolate the star from any possible contaminating sources
                m = d > 3 * istd
                labels, nl = label(m)
                if nl == 0:
                    self.radii[i] = nan
                else:
                    label_centers = atleast_2d([center_of_mass(d, labels, i) for i in range(1, nl + 1)])
                    label_distances = ((label_centers[:, 0] - center[0] + apt.bbox.ixmin) ** 2
                                       + (label_centers[:, 1] - center[1] + apt.bbox.iymin) ** 2)
                    m = binary_dilation(labels == argmin(label_distances) + 1, iterations=3)
                    d *= m

                    r = r[sids]
                    cflux = d.ravel()[sids].cumsum()
                    cflux /= cflux.max()
                    self.radii[i] = r[argmin(abs(cflux - 0.95))]

    def calculate_relative_fluxes(self, centers: ndarray = None):
        if centers is None:
            apts = self.apertures
        else:
            apts = CircularAperture(centers, self.r)
        return apts.do_photometry(self.image)[0] / self.image.sum()

    def detect_jump(self):
        relative_fluxes = self.calculate_relative_fluxes()
        if all(relative_fluxes < 0.8 * self.ref_relative_fluxes):
            return True
        else:
            return False

    def recover_from_jump(self):
        for radius in [100, 200, 300]:
            apt = CircularAperture(self.new_centers[0], radius)
            mask = apt.to_mask()
            d = mask.multiply(self.image)
            m = median_filter(d, 5) > 3 * self.istd
            xy = array(center_of_mass(m))[[1, 0]] + array([apt.bbox.ixmin, apt.bbox.iymin])
            new_centers = self.new_centers + (xy - self.new_centers[0])
            new_rfs = self.calculate_relative_fluxes(new_centers)
            if all(new_rfs > 0.8 * self.ref_relative_fluxes):
                self.update(new_centers)
                return True
        raise ValueError('Could not recover from a jump')

    def transform(self, p):
        return array(self._transform.transform(p.T)).T

    def centroid(self, image: ndarray, ref_centers: ndarray, *nargs, **kwargs):
        raise NotImplementedError

    def plot_profiles(self):
        raise NotImplementedError

    def plot_images(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class COMCentroider(Centroider):

    def centroid(self, image: ndarray, ref_centers: ndarray, maxiter: int = 3):
        self.image = image
        centers_p = ref_centers.copy()
        centers_c = ref_centers.copy()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=(AstropyUserWarning, RuntimeWarning))
            j, converged = 0, False
            while not converged and j < maxiter:
                self.apertures = CircularAperture(centers_p, self.r)
                masks = self.apertures.to_mask()
                good = ones(len(masks), bool)

                for i, (m, b) in enumerate(zip(masks, self.apertures.bbox)):
                    cut = m.cutout(image, fill_value=nan)
                    if cut is None:
                        good[i] = False
                    else:
                        good[i] = all(isfinite(cut))

                        _, med, _ = sigma_clipped_stats(cut)
                        centers_c[i] = array(center_of_mass(cut - med))[[1, 0]] + array([b.ixmin, b.iymin])

                if all(((centers_c[good] - centers_p[good]) ** 2).sum(1) < 0.5):
                    converged = True

                centers_p[good] = centers_c[good]
                j += 1

        self.ref_centers[:] = ref_centers
        self.new_centers[:] = centers_p
        self._transform = nudged.estimate(self.ref_centers[good], self.new_centers[good])
        self.estimate_radii(good)
        return self.new_centers

    def plot_images(self, max_cols: int = 3):
        ncols = min(max_cols, self.nstars)
        nrows = int(ceil(self.nstars / ncols))
        masks = self.apertures.to_mask()

        fig, axs = subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        for i in range(self.nstars):
            try:
                ax = axs.flat[i]
                dd = masks[i].cutout(self.image, fill_value=nan)
                ax.imshow(dd, extent=self.apertures[i].bbox.extent, origin='lower')
                ax.plot(*self.ref_centers[i], marker='+', c='w', ms=25)
                ax.plot(*self.new_centers[i], marker='x', c='k', ms=25)
                if isfinite(self.radii[i]):
                    a = CircularAperture(self.new_centers[i], r=self.radii[i])
                    a.plot(edgecolor='w', linestyle='--', axes=ax)
                self.apertures[i].plot(edgecolor='w', axes=ax)
            except (ValueError, TypeError):
                pass
        fig.tight_layout()
        return fig


class DFCOMCentroider(Centroider):

    def centroid(self, image: ndarray, sigma: float = 4, maxiter: int = 5):
        self.image = median_filter(image, 3)
        _, self.imed, self.istd = sigma_clipped_stats(self.image)
        self.image -= self.imed

        if self.detect_jump():
            self.recover_from_jump()

        centers_p = zeros_like(self.new_centers)
        centers_c = zeros_like(self.new_centers)
        j, converged = 0, False
        while not converged and j < maxiter:
            masks = self.apertures.to_mask()
            for i, (mask, bbox) in enumerate(zip(masks, self.apertures.bbox)):
                d = mask.multiply(self.image)
                m = d > sigma * self.istd
                xy = array(center_of_mass(m))[[1, 0]] + array([bbox.ixmin, bbox.iymin])
                centers_c[i] = xy
            if j > 0:
                if all(((centers_c - centers_p)**2).sum(1) < 1.0):
                    converged = True
            centers_p[:] = centers_c
            j += 1
        self.update(centers_c)
        self.estimate_radii()
        return self.new_centers

    def plot_images(self, max_cols: int = 3):
        ncols = min(max_cols, self.nstars)
        nrows = int(ceil(self.nstars / ncols))

        aps = CircularAperture(self.new_centers, self.r)
        masks = aps.to_mask()

        fig, axs = subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

        for i in range(self.nstars):
            ax = axs.flat[i]
            dd = masks[i].cutout(self.image, fill_value=nan)
            ax.imshow(dd, extent=aps[i].bbox.extent)
            ax.plot(*self.new_centers[i], marker='x', c='w', ms=25)
            if isfinite(self.radii[i]):
                a = CircularAperture(self.new_centers[i], r=self.radii[i])
                a.plot(edgecolor='w', linestyle='--', axes=ax)
            aps[i].plot(edgecolor='w', axes=ax)
        fig.tight_layout()
        return fig

    def plot_psf_profile(self, ax=None, figsize=None):
        if ax is None:
            fig, ax = subplots(figsize=figsize)
        else:
            fig, ax = None, ax

        center = self.new_centers[0]
        apt = self.apertures[0]
        d = apt.to_mask().multiply(self.image)
        l = d.shape[0]
        x, y = meshgrid(arange(l), arange(l))
        cx, cy = center[0] - apt.bbox.ixmin, center[1] - apt.bbox.iymin
        r = sqrt((x - cx) ** 2 + (y - cy) ** 2).ravel()
        sids = argsort(r)
        r = r[sids]

        ax.plot(r, d.ravel()[sids])