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
from matplotlib.pyplot import subplots
from numpy import inf, amin, nan, argsort, meshgrid, clip, percentile, flip, nanmean, log
from numpy import arange, zeros, sqrt, isfinite, array, all, squeeze, ndarray, zeros_like, ceil, argmin
from photutils import CircularAperture
from scipy.ndimage import median_filter as mf, center_of_mass as com, median_filter, center_of_mass
from scipy.optimize import minimize

import nudged

def is_inside_frame(p, width: float = 1024, height: float = 1024, xpad: float = 15, ypad: float = 15):
    return (p[:, 0] > ypad) & (p[:, 0] < height - ypad) & (p[:, 1] > xpad) & (p[:, 1] < width - xpad)

def entropy(a):
    a = a - a.min() + 1e-10
    a = a / a.sum()
    return -(a*log(a)).sum()

class Centroider:
    def __init__(self, ref_image: ndarray, ref_centers: ndarray, apt_radius: float = 20) -> None:
        _, self.imed, self.istd = sigma_clipped_stats(ref_image)
        self.image = ref_image - self.imed
        self.r = apt_radius
        self.nstars = len(ref_centers)

        self.centers = None
        self.old_centers = None
        self.apertures = None
        self._transform = None

        self.update(ref_centers.copy())
        self.ref_relative_fluxes = self.calculate_relative_fluxes()
        self.radii = zeros(self.nstars)

    def update(self, centers):
        if self.centers is not None:
            self.old_centers = self.centers.copy()
        else:
            self.old_centers = centers.copy()
        self.centers = centers.copy()
        self.apertures = CircularAperture(self.centers, self.r)

        if self.old_centers is not None:
            self._transform = nudged.estimate(self.old_centers, self.centers)

    def estimate_radii(self):
        try:
            for i, apt in enumerate(self.apertures):
                center = self.centers[i]
                d = apt.to_mask().multiply(self.image)
                x, y = meshgrid(arange(d.shape[1]), arange(d.shape[0]))
                cx, cy = center[0] - apt.bbox.ixmin, center[1] - apt.bbox.iymin
                r = sqrt((x - cx) ** 2 + (y - cy) ** 2).ravel()
                sids = argsort(r)
                r = r[sids]
                cflux = d.ravel()[sids].cumsum()
                cflux /= cflux.max()
                self.radii[i] = r[argmin(abs(cflux - 0.95))]
        except IndexError:
            pass

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
            apt = CircularAperture(self.centers[0], radius)
            mask = apt.to_mask()
            d = mask.multiply(self.image)
            m = median_filter(d, 5) > 3 * self.istd
            xy = array(center_of_mass(m))[[1, 0]] + array([apt.bbox.ixmin, apt.bbox.iymin])
            new_centers = self.centers + (xy - self.centers[0])
            new_rfs = self.calculate_relative_fluxes(new_centers)
            if all(new_rfs > 0.8 * self.ref_relative_fluxes):
                self.update(new_centers)
                return True
        raise ValueError('Could not recover from a jump')

    def transform(self, p):
        return array(self._transform.transform(p.T)).T

    def calculate_centroids(self, image: ndarray, *nargs, **kwargs):
        raise NotImplementedError

    def plot_profiles(self):
        raise NotImplementedError

    def plot_images(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class EntropyCentroider(Centroider):
    """An attempt to create a centroider that uses multiple stars"""

    def __init__(self, image, sids=None, nstars=5, aperture_radius=20):
        super().__init__(image, sids, nstars, aperture_radius)
        m = self.apt.to_mask()[0]
        x = arange(m.data.shape[1])
        y = arange(m.data.shape[0])
        self.X, self.Y = meshgrid(x, y)
        self.set_data(image)

    def shift_center(self, shifts):
        self.apt.positions[:, :] = self.c0 + shifts

    def c_entropy(self, shifts):
        self.shift_center(shifts)
        masks = self.apt.to_mask()
        total_entropy = 0.0
        a = 0.9
        for s, m in zip(self.apt.positions, masks):
            d = m.cutout(self.data).copy()
            d -= d.min()
            w = sqrt((self.X + m.bbox.ixmin - s[0]) ** 2 + (self.Y + m.bbox.iymin - s[1]) ** 2)
            w = (1-a)*self.ramp(w, self.r) + a*self.smootherstep(w, self.r, 4)
            total_entropy += entropy(d * w)
        return -total_entropy

    def ramp(self, x, d):
        return clip(1. - x / d, 0., 1.)

    def smootherstep(self, x, edge0, edge1):
        x = (x - edge0) / (edge1 - edge0)
        x = clip(x, 0.0, 1.0)
        return x * x * x * (x * (x * 6. - 15.) + 10.)

    def __call__(self, bounds=((-20, 20), (-20, 20))):
        return minimize(self.c_entropy, array([0., 0.]), bounds=bounds)

class COMCentroider(Centroider):

    def calculate_centroids(self, image: ndarray, sigma: float = 4, maxiter: int = 5):
        _, self.imed, self.istd = sigma_clipped_stats(image)
        self.image = image - self.imed

        if self.detect_jump():
            self.recover_from_jump()

        centers_p = zeros_like(self.centers)
        centers_c = zeros_like(self.centers)
        j, converged = 0, False
        while not converged and j < maxiter:
            masks = self.apertures.to_mask()
            for i, (mask, bbox) in enumerate(zip(masks, self.apertures.bbox)):
                d = mask.multiply(self.image)
                xy = array(center_of_mass(d))[[1, 0]] + array([bbox.ixmin, bbox.iymin])
                centers_c[i] = xy
            if j > 0:
                if all(((centers_c - centers_p)**2).sum(1) < 1.0):
                    converged = True
            centers_p[:] = centers_c
            j += 1
        self.update(centers_c)
        self.estimate_radii()
        return self.centers


class DFCOMCentroider(Centroider):

    def calculate_centroids(self, image: ndarray, sigma: float = 4, maxiter: int = 5):
        self.image = median_filter(image, 3)
        _, self.imed, self.istd = sigma_clipped_stats(self.image)
        self.image -= self.imed

        if self.detect_jump():
            self.recover_from_jump()

        centers_p = zeros_like(self.centers)
        centers_c = zeros_like(self.centers)
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
        return self.centers

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
            a = CircularAperture(self.centers[i], r=self.radii[i])
            a.plot(edgecolor='w', linestyle='--', axes=ax)
            aps[i].plot(edgecolor='w', axes=ax)
        fig.tight_layout()
        return fig

    def plot_psf_profile(self, ax=None, figsize=None):
        if ax is None:
            fig, ax = subplots(figsize=figsize)
        else:
            fig, ax = None, ax

        center = self.centers[0]
        apt = self.apertures[0]
        d = apt.to_mask().multiply(self.image)
        l = d.shape[0]
        x, y = meshgrid(arange(l), arange(l))
        cx, cy = center[0] - apt.bbox.ixmin, center[1] - apt.bbox.iymin
        r = sqrt((x - cx) ** 2 + (y - cy) ** 2).ravel()
        sids = argsort(r)
        r = r[sids]

        ax.plot(r, d.ravel()[sids])