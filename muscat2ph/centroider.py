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

from numpy import inf, amin, nan, argsort, meshgrid, clip, percentile, flip, nanmean, log
from numpy import arange, zeros, sqrt, isfinite, array, all, squeeze
from photutils import CircularAperture
from scipy.ndimage import median_filter as mf, center_of_mass as com
from scipy.optimize import minimize

import nudged

def is_inside_frame(p, width: float = 1024, height: float = 1024, xpad: float = 15, ypad: float = 15):
    return (p[:, 0] > ypad) & (p[:, 0] < height - ypad) & (p[:, 1] > xpad) & (p[:, 1] < width - xpad)

def entropy(a):
    a = a - a.min() + 1e-10
    a = a / a.sum()
    return -(a*log(a)).sum()

class Centroider:

    def __init__(self, image, sids=None, nstars: int = 10, aperture_radius: float = 20):
        self.image = image
        self.image.centroider = self
        self.r = aperture_radius

        if sids is None:
            self.select_stars(nstars)
        else:
            self.sids = squeeze(sids)
            self.nstars = self.sids.size
            self.apt = CircularAperture(self.image._cur_centroids_pix[self.sids], self.r)

    def calculate_minimum_distances(self):
        pos = self.image._cur_centroids_pix
        dmin = zeros(pos.shape[0])
        for i in range(pos.shape[0]):
            d = sqrt(((pos - pos[i]) ** 2).sum(1))
            d[i] = inf
            dmin[i] = d.min()
        return dmin

    def create_saturation_mask(self, saturation_limit: float = 50_000, apt_radius: float = 15):
        """Creates a mask that excludes stars close to the linearity limit"""
        apts = CircularAperture(self.image._cur_centroids_pix, r=apt_radius)
        data = self.image.reduced.copy()
        data[data > saturation_limit] = nan
        return isfinite(apts.do_photometry(data)[0])

    def create_separation_mask(self, minimum_separation: float = 15):
        dmin = self.calculate_minimum_distances()
        return dmin > minimum_separation

    def select_stars(self, nstars: int = 8,
                                 min_separation: float = 15,
                                 sat_limit: float = 50_000,
                                 apt_radius: float = 15,
                                 border_margin: float = 50):
        centroid_mask = (self.create_separation_mask(min_separation)
                         & self.create_saturation_mask(sat_limit, apt_radius)
                         & is_inside_frame(self.image._ref_centroids_pix, xpad=border_margin, ypad=border_margin))

        sids = argsort(self.image._flux_ratios)[::-1]
        centroid_mask = centroid_mask[sids]
        ids = arange(centroid_mask.size)[sids]
        ids = ids[centroid_mask][:nstars]
        self.sids = ids
        self.nstars = len(self.sids)
        self.apt = CircularAperture(self.image._cur_centroids_pix[self.sids], self.r)
        self.image.centroid_star_ids = ids

    def set_data(self, image, filter_footprint=4):
        self.data = mf(self.image.reduced, filter_footprint)

    def calculate_centroid_shift(self):
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

    def calculate_centroid_shift(self, pmin=80, pmax=95, niter=3):
        c0 = self.image._cur_centroids_pix[self.sids].copy()
        self.apt.positions[:] = c0
        reduced_frame = self.image.reduced
        shifts = zeros((self.nstars, 2))
        self.transform = None

        for iiter in range(niter):
            masks = self.apt.to_mask()
            for istar, mask in enumerate(masks):
                try:
                    cutout = mask.cutout(reduced_frame)
                except ValueError:
                    cutout = None
                if cutout is not None:
                    cutout = cutout.copy()
                    p = percentile(cutout, [pmin, pmax])
                    clipped_cutout = clip(cutout, *p) - p[0]
                    c = com(clipped_cutout)
                    shifts[istar,:] = flip(c,0) - self.apt.r
                else:
                    shifts[istar,:] = (nan, nan)
            m = all(isfinite(shifts),1)
            t = nudged.estimate(c0[m], c0[m]+shifts[m])
            self.apt.positions[:] = array(t.transform(c0.T)).T
        return c0 + shifts

    def calculate_and_apply(self, pmin=80, pmax=95, niter=3):
        shifts = self.calculate_centroid_shift(pmin, pmax, niter)
        m = all(isfinite(shifts), 1)
        self.transform = nudged.estimate(self.image._ref_centroids_pix[self.sids][m], shifts[m])
        self.image._cur_centroids_pix[:] = array(self.transform.transform(self.image._ref_centroids_pix.T)).T
        self.image._update_apertures(self.image._cur_centroids_pix)