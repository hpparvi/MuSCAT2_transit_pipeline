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
from astropy.stats import sigma_clipped_stats
from matplotlib.pyplot import subplots
from numpy import ndarray, argmin, arange, nan, ceil, array, zeros, all, meshgrid, sqrt, argsort, zeros_like
from photutils import CircularAperture
from scipy.ndimage import median_filter, center_of_mass

from .dfcentroider import Centroider

class CMCentroider(Centroider):
    def __init__(self, ref_image: ndarray, ref_centers: ndarray, apt_radius: float = 20):
        super().__init__()

        _, self.imed, self.istd = sigma_clipped_stats(ref_image)
        self.image = ref_image - self.imed
        self.r = apt_radius
        self.nstars = len(ref_centers)

        self.centers = None
        self.apertures = None
        self.update(ref_centers.copy())

        self.ref_relative_fluxes = self.calculate_relative_fluxes()
        self.radii = zeros(self.nstars)

    def update(self, centers):
        self.centers = centers
        self.apertures = CircularAperture(self.centers, self.r)

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

    def calculate_centroid(self, image: ndarray, x0: ndarray = None, update: bool = True, sigma: float = 5):
        _, self.imed, self.istd = sigma_clipped_stats(image)
        self.image = image - self.imed

        if self.detect_jump():
            self.recover_from_jump()

        masks = self.apertures.to_mask()
        centers = zeros_like(self.centers)
        for i, (mask, bbox) in enumerate(zip(masks, self.apertures.bbox)):
            d = mask.multiply(self.image)
            m = median_filter(d, 5) > sigma * self.istd
            xy = array(center_of_mass(m))[[1, 0]] + array([bbox.ixmin, bbox.iymin])
            centers[i] = xy

        self.update(centers)
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