import warnings
from textwrap import dedent

import numpy as np
import xarray as xa


from astropy.io import fits as pf
from astropy.visualization import simple_norm as sn
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord, FK5
from astropy.wcs import WCS, FITSFixedWarning
from astropy.table import Table
from matplotlib import cm
from matplotlib.pyplot import subplots, figure, subplot
from numpy import *
from pathlib import Path
from photutils import CircularAperture, CircularAnnulus
from photutils.centroids import centroid_com
from scipy.ndimage import label, labeled_comprehension, median_filter as mf
from scipy.ndimage import center_of_mass as com
from scipy.optimize import minimize
from scipy.stats import scoreatpercentile as sap
from scipy.spatial import distance as ds
from tqdm import tqdm
import pandas as pd
import astropy.units as u

from photutils import DAOStarFinder, DAOPhotPSFPhotometry, DAOGroup, CircularAperture
from scipy.stats import scoreatpercentile as sap, cauchy

passbands = 'g r i z_s'.split()

def dflat(root, passband):
    return root.joinpath('calibs', 'flat', passband)

def ddark(root, passband):
    return root.joinpath('calibs', 'dark', passband)

def entropy(a):
    a = a - a.min() + 1e-10
    a = a / a.sum()
    return -(a*log(a)).sum()

class Centroider:
    def __init__(self, image, sids=None, nstars=inf, aperture_radius=20):
        self.image = image
        self.sids = sids if sids is not None else arange(min(image._cur_centroids_pix.shape[0], nstars))
        self.r = aperture_radius
        self.nstars = len(self.sids)
        self.apt = CircularAperture(zeros((self.nstars, 2)), self.r)

    def select_stars(self, min_distance=20., min_edge=50.):
        im = self.image
        xy = im._ref_centroids_pix
        distances = ds.cdist(xy, xy)
        mask_distance = (distances < min_distance).sum(1) < 2
        mask_edge = (all(xy > min_edge, 1) & all(xy < im._data.shape[0] - min_edge, 1))
        self.sids = where(mask_distance & mask_edge)[0]
        self.nstars = len(self.sids)
        self.apt = CircularAperture(zeros((self.nstars, 2)), self.r)


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
        c0 = self.image._cur_centroids_pix[self.sids, :].copy()
        self.apt.positions[:] = c0
        #if any(cpix < 0.) or any(cpix > im._data.shape[0]):
        #    self.centroid = [nan, nan]
        #    raise ValueError("Star outside the image FOV.")
        reduced_frame = self.image.reduced
        shifts = zeros((self.nstars, 2))
        for iiter in range(niter):
            masks = self.apt.to_mask()
            for istar, mask in enumerate(masks):
                cutout = mask.cutout(reduced_frame)
                if cutout is not None:
                    cutout = cutout.copy()
                    p = percentile(cutout, [pmin, pmax])
                    clipped_cutout = clip(cutout, *p) - p[0]
                    c = com(clipped_cutout)
                    shifts[istar,:] = flip(c,0) - self.apt.r
                else:
                    shifts[istar,:] = (nan, nan)
            self.apt.positions[:] += nanmean(shifts, 0)
        return (self.apt.positions - c0).mean(0)

    def calculate_and_apply(self, pmin=80, pmax=95, niter=3):
        shift = self.calculate_centroid_shift(pmin, pmax, niter)
        self.image._cur_centroids_pix[:] += shift
        self.image._update_apertures(self.image._cur_centroids_pix)


class ImageFrame:
    def __init__(self, *nargs, **kwargs):
        raise NotImplementedError

    def plot(self, data=None, ax=None, figsize=(6,6), title='', minp=10, maxp=100):
        data = data if data is not None else self._data
        if not ax:
            fig = figure(figsize=figsize)
            if hasattr(self, '_wcs'):
                if self._wcs :
                    ax = subplot(projection=self._wcs)
                    ax.grid()
            else:
                ax = subplot()
        ax.imshow(data, cmap=cm.gray_r, origin='image',
                  norm=sn(data, stretch='log', min_percent=minp, max_percent=maxp))
        ax.set_title(title)
        return ax

class MasterFrame(ImageFrame):
    def __init__(self, root, passband, name, force=False, verbose=True):
        self.name = name
        self.passband = passband
        self._file = Path(root).joinpath('calibs', '{}_{}.fits'.format(name, passband))
        self._data = None
        self.root = root
        self.force = force
        self.verbose = verbose

    def __call__(self):
        raise NotImplementedError

    @property
    def generate_file(self):
        return not self._file.exists() or self.force

    def average_data(self, files, prefun=lambda d:d):
        files = list(files)
        for fn in tqdm(files, desc='Creating {}'.format(self.name)):
            with pf.open(fn) as f:
                et = float(f[0].header['exptime'])
                d = prefun(f[0].data.astype('d'))
                self._data = d if self._data is None else self._data + d
            self._data /= len(files)


class MasterDark(MasterFrame):
    def __init__(self, root, passband, force=False, verbose=True):
        super().__init__(root, passband, 'masterdark', force, verbose)
        if self.generate_file:
            files = list(ddark(root, passband).glob('*.fits'))
            datacube = zeros((len(files), 1024, 1024))
            for i,fn in tqdm(enumerate(files), desc='Creating masterdark'):
                datacube[i,:,:] = pf.getdata(fn)
            self._data = datacube.mean(0)
            pf.writeto(self._file, self._data, overwrite=True)
        else:
            self._data = pf.getdata(self._file)


class MasterFlat(MasterFrame):
    def __init__(self, root, passband, force=False, verbose=True):
        super().__init__(root, passband, 'masterflat', force, verbose)
        if self.generate_file:
            md = MasterDark(root, passband)
            files = list(dflat(root, passband).glob('*.fits'))
            datacube = zeros((len(files), 1024, 1024))
            for i,fn in tqdm(enumerate(files), desc='Creating masterflat'):
                datacube[i,:,:] = pf.getdata(fn) - md._data
            self._data = datacube.mean(0)
            self._data /= median(self._data)
            pf.writeto(self._file, self._data, overwrite=True)
        else:
            self._data = pf.getdata(self._file)

class ScienceFrame(ImageFrame):

    height = 1024
    width  = 1024

    def __init__(self, root, passband, masterdark=None, masterflat=None, aperture_radii=(8, 12, 16, 20, 24, 30, 40, 50, 60)):
        self.passband = passband
        self.aperture_radii = aperture_radii
        self.napt = len(aperture_radii)

        # Calibration files
        # ------------------
        self.masterdark = self.dark = masterdark or MasterDark(root, self.passband)
        self.masterflat = self.flat = masterflat or MasterFlat(root, self.passband)

        # Basic data
        # ----------
        self._data = None               # FITS image data
        self._header = None             # FITS header
        self._wcs = None                # WCS data (if available)
        self._ref_centroids_sky = None  # Star centroids in sky coordinates [DataFrame]
        self._ref_centroids_pix = None  # Star centroids in pixel coordinates [DataFrame]
        self._cur_centroids_pix = None  # Star centroids in pixel coordinates [DataFrame]
        self._apertures_obj = None      # Photometry apertures
        self._apertures_sky = None      # Photometry apertures

        self._flux = None
        self._entropy = None
        self._cshift = None
        self._sky_entropy = None
        self._sky_median = None

        self._xad_star = None
        self._xad_apt = xa.IndexVariable(data=array(aperture_radii), dims='aperture')

        self.soft_centroider = self.centroid_soft
        self.rigid_centroider = None


    def load_fits(self, filename):
        """Load the image data and WCS information (if available)."""
        filename = Path(filename)
        with pf.open(filename) as f:
            self._data = f[0].data.astype('d')
            self._header = f[0].header
        wcsfile = filename.parent.joinpath(filename.stem+'.wcs')
        if wcsfile.exists():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FITSFixedWarning)
                self._wcs = WCS(pf.getheader(wcsfile))
            if self._ref_centroids_sky is not None:
                self._cur_centroids_pix = array(self._ref_centroids_sky.to_pixel(self._wcs)).T


    def _initialize_tables(self, nstars, napt):
        iapt = self._xad_apt
        self._xad_star = istar = xa.IndexVariable(dims='star', data=range(self.nstars))
        self._flux = xa.DataArray(zeros([nstars, napt]),
                                  name='flux', dims=['star', 'aperture'],
                                  coords={'star':istar, 'aperture': iapt})
        self._entropy = xa.DataArray(zeros([nstars, napt]),
                                     name='aperture_entropy', dims=['star', 'aperture'],
                                     coords={'star':istar, 'aperture': iapt})
        self._cshift = xa.DataArray(zeros([nstars, 2]),
                                    name='centroid', dims=['star', 'axis'],
                                    coords={'star':istar, 'axis': ['x', 'y']})
        self._sky_median = xa.DataArray(zeros(nstars), name='sky_median', dims='star', coords={'star':istar})
        self._sky_entropy = xa.DataArray(zeros(nstars), name='sky_entropy', dims='star', coords={'star':istar})


    @property
    def reduced(self):
        """Reduced image."""
        return (self._data - self.dark._data) / self.flat._data

    @property
    def raw(self):
        return self._data

    @property
    def flux(self):
        return self._flux.copy()

    @property
    def apt_entropy(self):
        return self._entropy.copy()

    @property
    def cshift(self):
        return self._cshift.copy()

    def load_reference_stars(self, fname):
        t = Table.read(fname)
        csky = array(t[['ra', 'dec']]) if 'ra' in t.colnames else None
        cpix = t.to_pandas()[['x', 'y']].values
        self.set_reference_stars(cpix, csky)
        self._initialize_tables(self.nstars, self.napt)

    def save_reference_stars(self, fname):
        if self._wcs:
            t = Table(c_[self._ref_centroids_sky.ra.deg, self._ref_centroids_sky.dec.deg, self._ref_centroids_pix],
                      names='ra dec x y'.split())
        else:
            t = Table(self._ref_centroids_pix, names=['x', 'y'])
        t.write(fname)


    def set_reference_stars(self, cpix, csky=None, wsky=15):
        self._ref_centroids_pix = array(cpix)
        if csky is not None:
            self._ref_centroids_sky = SkyCoord(array(csky), frame=FK5, unit=u.deg)
            if self._wcs is not None:
                self._ref_centroids_pix = array(self._ref_centroids_sky.to_pixel(self._wcs)).T
        self._cur_centroids_pix = cpix = self._ref_centroids_pix.copy()
        self._apertures_obj = [CircularAperture(cpix, r) for r in self.aperture_radii]
        self._apertures_sky = CircularAnnulus(cpix, self.aperture_radii[-1], self.aperture_radii[-1] + wsky)
        self.nstars = self._ref_centroids_pix.shape[0]


    def _update_apertures(self, positions):
        if self._apertures_obj:
            for apt in self._apertures_obj:
                apt.positions[:] = positions
            self._apertures_sky.positions[:] = positions

    def is_inside_boundaries(self, boundary=40):
        blow = (self._aps.positions < self._data.shape[0] - boundary).all(1)
        bhigh = (self._aps.positions > boundary).all(1)
        return blow & bhigh


    def find_stars(self, treshold=99.5, maxn=10):
        objects = mf(self.reduced, 6) > sap(self.reduced, treshold)
        labels, nl = label(objects)
        fluxes = [self.reduced[labels == l].mean() for l in range(1, nl + 1)]
        fids = argsort(fluxes)[::-1] + 1
        maxn = min(maxn, nl)
        self.nstars = maxn

        sorted_labels = zeros_like(labels)
        for i, fid in enumerate(fids[:maxn]):
            sorted_labels[labels == fid] = i + 1
        cpix = flip(array([com(self.reduced - median(self.reduced), sorted_labels, i) for i in range(1, maxn+1)]), 1)

        self._initialize_tables(self.nstars, self.napt)
        if self._wcs:
            csky = pd.DataFrame(self._wcs.all_pix2world(cpix, 0), columns='RA Dec'.split())
            self.set_reference_stars(cpix, csky=csky)
        else:
            self.set_reference_stars(cpix)


    def find_stars_dao(self, fwhm=10, treshold=90, maxn=10):
        sfinder = DAOStarFinder(sap(self.reduced, treshold), fwhm, exclude_border=True)
        stars = sfinder(self.reduced).to_pandas().sort_values('flux', ascending=False)[:maxn]
        stars.index = arange(stars.shape[0])
        stars.drop('id roundness1 roundness2 npix sky flux mag'.split(), axis=1, inplace=True)
        return stars

    def centroid_single(self, star, r=20, pmin=80, pmax=95, niter=1):
        #if any(cpix < 0.) or any(cpix > im._data.shape[0]):
        #    self.centroid = [nan, nan]
        #    raise ValueError("Star outside the image FOV.")
        apt = CircularAperture(self._cur_centroids_pix[star, :], r)
        reduced_frame = self.reduced
        for iiter in range(niter):
            mask = apt.to_mask()[0]
            cutout = mask.cutout(reduced_frame).copy()
            p = percentile(cutout, [pmin, pmax])
            clipped_cutout = clip(cutout, *p) - p[0]
            c = com(clipped_cutout)
            apt.positions[:] = flip(c, 0) + array([mask.bbox.slices[1].start, mask.bbox.slices[0].start])
        self._cur_centroids_pix[star, :] = apt.positions
        self._update_apertures(self._cur_centroids_pix)


    def centroid_soft(self, r=20, pmin=80, pmax=95, niter=1):
        for istar in range(self.nstars):
            try:
                self.centroid_single(istar, r, pmin, pmax, niter)
            except ValueError:
                pass

    def centroid_rigid(self):
        self.rigid_centroider.set_data(self.reduced)
        self._centroid_result = r = self.rigid_centroider()
        self._cur_centroids_pix[:] += r.x
        self._update_apertures(self._cur_centroids_pix)

    def get_aperture(self, aid=0):
        m = self._apertures_obj[-1].to_mask()[aid]
        return m.multiply(self.reduced)

    def plot_aperture_masks(self, radius=None, minp=0.0, maxp=99.9, cols=5, figsize=(11, 2.5)):
        if radius is not None:
            aps = CircularAperture([self._stars.xcentroid, self._stars.ycentroid], radius)
        else:
            aps = self._aps
        fig, axs = subplots(int(ceil(self.nstars / cols)), cols, figsize=figsize, sharex=True, sharey=True)
        for m, ax in zip(aps.to_mask(), axs.flat):
            #d = m.multiply(self.reduced)
            d = where(m.data.astype('bool'), m.cutout(self.reduced), nan)
            ax.imshow(d, cmap=cm.gray_r, origin='image',
                      norm=sn(d, stretch='log', min_percent=minp, max_percent=maxp))
        fig.tight_layout()

    def plot_sky_masks(self, r1, r2, minp=0.0, maxp=99.9, cols=5, figsize=(11, 2.5)):
        aps = CircularAnnulus([self._stars.xcentroid, self._stars.ycentroid], r1, r2)
        fig, axs = subplots(int(ceil(self.nstars / cols)), cols, figsize=figsize, sharex=True, sharey=True)
        for m, ax in zip(aps.to_mask(), axs.flat):
            d = where(m.data.astype('bool'), m.cutout(self.reduced), nan) #m.multiply(self.reduced)
            ax.imshow(d, cmap=cm.gray_r, origin='image',
                      norm=sn(d, stretch='linear', min_percent=minp, max_percent=maxp))
        fig.tight_layout()

    def plot_apertures(self, ax, offset=9):
        if self._apertures_obj is not None:
            [apt.plot(ax=ax, alpha=0.25) for apt in self._apertures_obj]
            for istar, (x,y) in enumerate(self._apertures_obj[0].positions):
                ax.text(x+offset, y+offset, istar)
        if self._apertures_sky is not None:
            self._apertures_sky.plot(ax=ax, alpha=0.25)

    def plot_raw(self, ax=None, figsize=(6,6), plot_apertures=True, minp=10, maxp=100):
        ax = super().plot(self.raw, ax=ax, figsize=figsize, title='Reduced image',
                            minp=minp, maxp=maxp)
        if plot_apertures:
            self.plot_apertures(ax)
        return ax

    def plot_reduced(self, ax=None, figsize=(6,6), plot_apertures=True,  minp=10, maxp=100, flt=lambda a:a, **kwargs):
        ax = super().plot(flt(self.reduced), ax=ax, figsize=figsize, title='Reduced image', minp=minp, maxp=maxp)
        if plot_apertures:
            self.plot_apertures(ax, **kwargs)
        return ax

    def plot_reduction(self, figsize=(11,12)):
        fig, axs = subplots(2,2, figsize=figsize, sharex=True, sharey=True)
        super().plot(self.raw, axs[0,0], title='Raw image')
        super().plot(self.reduced, axs[0,1], title='Reduced image')
        super().plot(self.flat._data, axs[1,0], title='Masterflat')
        super().plot(self.dark._data, axs[1,1], title='Masterdark')
        fig.tight_layout()
        return fig, axs

    def estimate_sky(self):
        masks = self._apertures_sky.to_mask()
        for istar, m in enumerate(masks):
            d = m.cutout(self.reduced)
            if d is not None:
                self._sky_median[istar] = sigma_clipped_stats(d[m.data.astype('bool')], sigma=4)[1]
            else:
                self._sky_median[istar] = nan
        return self._sky_median

    def estimate_entropy(self):

        # Sky entropy
        # -----------
        masks = self._apertures_sky.to_mask()
        for istar, m in enumerate(masks):
            d = m.cutout(self.reduced)
            if d is not None:
                self._sky_entropy[istar] = entropy(d[m.data.astype('bool')])
            else:
                self._sky_entropy[istar] = nan

                # Object entropy
        # --------------
        for iaperture, apt in enumerate(self._apertures_obj):
            masks = apt.to_mask()
            for istar, m in enumerate(masks):
                d = m.cutout(self.reduced)
                if d is not None:
                    self._entropy[istar, iaperture] = entropy(d[m.data.astype('bool')])
                else:
                    self._entropy[istar, iaperture] = nan

        return self._entropy, self._sky_entropy


    def photometry(self, centroid=True):
        if centroid:
            self.centroid_rigid()
        self.estimate_sky()
        self.estimate_entropy()
        for iapt, apt in enumerate(self._apertures_obj):
            self._flux[:, iapt] = apt.do_photometry(self.reduced)[0] - self._sky_median * apt.area()
        self._cshift[:] = self._cur_centroids_pix - self._ref_centroids_pix
        return self.flux, self._sky_median.copy(), self.apt_entropy, self._sky_entropy.copy(), self.cshift


def apt_values(apt, im):
    m = apt.to_mask()[0]
    return m.cutout(im.reduced)[m.data.astype('bool')]
