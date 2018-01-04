import warnings
from textwrap import dedent

import numpy as np
import xarray as xa


from astropy.io import fits as pf
from astropy.visualization import simple_norm as sn
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS, FITSFixedWarning
from matplotlib import cm
from matplotlib.pyplot import subplots
from numpy import *
from pathlib import Path
from photutils import CircularAperture, CircularAnnulus
from photutils.centroids import centroid_com
from scipy.ndimage import label, labeled_comprehension, median_filter as mf
from scipy.ndimage import center_of_mass as com
from scipy.stats import scoreatpercentile as sap
from tqdm import tqdm
import pandas as pd

from photutils import DAOStarFinder, DAOPhotPSFPhotometry, DAOGroup, CircularAperture
from scipy.stats import scoreatpercentile as sap, cauchy

from .psf import GaussianDFPSF, centroid_m

passbands = 'g r i z_s'.split()

def dflat(root, passband):
    return root.joinpath('calibs', 'flat', passband)

def ddark(root, passband):
    return root.joinpath('calibs', 'dark', passband)

def entropy(a):
    a = a - a.min() + 1e-10
    a = a / a.sum()
    return -(a*log(a)).sum()

class ImageFrame:
    def __init__(self, *nargs, **kwargs):
        raise NotImplementedError

    def plot(self, data=None, ax=None, figsize=(6,6), title='', minp=10, maxp=100):
        data = data if data is not None else self._data
        if not ax:
            fig, ax = subplots(1, 1, figsize=figsize)
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
    def __init__(self, root, passband, masterdark=None, masterflat=None):
        self.passband = passband

        # Calibration files
        # ------------------
        self.masterdark = self.dark = masterdark or MasterDark(root, self.passband)
        self.masterflat = self.flat = masterflat or MasterFlat(root, self.passband)

        # Basic data
        # ----------
        self._data = None       # FITS image data
        self._header = None     # FITS header
        self._wcs = None        # WCS data
        self._centroids = None  # Star centroids in sky coordinates [DataFrame]
        self._stars = None      # Star instances

    def load_fits(self, filename):
        """Load the image data and WCS information from a fits."""
        filename = Path(filename)
        with pf.open(filename) as f:
            self._data = f[0].data.astype('d')
            self._header = f[0].header
        wcsfile = filename.parent.joinpath(filename.stem+'.wcs')
        if wcsfile.exists():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FITSFixedWarning)
                self._wcs = WCS(pf.getheader(wcsfile))
        else:
            self._wcs = WCS(self._header)

    @property
    def reduced(self):
        """Reduced image."""
        return (self._data - self.dark._data) / self.flat._data

    @property
    def raw(self):
        return self._data

    def set_stars(self, csky, apertures=[8,12,16,20]):
        self._centroids = csky
        self._stars = [Star(ra, dec, apertures=apertures) for ra,dec in csky.values]
        for s in self._stars:
            try:
                s.calculate_centroid(self)
            except ValueError:
                pass

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

        sorted_labels = zeros_like(labels)
        for i, fid in enumerate(fids[:maxn]):
            sorted_labels[labels == fid] = i + 1

        cpix = flip(array([com(self.reduced - median(self.reduced), sorted_labels, i) for i in range(1, maxn+1)]), 1)
        csky = pd.DataFrame(self._wcs.all_pix2world(cpix, 1), columns='RA Dec'.split())
        self.set_stars(csky)

    def find_stars_dao(self, fwhm=10, treshold=90, maxn=10):
        sfinder = DAOStarFinder(sap(self.reduced, treshold), fwhm, exclude_border=True)
        stars = sfinder(self.reduced).to_pandas().sort_values('flux', ascending=False)[:maxn]
        stars.index = arange(stars.shape[0])
        stars.drop('id roundness1 roundness2 npix sky flux mag'.split(), axis=1, inplace=True)
        return stars

    def centroid_stars(self, r=20, pmin=80, pmax=95, niter=1):
        for s in self._stars:
            try:
                s.calculate_centroid(self, r, pmin, pmax, niter)
            except ValueError:
                pass

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

    def plot_apertures(self, ax, offset=5):
        if self._stars is not None:
            for istar,s in enumerate(self._stars):
                if all(isfinite(s.centroid)):
                    apt = s._aobj[-1]
                    apt.plot(ax=ax)
                    ax.text(apt.positions[0,0]+offset, apt.positions[0,1]+offset, istar)

    def plot_raw(self, ax=None, figsize=(6,6), plot_apertures=True, minp=10, maxp=100):
        ax = super().plot(self.raw, ax=ax, figsize=figsize, title='Reduced image',
                            minp=minp, maxp=maxp)
        if plot_apertures:
            self.plot_apertures(ax)
        return ax

    def plot_reduced(self, ax=None, figsize=(6,6), plot_apertures=True,  minp=10, maxp=100, flt=lambda a:a, **kwargs):
        ax = super().plot(flt(self.reduced), ax=ax, figsize=figsize, title='Reduced image',
                           minp=minp, maxp=maxp)
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


def apt_values(apt, im):
    m = apt.to_mask()[0]
    return m.cutout(im.reduced)[m.data.astype('bool')]


class Star:
    def __init__(self, ra, dec, apertures, wsky=10):
        self.coords = SkyCoord(ra, dec, unit='deg', frame='fk5')
        self.apertures = apertures
        self._aobj = [CircularAperture([0., 0.], r) for r in apertures]
        self._asky = CircularAnnulus([0., 0.], apertures[-1], apertures[-1] + wsky)
        self.napt = napt = len(apertures)

        self._flux = xa.DataArray(zeros(napt), name='flux', dims='aperture',
                                  coords={'aperture': list(apertures)})
        self._apt_entropy = xa.DataArray(zeros(napt), name='aperture_entropy', dims='aperture',
                                         coords={'aperture': list(apertures)})
        self._centroid = xa.DataArray(self._asky.positions[0], name='centroid', dims='axis',
                                      coords={'axis': ['x', 'y']})
        self._sky_entropy = None
        self._sky_median = None


    def __repr__(self):
        st = """
             Star: Apertures {}
               (RA, Dec): {:6.3f} {:6.3f}
               (x,  y):   {:6.1f} {:6.1f}
             """
        return dedent(st).format(self.apertures, self.coords.ra.value, self.coords.dec.value,
                         self._asky.positions[0,0], self._asky.positions[0,1])

    @property
    def flux(self):
        return self._flux.copy()

    @property
    def apt_entropy(self):
        return self._apt_entropy.copy()

    @property
    def centroid(self):
        return self._centroid.copy()

    @centroid.setter
    def centroid(self, xy):
        self._centroid[:] = xy
        self._asky.positions[:] = xy
        for ao in self._aobj:
            ao.positions[:] = xy

    def calculate_centroid(self, im, r=20, pmin=80, pmax=95, niter=1):
        cpix = array(self.coords.to_pixel(im._wcs))
        #if any(cpix < 0.) or any(cpix > im._data.shape[0]):
        #    self.centroid = [nan, nan]
        #    raise ValueError("Star outside the image FOV.")
        apt = CircularAperture(cpix, r)
        for iiter in range(niter):
            mask = apt.to_mask()[0]
            d = mask.cutout(im.reduced)
            p = percentile(d, [pmin, pmax])
            d2 = clip(d, *p) - p[0]
            c = com(d2)
            apt.positions[:] = flip(c, 0) + array([mask.bbox.slices[1].start, mask.bbox.slices[0].start])
        self.centroid = apt.positions

    def estimate_sky(self, im):
        sky_median = sigma_clipped_stats(apt_values(self._asky, im), sigma=4)[1]
        return sky_median

    def estimate_entropy(self, im):
        sky_entropy = entropy(apt_values(self._asky, im))
        obj_entropy = array([entropy(apt_values(apt, im)) for apt in self._aobj])
        return obj_entropy, sky_entropy

    def photometry(self, im):
        try:
            self.calculate_centroid(im)
            self._sky_median = self.estimate_sky(im)
            self._apt_entropy[:], self._sky_entropy = self.estimate_entropy(im)
            self._flux[:] = [ap.do_photometry(im.reduced)[0] - self._sky_median * ap.area() for ap in self._aobj]
        except ValueError:
            self._centroid[:] = nan
            self._sky_median = nan
            self._sky_entropy = nan
            self._apt_entropy[:] = nan
            self._flux[:] = nan

        return self.flux, self._sky_median, self.apt_entropy, self._sky_entropy, self._centroid