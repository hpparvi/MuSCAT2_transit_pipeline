import xarray as xa
from matplotlib.mlab import normpdf

from numpy import inf, sqrt, dot, exp, linspace, log, zeros, array, arange, meshgrid
import patsy
from numpy.linalg import lstsq
from photutils import CircularAperture
from scipy.interpolate import interp1d
from astropy.time import Time
from astropy import coordinates as coord, units as u

lapalma = coord.EarthLocation.from_geodetic(-17.8799*u.deg, 28.758*u.deg, 2327*u.m)

class PhotometryData:
    def __init__(self, fname, tid, cids, fstart=0, fend=inf):
        with xa.open_dataset(fname) as ds:
            self._ds = ds.load()
        self._flux = self._ds.flux
        self._aux = self._ds.aux
        self._cnt = self._ds.centroid
        self._sky = self._ds.sky_median

        self.nframes = self._flux.shape[0]
        self.nobj = self._flux.shape[1]
        self.napt = self._flux.shape[2]

        self.tid = tid
        self.cids = cids
        self.fstart = max(0, fstart)
        self.fend = min(self.nframes, fend)
        self.iapt = -1
        self.apt = self._flux.aperture[self.iapt]
        self.lin_formula = 'mjd + sky + xshift + yshift + entropy + airmass'
        self._entropy_table = None
        self._calculate_effective_fwhm()


    def _calculate_effective_fwhm(self):
        """Calculates the table to convert between entropy and effective PSF FWHM"""
        def entropy(a):
            a = a - a.min() + 1e-10
            a = a / a.sum()
            return -(a * log(a)).sum()

        fwhms = exp(linspace(log(1), log(50), 100))
        self._entropy_table = xa.DataArray(zeros((3, 100)), name='entropy', dims='aperture fwhm'.split(),
                                           coords={'aperture': self._ds.aperture, 'fwhm': fwhms})

        for i, ar in enumerate(array(self._ds.aperture)):
            apt = CircularAperture((0, 0), ar)
            msk = apt.to_mask()[0]
            mb = msk.array.astype('bool')
            x, y = meshgrid(arange(mb.shape[0]), arange(mb.shape[1]))
            r = sqrt((x - ar)**2 + (y - ar)**2)
            sigmas = fwhms / 2.355
            self._entropy_table[i, :] = [entropy(normpdf(r, 0, sigma)[mb]) for sigma in sigmas]

        self._fwhm = self._ds.obj_entropy.copy()
        self._fwhm.name = 'fwhm'

        for aperture in self._fwhm.aperture:
            ip = interp1d(self._entropy_table.loc[aperture, :], self._entropy_table.fwhm, bounds_error=False,
                          fill_value=tuple(self._entropy_table.loc[aperture][[0, -1]]))
            for star in self._fwhm.star:
                self._fwhm.loc[:, star, aperture] = ip(self._ds.obj_entropy.loc[:, star, aperture])

    @property
    def aux(self):
        return xa.DataArray(list(map(array, [self.mjd.value, self.sky, self.airmass, self.xshift, self.yshift, self.entropy])),
                     name='aux', dims='quantity frame'.split(),
                     coords={'frame': self._ds.frame[self.fstart:self.fend],
                             'quantity': 'mjd sky airmass xshift yshift entropy'.split()}).T

    @property
    def flux(self):
        return self._flux[self.fstart:self.fend, :, :]

    @property
    def _auxr(self):
        return self._aux[self.fstart:self.fend]

    @property
    def mjd(self):
        return Time(self._auxr.loc[:,'mjd'], format='mjd', scale='utc', location=lapalma)

    @property
    def jd(self):
        return self.mjd.jd

    @property
    def bjd(self):
        return self.mjd.jd

    @property
    def airmass(self):
        return self._auxr.loc[:, 'airmass']

    @property
    def sky(self):
        return self._sky[self.fstart:self.fend,:].mean('star')

    @property
    def entropy(self):
        return self._ds.obj_entropy[self.fstart:self.fend,:,self.iapt].mean('star')

    @property
    def effective_fwhm(self):
        return self._fwhm[self.fstart:self.fend,:,self.iapt].mean('star')

    @property
    def xshift(self):
        return (self._cnt[self.fstart:self.fend,:,0] - self._cnt[self.fstart,:,0]).mean('star')

    @property
    def yshift(self):
        return (self._cnt[self.fstart:self.fend,:,1] - self._cnt[self.fstart,:,1]).mean('star')

    @property
    def relative_flux(self):
        f = self.flux[:, self.tid, :] / self.flux[:, self.cids, :].sum('star')
        return f / f.median('frame')

    @property
    def linear_trend(self):
        dm = patsy.dmatrix(self.lin_formula, self.aux.to_pandas())
        a,_,_,_ = lstsq(dm, self.relative_flux)
        return dot(dm, a)

    @property
    def detrended_relative_flux(self):
        f = self.relative_flux / self.linear_trend
        return f / f.median('frame')

    @property
    def normalized_std_per_star(self):
        return (self.flux / self.flux.median('frame')).std('frame')

    @property
    def normalized_ptps_per_star(self):
        return (self.flux / self.flux.median('frame')).diff('frame').std('frame') / sqrt(2)

    @property
    def normalized_relative_flux_ptps(self):
        return self.relative_flux.diff('frame').std('frame') / sqrt(2)

    def plot_single(self, iobj, iapt, plot_excluded=False):
        self.flux[:, iobj, iapt].plot(marker='.', linestyle='')
        if plot_excluded:
            self._flux[:, iobj, iapt].plot(alpha=0.25, c='k', marker='.', linestyle='')
