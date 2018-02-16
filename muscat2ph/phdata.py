import xarray as xa
from matplotlib.mlab import normpdf

from numpy import inf, sqrt, dot, exp, linspace, log, zeros, array, arange, meshgrid, ones, r_
import patsy
from numpy.linalg import lstsq
from photutils import CircularAperture
from scipy.interpolate import interp1d
from astropy.time import Time
from astropy import coordinates as coord, units as u
from astropy.stats import mad_std

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

        self._fmask = ones(self.nframes, 'bool')
        self._fmask[:self.fstart] = 0
        self._fmask[self.fend:] = 0
        self._fmask &= self._flux.notnull().any(['star','aperture'])
        self._fmask = array(self._fmask)

        self._entropy_table = None
        self._calculate_effective_fwhm()


    def _calculate_effective_fwhm(self):
        """Calculates the table to convert between entropy and effective PSF FWHM"""
        def entropy(a):
            a = a - a.min() + 1e-10
            a = a / a.sum()
            return -(a * log(a)).sum()

        fwhms = exp(linspace(log(1), log(50), 100))
        self._entropy_table = xa.DataArray(zeros((self.napt, 100)), name='entropy', dims='aperture fwhm'.split(),
                                           coords={'aperture': self._ds.aperture, 'fwhm': fwhms})

        for i, ar in enumerate(array(self._ds.aperture)):
            apt = CircularAperture((0, 0), ar)
            msk = apt.to_mask()[0]
            mb = msk.data.astype('bool')
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
                m = self._fwhm.loc[:, star, aperture].notnull()
                self._fwhm.loc[m, star, aperture] = ip(self._ds.obj_entropy.loc[m, star, aperture])


    def select_aperture(self):
        self.iapt = int(self.normalized_relative_flux_ptps.argmin())


    @property
    def aux(self):
        return xa.DataArray(list(map(array, [self.mjd.value, self.sky, self.airmass, self.xshift, self.yshift, self.entropy])),
                     name='aux', dims='quantity mjd'.split(),
                     coords={'mjd': self._ds.mjd[self.fstart:self.fend],
                             'quantity': 'mjd sky airmass xshift yshift entropy'.split()}).T

    @property
    def flux(self):
        return self._flux[self._fmask, :, :]

    @property
    def mjd(self):
        return Time(self._aux.loc[self._fmask,'mjd'], format='mjd', scale='utc', location=lapalma)

    @property
    def jd(self):
        return self.mjd.jd

    @property
    def bjd(self):
        return self.mjd.jd

    @property
    def airmass(self):
        return self._aux.loc[self._fmask, 'airmass']

    @property
    def sky(self):
        return self._sky[self._fmask,r_[self.tid, self.cids]].mean('star')

    @property
    def entropy(self):
        return self._ds.obj_entropy[self._fmask, r_[self.tid, self.cids], self.iapt].mean('star')

    @property
    def effective_fwhm(self):
        return self._fwhm[self._fmask, r_[self.tid, self.cids], self.iapt].mean('star')

    @property
    def xshift(self):
        return (self._cnt[self._fmask, r_[self.tid, self.cids], 0] - self._cnt[self.fstart,r_[self.tid, self.cids],0]).mean('star')

    @property
    def yshift(self):
        return (self._cnt[self._fmask, r_[self.tid, self.cids], 1] - self._cnt[self.fstart,r_[self.tid, self.cids],1]).mean('star')

    @property
    def relative_flux(self):
        f = self.flux[:, self.tid, self.iapt] / self.flux[:, self.cids, self.iapt].sum('star')
        return f / f.median('mjd')

    @property
    def relative_fluxes(self):
        f = self.flux[:, self.tid, :] / self.flux[:, self.cids, :].sum('star')
        return f / f.median('mjd')

    @property
    def linear_trend(self):
        dm = patsy.dmatrix(self.lin_formula, self.aux.to_pandas())
        a,_,_,_ = lstsq(dm, self.relative_flux)
        return dot(dm, a)

    @property
    def detrended_relative_flux(self):
        f = self.relative_flux / self.linear_trend
        return f / f.median('mjd')

    @property
    def normalized_std_per_star(self):
        return (self.flux / self.flux.median('mjd')).std('mjd')

    @property
    def normalized_ptps_per_star(self):
        return (self.flux / self.flux.median('mjd')).diff('mjd').std('mjd') / sqrt(2)

    @property
    def normalized_relative_flux_ptps(self):
        return xa.DataArray(mad_std(self.relative_fluxes.diff('mjd'), 0) / sqrt(2),
                            dims='aperture', coords={'aperture': self._flux.aperture})

    def plot_single(self, iobj, iapt, plot_excluded=False):
        self.flux[:, iobj, iapt].plot(marker='.', linestyle='')
        if plot_excluded:
            self._flux[:, iobj, iapt].plot(alpha=0.25, c='k', marker='.', linestyle='')
