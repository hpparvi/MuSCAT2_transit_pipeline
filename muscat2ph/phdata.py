import xarray as xa
import seaborn as sb
from astropy.coordinates import SkyCoord, FK5
from astroquery.simbad import Simbad
from matplotlib.pyplot import subplots, setp

from numpy import inf, sqrt, dot, exp, linspace, log, zeros, array, arange, meshgrid, ones, r_, isin, ceil, ones_like, \
    newaxis, c_, all, percentile, isnan
import patsy
from numpy.linalg import lstsq
from numpy.polynomial.legendre import legvander
from photutils import CircularAperture
from scipy.interpolate import interp1d
from scipy.stats import norm
from astropy.time import Time
from astropy import coordinates as coord, units as u
from astropy.stats import mad_std
from tqdm import tqdm

from muscat2ta.lc import M2LightCurve

lapalma = coord.EarthLocation.from_geodetic(-17.8799*u.deg, 28.758*u.deg, 2327*u.m)

N = lambda a: a/a.median('mjd')


class ReferenceStarSet:
    def __init__(self, ph):
        self.ph = ph
        self.tid = ph.tid
        self.sc = None
        self.tap = None
        self.cids = []
        self.caps = []
        self.ptps = []

    def select_best(self, n_stars=1, start_id=0, end_id=5, start_apt=0, end_apt=None, model=None):
        N = lambda a: a / a.median('mjd')

        self.tap, self.cids, self.caps, self.ptps = None, [], [], []
        cids = arange(start_id, end_id)
        apts = arange(start_apt, end_apt or self.ph.napt)
        for i in tqdm(range(n_stars), desc='Optimising comparison stars'):
            cids = cids[isin(cids, self.cids + [self.tid], invert=True)]
            sc = xa.DataArray(zeros([cids.size, apts.size, apts.size]),
                              dims='cstar tapt capt'.split(),
                              coords={'cstar': cids, 'tapt': apts, 'capt': apts})

            for icid, cid in enumerate(cids):
                for iaid, aid in enumerate(apts):
                    ft = self.ph.flux[:, self.tid, aid]
                    if model is not None:
                        ft = ft / model
                    fc = self.reference_flux + self.ph.flux.loc[:, cid][:, apts]
                    sc[icid, iaid, :] = (self.ptp_scatter(ft / fc, 1) / sqrt(2)
                                         + self.ptp_scatter(ft / fc, 2) / sqrt(6)
                                         + self.ptp_scatter(ft / fc, 3) / sqrt(20))

            s = sc.where(sc == sc.min(), drop=True)
            self.sc = sc
            if self.tap is None:
                self.tap = int(s.tapt)
            self.cids.append(int(s.cstar))
            self.caps.append(int(s.capt))
            self.ptps.append(float(s))
        return s

    @staticmethod
    def ptp_scatter(flux, n=1):
        return array((flux / flux.median('mjd')).diff('mjd', n=n).std('mjd') * sqrt(2))

    @staticmethod
    def std(flux):
        return array((flux / flux.median('mjd')).std('mjd'))

    @property
    def reference_flux(self):
        f = 0.
        for cid, aid in zip(self.cids, self.caps):
            f += self.ph.flux.loc[:, cid][:, aid]
        return f

    @property
    def normalized_relative_flux(self):
        return N(self.ph.flux[:, self.tid, self.tap] / self.reference_flux)


class PhotometryData:
    def __init__(self, fname, tid, cids, objname=None, objskycoords=None, excluded_ranges=None, **kwargs):
        with xa.open_dataset(fname) as ds:
            self._ds = ds.load()
        self._flux = self._ds.flux
        self._mjd = Time(self._ds.aux.loc[:,'mjd'], format='mjd', scale='utc', location=lapalma)
        self.objname = objname
        self.objskycoords = objskycoords
        self._excluded_ranges = excluded_ranges

        self._aux = self._ds.aux
        self._cnt = self._ds.centroid
        self._sky = self._ds.sky_median

        self.nframes = self._flux.shape[0]
        self.nobj = self._flux.shape[1]
        self.napt = self._flux.shape[2]

        self.tid = tid
        self.cids = cids
        self.mjd_start = kwargs.get('mjd_start', None) or kwargs.get('tstart', -inf)
        self.mjd_end = kwargs.get('mjd_end', None) or kwargs.get('tend', inf)
        self.iapt = -1
        self.apt = self._flux.aperture[self.iapt]
        self.lin_formula = 'mjd + sky + xshift + yshift + entropy + airmass'

        try:
            self.centroids_pix = cpix = self._ds.centroids_pix.values
            self.distances_pix = sqrt(((cpix - cpix[tid])**2).sum(1))
        except AttributeError:
            self.centroids_pix =  None
            self.distances_pix =  None

        try:
            self.centroids_sky = SkyCoord(array(self._ds.centroids_sky), frame=FK5, unit=(u.deg, u.deg))
            self.distances_arcmin = self.centroids_sky[tid].separation(self.centroids_sky).arcmin
            if not self.objskycoords:
                self.objskycoords = self.centroids_sky[tid]
        except:
            self.centroids_sky = None
            self.distances_arcmin = None

        if self.distances_pix is not None and (self.centroids_sky is None or all(isnan(self.distances_arcmin))):
            self.distances_arcmin = self.distances_pix * (0.44 * u.arcsec).to(u.arcmin).value

        if not self.objskycoords:
            try:
                obj = Simbad.query_object(self.objname)
                self.objskycoords = SkyCoord(obj['RA'][0], obj['DEC'][0], frame=FK5, unit=(u.hourangle, u.deg))
            except:
                print('Could not set the target sky coordinates')

        self._fmask = ones(self.nframes, 'bool')
        self._fmask[self._mjd.value < self.mjd_start] = 0
        self._fmask[self._mjd.value > self.mjd_end] = 0
        self._fmask &= self._flux.notnull().any(['star','aperture'])
        self._fmask = array(self._fmask)

        if self._excluded_ranges is not None:
            self.exclude_mjd_ranges(self._excluded_ranges)

        self._entropy_table = None
        self._calculate_effective_fwhm()
        self._rset = ReferenceStarSet(self)
        self._rset.tid = tid
        self._rset.tap = 2
        self._rset.cids = cids
        self._rset.caps = len(cids)*[2]


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
            msk = apt.to_mask()
            mb = msk.data.astype('bool')
            x, y = meshgrid(arange(mb.shape[0]), arange(mb.shape[1]))
            r = sqrt((x - ar)**2 + (y - ar)**2)
            sigmas = fwhms / 2.355
            self._entropy_table[i, :] = [entropy(norm.pdf(r, 0, sigma)[mb]) for sigma in sigmas]

        self._fwhm = self._ds.obj_entropy.copy()
        self._fwhm.name = 'fwhm'

        for aperture in self._fwhm.aperture:
            ip = interp1d(self._entropy_table.loc[aperture, :], self._entropy_table.fwhm, bounds_error=False,
                          fill_value=tuple(self._entropy_table.loc[aperture][[0, -1]]))
            for star in self._fwhm.star:
                m = self._fwhm.mjd[self._fwhm.loc[:, star, aperture].notnull()]
                self._fwhm.loc[m, star, aperture] = ip(self._ds.obj_entropy.loc[m, star, aperture])

    def select_aperture(self):
        self.iapt = int(self.normalized_relative_flux_ptps.argmin())

    def exclude_mjd_ranges(self, mjd_ranges: tuple) -> None:
        mask = ones(self.nframes, bool)
        for mr in mjd_ranges:
            mask &= ~((self._mjd.value > mr[0]) & (self._mjd.value < mr[1]))
        self._fmask &= mask

    @property
    def aux(self):
        return xa.DataArray(list(map(array, [self.mjd.value, self.sky, self.airmass, self.xshift, self.yshift, self.entropy])),
                     name='aux', dims='quantity mjd'.split(),
                     coords={'mjd': self.mjd.value,
                             'quantity': 'mjd sky airmass xshift yshift entropy'.split()}).T

    @property
    def flux(self):
        return self._flux[self._fmask, :, :] / array(self._aux.loc[self._fmask, 'exptime'])[:, newaxis, newaxis]

    @property
    def mjd(self):
        return self._mjd[self._fmask]

    @property
    def jd(self):
        return self.mjd.jd

    @property
    def bjd(self):
        bjd = self.mjd.tdb + self.mjd.light_travel_time(self.objskycoords)
        return bjd.jd

    @property
    def airmass(self):
        return self._aux.loc[self._fmask, 'airmass']

    @property
    def sky(self):
        return self._sky[self._fmask, r_[self.tid, self.cids].astype(int)].mean('star') / array(self._aux.loc[self._fmask, 'exptime'])

    @property
    def entropy(self):
        return self._ds.obj_entropy[self._fmask, r_[self.tid, self.cids].astype(int), self.iapt].mean('star')

    @property
    def effective_fwhm(self):
        sids = r_[self.tid, self.cids].astype(int)
        return self._fwhm[self._fmask, sids, self.iapt].mean('star')

    @property
    def xshift(self):
        sids = r_[self.tid, self.cids].astype(int)
        return (self._cnt[self._fmask, sids, 0] - self._cnt[self._fmask, sids, 0][0]).mean('star')

    @property
    def yshift(self):
        sids = r_[self.tid, self.cids].astype(int)
        return (self._cnt[self._fmask, sids, 1] - self._cnt[self._fmask, sids,1][0]).mean('star')

    @property
    def relative_flux(self):
        """Normalized relative flux"""
        return self._rset.normalized_relative_flux

    @property
    def relative_error(self):
        """Normalized relative uncertainty"""
        return sqrt(abs(1/self.flux[:, self._rset.tid, self._rset.tap]) + abs(1/self._rset.reference_flux))

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

    def transit_coverage(self, zero_epoch: tuple, period: tuple, duration: tuple, nsamples: int = 2500):
        de = norm(*zero_epoch)
        dp = norm(*period)
        dd = norm(*duration)
        t1, t2 = self.bjd[[0, -1]]
        tn = int(round((self.bjd.mean() - zero_epoch[0]) / period[0]))
        centers = de.rvs(nsamples) + tn * dp.rvs(nsamples)
        durations = dd.rvs(nsamples)
        transit_start = centers - 0.5 * durations
        transit_end = centers + 0.5 * durations
        p_start = ((t1 < transit_start) & (transit_start < t2)).mean()
        p_end = ((t1 < transit_end) & (transit_end < t2)).mean()
        p_full = ((t1 < transit_start) & (transit_end < t2)).mean()
        p_all  = ((t1 > transit_start) & (transit_end > t2)).mean()
        p_miss = ((t1 > transit_end) | (transit_start > t2)).mean()
        return p_full, p_miss, p_all, p_start, p_end

    def event_limits(self, zero_epoch: tuple, period: tuple, duration: tuple, event='center', nsamples: int = 2500):
        events = 'center', 'start', 'end'
        assert event in events
        de = norm(*zero_epoch)
        dp = norm(*period)
        dd = norm(*duration)
        t1, t2 = self.bjd[[0, -1]]
        tn = int(round((self.bjd.mean() - zero_epoch[0]) / period[0]))
        centers = de.rvs(nsamples) + tn * dp.rvs(nsamples)
        durations = dd.rvs(nsamples)
        if event == 'center':
            return percentile(centers, [50, 16, 84, 0.5, 99.5])
        elif event == 'start':
            return percentile(centers - 0.5 * durations, [50, 16, 84, 0.5, 99.5])
        elif event == 'end':
            return percentile(centers + 0.5 * durations, [50, 16, 84, 0.5, 99.5])

    def plot_single(self, iobj, iapt, plot_excluded=False):
        self.flux[:, iobj, iapt].plot(marker='.', linestyle='')
        if plot_excluded:
            self._flux[:, iobj, iapt].plot(alpha=0.25, c='k', marker='.', linestyle='')

    def plot_raw(self, nstars, ylim=(0.85, 1.15), ncols=4, figsize=(11,11)):
        apertures = self._ds.aperture.data
        date = self.mjd[0].iso.split(' ')[0]
        nrows = int(ceil(nstars / ncols))
        fig, axs = subplots(nrows, ncols, figsize=figsize, sharex='all', sharey='all')
        nflux = self.flux / self.flux.median('mjd')
        aids = abs(nflux.diff('mjd')).median('mjd').argmin('aperture')
        for i,(ax,iapt) in enumerate(zip(axs.flat, aids[:nstars])):
            apt = apertures[iapt]
            ax.plot(self.mjd.value, nflux[:, i, iapt], 'k')
            ax.text(0.05, 0.99, f'star id={i}\naperture id = {int(iapt)}\nr={int(apt)} pix', va='top', transform=ax.transAxes)
        setp(axs, ylim=ylim)
        sb.despine(fig)
        fig.tight_layout()
        return fig

    def plot_relative_set(self, targets: list, references: list, maxr: int = 3, figsize: tuple = (11, 11)) -> None:
        nt = len(targets)
        fig, axs = subplots(nt, maxr + 1, figsize=figsize, sharex=True, gridspec_kw={'hspace': 0.01, 'wspace': 0})
        x = linspace(-1, 1, self.mjd.size)
        covs = c_[legvander(x, 1), array([self.xshift, self.yshift, self.entropy, self.sky]).T]
        cnames = 'c0 c1 xshift yshift entr sky'.split()

        for i, tid in enumerate(targets):
            s = set(references)
            s.discard(tid)
            s = list(s)[:maxr]
            for j, cid in enumerate(s):
                ax = axs[i, j]
                flux = N(self.flux[:, tid, 3] / self.flux[:, cid, 3])
                lc = M2LightCurve(0, self.bjd, flux, ones_like(flux), covs, cnames)
                c, _, _, _ = lstsq(lc.covariates, lc.flux, rcond=None)
                fsys = dot(lc.covariates, c)
                lc._flux = 1 + lc._flux - fsys
                lc.downsample_time(300)
                ax.errorbar(lc.btime, lc.bflux, lc.bwn, drawstyle='steps-mid')
                ax.text(0.05, 0.95, f"star {tid} / star {cid}", ha='left', va='top', transform=ax.transAxes)

            flux = N(self.flux[:, tid, 3] / self.flux[:, s, 3].mean('star'))
            lc = M2LightCurve(0, self.bjd, flux, ones_like(flux), covs, cnames)
            c, _, _, _ = lstsq(lc.covariates, lc.flux, rcond=None)
            fsys = dot(lc.covariates, c)
            lc._flux = 1 + lc._flux - fsys
            lc.downsample_time(300)
            axs[i, maxr].errorbar(lc.btime, lc.bflux, lc.bwn, drawstyle='steps-mid', c='k')
        setp(axs[:, 1:], yticks=[])
        fig.tight_layout()
        return fig
