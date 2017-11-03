import xarray as xa

from numpy import inf, sqrt, dot
import patsy
from numpy.linalg import lstsq


class PhotometryData:
    def __init__(self, fname, tid, cids, fstart=0, fend=inf):
        with xa.open_dataset(fname) as ds:
            self._ds = ds.load()
        self._flux = self._ds.flux
        self._aux = self._ds.aux

        self.nframes = self._flux.shape[0]
        self.nobj = self._flux.shape[1]
        self.napt = self._flux.shape[2]

        self.tid = tid
        self.cids = cids
        self.fstart = max(0, fstart)
        self.fend = min(self.nframes, fend)
        self.iapt = 0


    @property
    def flux(self):
        return self._flux[self.fstart:self.fend, :, :]

    @property
    def _auxr(self):
        return self._aux[self.fstart:self.fend]

    @property
    def mjd(self):
        return self._auxr.loc[:,'mjd']

    @property
    def airmass(self):
        return self._auxr.loc[:, 'airmass']

    @property
    def sky(self):
        return self._auxr.loc[:, 'sky']

    @property
    def entropy(self):
        return self._auxr.loc[:, 'entropy']

    @property
    def xshift(self):
        return self._auxr.loc[:, 'x']

    @property
    def yshift(self):
        return self._auxr.loc[:, 'y']

    @property
    def relative_flux(self):
        f = self.flux[:, self.tid, :] / self.flux[:, self.cids, :].sum('star')
        return f / f.median('frame')

    @property
    def linear_trend(self):
        dm = patsy.dmatrix('mjd + sky + x + y + entropy + airmass', self._auxr.to_pandas())
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
