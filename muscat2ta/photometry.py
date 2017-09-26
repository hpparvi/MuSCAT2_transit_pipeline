import pandas as pd
from numpy import *
import numpy as np
from astropy.io import fits as pf
import astropy.visualization as av
from glob import glob
from tqdm import tqdm
import seaborn as sb

from photutils import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.centroids import centroid_com, centroid_2dg

passbands = 'g r i z_s'.split()

from scipy.ndimage import label, labeled_comprehension, median_filter as mf
from scipy.stats import scoreatpercentile as sap

class MasterFrame:
    def __init__(self, files, passbands):
        self.passbands = passbands
        self._data = {pb: None for pb in passbands}

    def __call__(self, pb):
        assert pb in self.passbands
        return self._data[pb]

    @property
    def g(self):
        return self('g')

    @property
    def r(self):
        return self('r')

    @property
    def i(self):
        return self('i')

    @property
    def z(self):
        return self('z')


class MasterDark(MasterFrame):
    def __init__(self, files, passbands):
        super().__init__(files, passbands)
        qstr = '(ccd=="{}") & (object=="DARK")'
        self.files = {pb: list(files.query(qstr.format(pb))['file'])
                      for pb in passbands}

        for pb in passbands:
            for fn in tqdm(self.files[pb]):
                with pf.open(fn) as f:
                    et = float(f[0].header['exptime'])
                    d = f[0].data.astype('d')  # / et
                    self._data[pb] = d if self._data[pb] is None else self._data[pb] + d
            self._data[pb] /= len(self.files[pb])


class MasterFlat(MasterFrame):
    def __init__(self, files, dark, passbands):
        super().__init__(files, passbands)
        self.dark = dark
        qstr = '(ccd=="{}") & (object=="DomeFlat")'
        self.files = {pb: list(files.query(qstr.format(pb))['file'])
                      for pb in passbands}

        for pb in passbands:
            for fn in tqdm(self.files[pb]):
                with pf.open(fn) as f:
                    d = f[0].data.astype('d') - self.dark(pb)
                    self._data[pb] = d if self._data[pb] is None else self._data[pb] + d
            self._data[pb] /= median(self._data[pb])


class PhotometryFrame:
    def __init__(self, name, xlims, ylims, apertures, sradius):
        self.name = name
        self.slice = s_[ylims[0]:ylims[1], xlims[0]:xlims[1]]
        self.apertures = np.asarray(apertures)
        self.sradius = sradius
        self.napt = self.apertures.size

    def __call__(self, data, plot=False):
        frame = data[self.slice]
        cnt = self.centroid(frame)
        flux = zeros(self.napt)
        for i,aperture in enumerate(self.apertures):
            aobj = CircularAperture(cnt, aperture)
            asky = CircularAnnulus(cnt, aperture, self.sradius)
            flux[i] = aobj.do_photometry(frame)[0] - asky.do_photometry(frame)[0] / asky.area() * aobj.area()
        return flux

    def centroid(self, d):
        dm = mf(d, 5) > sap(d, 80)
        labels, nl = label(dm)
        l = argmax(labeled_comprehension(d, labels, arange(1, nl + 1), sum, np.float64, -1)) + 1
        return centroid_com(d, ~(labels == l))