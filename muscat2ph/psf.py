from collections import namedtuple

from numpy import sqrt, argmax, zeros, meshgrid, arange, exp, clip, median, inf
from scipy.stats import scoreatpercentile as sap
from scipy.ndimage import center_of_mass as cmass, label, median_filter as mf
from scipy.optimize import minimize

qnorm = 1/sqrt(2)
rt = namedtuple('rt', 'a cx cy fwhm radius sky xl yl')

def centroid_m(d):
    mask = (mf(d, 4) > sap(d, 95)) & (mf(d, 4) < sap(d, 99))
    labels, nl = label(mask)
    d = d - sap(d, 92)
    brightest_object = argmax([d[labels==i].mean() for i in range(nl+1)])
    return cmass(d, labels, index=brightest_object)

class DefocusedPSF:
    def __init__(self, shape):
        self.shape = shape
        self.psf = zeros(self.shape)
        self.sx, self.sy = self.shape[1], self.shape[0]
        self.X, self.Y = meshgrid(arange(shape[1]), arange(shape[0]))

    def __call__(self, **kwargs):
        raise NotImplementedError

    def chi2(self, pv, frame):
        return ((frame - self(*pv))**2).sum()

    def fit(self, frame):
        cx, cy = centroid_m(frame)
        sky = median(frame)
        amp = sap(frame-sky, 99)
        bounds = [[0.8*amp, 1.2*amp],
                  [0.2*self.sx, 0.8*self.sx], [0.2*self.sy, 0.8*self.sy],
                  [0.5, 20], [1, 25],
                  [0.8*sky, 1.2*sky],
                  [-10,10], [-10,10]]

        self.pv0 = pv0 = [amp, cx, cy, 7, 6, sky, 0, 0]
        self.fit_result = minimize(self.chi2, pv0, (frame, ), bounds=bounds)
        return rt(*self.fit_result.x)

    def plot(self):
        import matplotlib.pyplot as pl
        pl.imshow(self(*self.fit_result.x))

class GaussianDFPSF(DefocusedPSF):
    def __call__(self, a, xc, yc, fwhm, radius, sky, xl, yl):
        r = sqrt((self.X-xc)**2 + (self.Y-yc)**2)
        torus = a * exp(-0.5 * ((r-radius)/(fwhm/2.355))**2)
        xterm = clip((1+(self.X-xc)*xl), 0, inf)
        yterm = clip((1+(self.Y-yc)*yl), 0, inf)
        self.psf[:,:] = sky + torus * xterm * yterm
        return self.psf

class QuarticDFPSF(DefocusedPSF):

    def __call__(self, a, xc, yc, width, radius, xl=0., yl=0.):
        r = sqrt((self.X-xc)**2 + (self.Y-yc)**2)
        xp = qnorm * (r-radius) / width
        mask = xp < qnorm
        self.psf[mask] = a + a * 4. * (xp[mask]**4 - xp[mask]**2)
        xterm = clip((1+(self.X-xc)*xl), 0, inf)
        yterm = clip((1+(self.Y-yc)*yl), 0, inf)
        self.psf *= xterm * yterm
        return self.psf
