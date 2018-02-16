import numpy as np

from pathlib import Path

from tqdm import tqdm
from matplotlib.pyplot import subplots, setp
from numpy import array, arange, min, max, sqrt, inf, floor

from muscat2ph.phdata import PhotometryData
from muscat2ta.lc import M2LCSet, M2LightCurve
from muscat2ta.lpf import StudentLSqLPF
from muscat2ta.gp import M2GP

class TransitAnalysis:
    pbs = 'g r i z_s'.split()

    def __init__(self, datadir, target, tid, cids, fend=inf, nbins=500):
        self.ddata = dd = Path(datadir)
        self.phs = [PhotometryData(dd.joinpath('{}-{}.nc'.format(target, pb)), tid, cids, fend=fend)
                    for pb in self.pbs]
        [ph.select_aperture() for ph in self.phs]
        self.lcs = M2LCSet([M2LightCurve(ph.jd, ph.relative_flux, ph.aux) for ph in self.phs])
        self.lcs.mask_outliers()
        self.lcs.mask_covariate_outliers()

        for ph, lc in zip(self.phs, self.lcs):
            et = float(ph._aux.loc[:, 'exptime'].mean())
            lc.downsample_nbins(nbins)
        self.lpf = StudentLSqLPF(target, self.lcs, self.pbs)

        self.de_minimum = None
        self.gpfs = None

    def optimize_global(self, niter=100):
        self.lpf.optimize_global(niter)
        self.de_minimum = self.lpf.de.minimum_location

    def mask_flux_outliers(self, sigma=4):
        assert self.de_minimum is not None, "Need to run global optimization before calling outlier masking"
        self.lpf.mask_outliers(sigma=sigma,
                               means=self.lpf.compute_lc_model(self.de_minimum))

    def learn_gp_hyperparameters(self, max_pts=600, kernelf=None, kernelfarg=(), covariates=(2,3,4,5,6)):
        transits = self.lpf.compute_transit(self.de_minimum)
        self.gpfs = [M2GP(self.lpf, transits, pb, max_pts=max_pts, kernelf=kernelf, kernelfarg=kernelfarg, covariates=covariates) for pb in range(4)]
        for gpf in tqdm(self.gpfs):
            gpf.optimize_hps()
        self.gp_predictions = [gpf.predict() for gpf in self.gpfs]

    def plot_global_optimization(self):
        models = [self.lpf.compute_lc_model(pv) for pv in self.lpf.de.population]
        fig, axs = subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
        for i, (ax, time, flux) in enumerate(zip(axs.flat, self.lpf.times, self.lpf.fluxes)):
            ax.plot(time, flux, '.', alpha=0.5)
            for model in models:
                ax.plot(time, model[i], 'k', alpha=0.1)
        [setp(ax, title=flt) for flt, ax in zip('g r i z'.split(), axs.flat)];
        setp(axs[0, :], xlabel='')
        setp(axs[:, 1], ylabel='')
        setp(axs[:, 0], ylabel='Normalized flux')
        fig.tight_layout()

    def plot_light_curve(self, detrend_obs=False, detrend_mod=True, include_mod=True, figsize=(11, 6)):
        lpf = self.lpf
        if detrend_obs:
            lc_baseline = lpf.compute_baseline(lpf.de.minimum_location)
            fluxes_obs = [lpf.fluxes[i] / lc_baseline[i] for i in range(4)]
        else:
            fluxes_obs = lpf.fluxes

        if detrend_mod:
            fluxes_mod = lpf.compute_transit(lpf.de.minimum_location)
        else:
            fluxes_mod = lpf.compute_lc_model(lpf.de.minimum_location)

        fig, axs = subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
        for i in range(4):
            axs.flat[i].plot(lpf.times[i], fluxes_obs[i], 'k.')
            if include_mod:
                omin = fluxes_obs[i].min()
                rmax = (fluxes_obs[i] - fluxes_mod[i]).max()
                shift = omin - rmax
                axs.flat[i].plot(lpf.times[i], fluxes_mod[i], 'w-', lw=3)
                axs.flat[i].plot(lpf.times[i], fluxes_mod[i], '-', lw=1)
                axs.flat[i].plot(lpf.times[i], fluxes_obs[i] - fluxes_mod[i] + shift, 'k.')
        fig.tight_layout()

    def plot_noise(self, xtype='time'):
        binf = arange(1, 50)
        residuals = [gpf.residuals for gpf in self.gpfs]
        prediction = self.gp_predictions

        fig, axs = subplots(4, 2, figsize=(11, 6), sharey='row', sharex='col',
                            gridspec_kw={'height_ratios': [0.7, 0.3, 0.7, 0.3]})
        ymin, ymax = inf, -inf
        for i, (res, pre) in enumerate(zip(residuals, prediction)):
            nbins = floor(res.size / binf)
            totp = (binf * nbins).astype(np.int)
            s0 = 1e6 * array([res[:tp].reshape([-1, nb]).mean(1).std() for nb, tp in zip(binf, totp)])
            s1 = 1e6 * array([(res - pre)[:tp].reshape([-1, nb]).mean(1).std() for nb, tp in zip(binf, totp)])
            ymin = min(ymin, min(s0.min(), s1.min()))
            ymax = max(ymax, max(s0.max(), s1.max()))
            x = binf * 15 / 60 if xtype == 'time' else sqrt(binf)
            axs[::2].flat[i].plot(x, s0)
            axs[::2].flat[i].plot(x, s1)
            axs[1::2].flat[i].plot(x, s0 - s1)
        xlabel = 'Integration time [min]' if xtype == 'time' else 'sqrt(nbins)'
        setp(axs[0::2, 0], ylabel='STD [ppm]', ylim=(0.95 * ymin, 1.05 * ymax))
        setp(axs[1::2, 0], ylabel='$\Delta$ STD [ppm]')
        setp(axs[-1, :], xlabel=xlabel)
        fig.tight_layout()

