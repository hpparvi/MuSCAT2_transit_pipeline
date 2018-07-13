import numpy as np
import pandas as pd
import xarray as xa

from pathlib import Path
from time import strftime

from tqdm import tqdm
from matplotlib.pyplot import subplots, setp
from numpy import array, arange, min, max, sqrt, inf, floor, diff, percentile, median, isin, zeros
from numpy.random import permutation
from astropy.stats import mad_std
from astropy.table import Table
from astropy.io import fits as pf

from muscat2ph.phdata import PhotometryData
from muscat2ta.lc import M2LCSet, M2LightCurve
from muscat2ta.lpf import StudentLSqLPF, GPLPF

class TransitAnalysis:
    pbs = 'g r i z_s'.split()

    def __init__(self, datadir, target, date, tid, cids, fends=(inf, inf, inf, inf), etime=30.,
                 free_k=False, contamination=False, npop=100, teff=None, **kwargs):
        self.ddata = dd = Path(datadir)
        self.target = target
        self.date = date
        self.tid = tid
        self.cids = cids
        self.free_k = free_k
        self.contamination = contamination
        self.teff = teff
        self.npop = npop
        self.etime = etime

        self.phs = [PhotometryData(dd.joinpath('{}_{}_{}.nc'.format(target, date, pb)), tid, cids, fend=fend, objname=target,
                                   **kwargs)
                    for pb,fend in zip(self.pbs, fends)]
        self._init_lcs()

        self.lm_pv = None
        self.gp_pv = None


    def _init_lcs(self):
        self.lcs = M2LCSet([M2LightCurve(ph.bjd, ph.relative_flux, ph.relative_error, ph.aux) for ph in self.phs])
        self.lcs.mask_outliers()
        self.lcs.mask_covariate_outliers()

        for ph, lc in zip(self.phs, self.lcs):
            et = float(ph._aux.loc[:, 'exptime'].mean())
            lc.downsample_time(self.etime)

        try:
            self.lmlpf = StudentLSqLPF(self.target, self.lcs, self.pbs, free_k=self.free_k, contamination=self.contamination, teff=self.teff)
            self.gplpf = GPLPF(self.target, self.lcs, self.pbs, free_k=self.free_k, contamination=self.contamination, teff=self.teff)
        except ValueError:
            print("Couldn't initialise the LPFs")


    def print_ptp_scatter(self):
        r1s = [res.std() for res in self.gplpf.residuals(self.gp_pv)]
        r2s = [(res - pre).std() for res, pre in zip(self.gplpf.residuals(self.gp_pv), self.gplpf.predict(self.gp_pv))]
        for r1, r2, pb in zip(r1s, r2s, 'g r i z'.split()):
            print('{} {:5.0f} ppm -> {:5.0f} ppm'.format(pb, 1e6 * r1, 1e6 * r2))


    def detrend_polynomial(self, npol=20):
        time, fcorr, fsys, fbase = [], [], [], []
        for lc in self.lcs:
            tm, fc, fs, fb = lc.detrend_poly(npol)
            time.append(tm)
            fcorr.append(fc)
            fsys.append(fs)
            fbase.append(fb)
        return time, fcorr, fsys, fbase


    def optimize_linear_model(self, niter=100):
        self.lmlpf.optimize_global(niter, self.npop, label='Optimizing linear model')
        self.lm_pv = self.lmlpf.de.minimum_location

    def optimize_gp_model(self, niter=100):
        self.gplpf.optimize_global(niter, self.npop, label='Optimizing GP model')
        self.gp_pv = self.gplpf.de.minimum_location

    def sample_gp_model(self, niter=100, thin=5):
        self.gplpf.sample_mcmc(niter, thin=thin, label='Sampling GP model')

    def mask_flux_outliers(self, sigma=4):
        assert self.lm_pv is not None, "Need to run global optimization before calling outlier masking"
        self.lmlpf.mask_outliers(sigma=sigma, means=self.lmlpf.flux_model(self.lm_pv))


    def learn_gp_hyperparameters(self, joint_fit=True, method='L-BFGS-B'):
        assert self.lm_pv is not None, 'Must carry out linear model optimisation before GP hyperparameter optimisation'
        if joint_fit:
            self.gplpf.optimize_hps_jointly(self.lm_pv, method=method)
        else:
            self.gplpf.optimize_hps(self.lm_pv, method=method)

    def posterior_samples(self, nsteps=0, model='gp', include_ldc=False):
        assert model in ('linear', 'gp')
        lpf = self.lmlpf if model == 'linear' else self.gplpf
        ldstart = lpf._slld[0].start
        fc = lpf.sampler.chain[:,-nsteps:,:].reshape([-1, lpf.de.n_par])
        if include_ldc:
            return pd.DataFrame(fc, columns=lpf.ps.names)
        else:
            return pd.DataFrame(fc[:,:ldstart], columns=lpf.ps.names[:ldstart])

    def plot_mcmc_chains(self, pid=0, model='gp', alpha=0.1):
        assert model in ('linear', 'gp')
        lpf = self.lmlpf if model == 'linear' else self.gplpf
        fig, ax = subplots()
        ax.plot(lpf.sampler.chain[:,:,pid].T, 'k', alpha=alpha)
        fig.tight_layout()
        return fig

    def plot_light_curve(self, model='linear', method='de', detrend_obs=False, detrend_mod=True, include_mod=True,
                         figsize=(11, 6)):
        assert model in ('linear', 'gp')
        assert method in ('de', 'mcmc')

        lpf = self.lmlpf if model == 'linear' else self.gplpf
        posterior_limits = None

        if method == 'de':
            pv = lpf.de.minimum_location
        else:
            assert lpf.sampler is not None
            niter = lpf.sampler.chain.shape[1]
            fc = lpf.sampler.chain[:, niter // 2:, :].reshape([-1, lpf.de.n_par])
            fms = [lpf.flux_model(pv) for pv in permutation(fc)[:100]]
            fms = [array([fm[i] for fm in fms]) for i in range(4)]
            posterior_limits = pl = [percentile(fm, [16, 84, 0.5, 99.5], 0) for fm in fms]
            pv = median(fc, 0)

        if model == 'linear':
            baseline = lpf.baseline(pv)
            trends = lpf.trends(pv)
            fluxes_obs = [lpf.fluxes[i] / baseline[i] - trends[i] for i in range(4)] if detrend_obs else [f/b for f,b in zip(lpf.fluxes, baseline)]
            fluxes_trm = lpf.transit_model(pv)
            fluxes_mod = lpf.transit_model(pv) if detrend_mod else lpf.flux_model(pv)
        else:
            baseline = lpf.predict(pv)
            fluxes_obs = [lpf.fluxes[i] - baseline[i] for i in range(4)] if detrend_obs else lpf.fluxes
            fluxes_trm = lpf.transit_model(pv)
            fluxes_mod = lpf.flux_model(pv)

        fig, axs = subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
        for i, (ax, time) in enumerate(zip(axs.flat, lpf.times)):

            # Plot the observations
            # ---------------------
            ax.plot(time, fluxes_obs[i], 'k.', alpha=0.25)


            if include_mod:
                # Plot the residuals
                # ------------------
                omin = fluxes_obs[i].min()
                rmax = (fluxes_obs[i] - fluxes_mod[i]).max()
                shift = omin - rmax
                ax.plot(time, fluxes_obs[i] - fluxes_mod[i] + shift, 'k.', alpha=0.25)
                ax.text(0.97, 0.9, 'STD: {:3.0f} ppm'.format((fluxes_obs[i] / fluxes_trm[i]).std()*1e6),
                        ha='right', transform=ax.transAxes)

                # Plot the model
                # ---------------
                if posterior_limits:
                    ax.fill_between(time, pl[i][0], pl[i][1], facecolor='k', alpha=0.5)
                    ax.fill_between(time, pl[i][2], pl[i][3], facecolor='k', alpha=0.2)
                else:
                    ax.plot(lpf.times[i], fluxes_mod[i], 'w-', lw=3)
                    ax.plot(lpf.times[i], fluxes_mod[i], 'k-', lw=1)
            else:
                ax.text(0.97, 0.9, 'STD: {:3.0f} ppm'.format((fluxes_obs[i]).std()*1e6),
                        ha='right', transform=ax.transAxes)

        setp(axs, xlim=lpf.times[0][[0, -1]])
        fig.tight_layout()
        return fig

    def plot_gpfit(self, figsize=(11,6)):
        lpf = self.gplpf
        pv = lpf.de.minimum_location
        predictions = lpf.predict(pv, True)
        flux_m = lpf.flux_model(pv)
        fig, axs = subplots(2, 2, figsize=figsize, sharey=True, sharex=True)
        for i,(tm,fo,fm,pr) in enumerate(zip(lpf.times, lpf.fluxes, flux_m, predictions)):
            axs.flat[i].plot(tm, fo, 'k.', alpha=0.2)
            axs.flat[i].plot(tm, fm + pr[0], 'k')
            axs.flat[i].fill_between(tm, pr[0] + fm + 3*sqrt(pr[1]), pr[0] + fm - 3*sqrt(pr[1]), alpha=0.5)
        fig.tight_layout()
        return fig

    def plot_noise(self, xtype='time'):
        binf = arange(1, 50)
        residuals = self.gplpf.residuals(self.lm_pv)
        prediction = self.gplpf.predict(self.lm_pv)
        exptime = diff(self.lcs[0].btime).mean()

        fig, axs = subplots(4, 2, figsize=(11, 6), sharey='row', sharex='col',
                            gridspec_kw={'height_ratios': [0.7, 0.3, 0.7, 0.3]})
        ymin, ymax = inf, -inf
        for i, (res, pre) in enumerate(zip(residuals, prediction)):
            nbins = floor(res.size / binf)
            totp = (binf * nbins).astype(np.int)
            s0 = 1e6 * array([mad_std(res[:tp].reshape([-1, nb]).mean(1)) for nb, tp in zip(binf, totp)])
            s1 = 1e6 * array([mad_std((res - pre)[:tp].reshape([-1, nb]).mean(1)) for nb, tp in zip(binf, totp)])
            ymin = min([ymin, min([s0.min(), s1.min()])])
            ymax = max([ymax, max([s0.max(), s1.max()])])
            x = binf * exptime * 24 * 60 if xtype == 'time' else sqrt(binf)
            axs[::2].flat[i].plot(x, s0)
            axs[::2].flat[i].plot(x, s1)
            axs[1::2].flat[i].plot(x, s0 - s1)
        xlabel = 'Integration time [min]' if xtype == 'time' else 'sqrt(nbins)'
        setp(axs[0::2, 0], ylabel='STD [ppm]', ylim=(0.95 * ymin, 1.05 * ymax))
        setp(axs[1::2, 0], ylabel='$\Delta$ STD [ppm]')
        setp(axs[-1, :], xlabel=xlabel)
        fig.tight_layout()
        return fig


    def save(self):
        delm = xa.DataArray(self.lmlpf.de.population, dims='pvector lm_parameter'.split(),
                            coords={'lm_parameter': self.lmlpf.ps.names})
        degp = xa.DataArray(self.gplpf.de.population, dims='pvector gp_parameter'.split(),
                            coords={'gp_parameter': self.gplpf.ps.names})
        gphp = xa.DataArray(self.gplpf.gphps, dims='filter gp_hyperparameter'.split(),
                            coords={'filter': 'g r i z'.split(),
                                    'gp_hyperparameter': 'sky airmass xy_amplitude xy_scale entropy'.split()})
        gpmc = xa.DataArray(self.gplpf.sampler.chain, dims='pvector step gp_parameter'.split(),
                            coords={'gp_parameter': self.gplpf.ps.names},
                            attrs={'ndim': self.gplpf.de.n_par, 'npop': self.gplpf.de.n_pop})
        ds = xa.Dataset(data_vars={'de_population_lm': delm, 'de_population_gp': degp,
                                   'gp_hyperparameters': gphp, 'gp_mcmc': gpmc},
                        attrs={'created': strftime('%Y-%m-%d %H:%M:%S'), 'obsnight':self.ddata.absolute().name,
                               'tid':self.tid, 'cids':self.cids, 'target':self.target})
        ds.to_netcdf('{}_{}_{}_{}_fit.nc'.format(self.target, self.ddata.absolute().name,
                                                 'nongray' if self.free_k else 'gray',
                                                 'blended' if self.contamination else 'noblend'))


    def save_fits(self, model='linear', npoly=10):
        assert model in ('linear', 'gp', 'linpoly')
        phdu = pf.PrimaryHDU()
        phdu.header.update(target=self.target, night=self.date)
        hdul = pf.HDUList(phdu)

        if model == 'linpoly':
            times, fcorr, fsys, fbase = self.detrend_polynomial(20)
            for i, pb in enumerate('g r i z'.split()):
                df = Table(np.transpose([times[i], fcorr[i], fsys[i], fbase[i]]),
                           names='time flux trend model'.split(),
                           meta={'extname': pb, 'filter': pb, 'trends': model})
                hdul.append(pf.BinTableHDU(df))
            fname = '{}_{}_{}.fits'.format(self.target, self.date, model)
        else:
            if model == 'linear':
                lpf = self.lmlpf
                pv = self.lm_pv
                trends = lpf.trends(pv)
            else:
                lpf = self.gplpf
                pv = self.gp_pv
                trends = lpf.predict(pv)

            transit = lpf.transit_model(pv)

            for i, pb in enumerate('g r i z'.split()):
                df = Table(np.transpose([lpf.times[i], lpf.fluxes[i] - trends[i], trends[i], transit[i]]),
                           names='time flux trend model'.split(),
                           meta={'extname': pb, 'filter': pb, 'trends': model})
                hdul.append(pf.BinTableHDU(df))

            fname = '{}_{}_{}_{}_{}.fits'.format(self.target, self.date, model,
                                                 'nongray' if self.free_k else 'gray',
                                                 'blended' if self.contamination else 'noblend')
        hdul.writeto(fname, overwrite=True)
