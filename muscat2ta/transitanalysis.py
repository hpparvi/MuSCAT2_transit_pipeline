from glob import glob

import numpy as np
import pandas as pd
import xarray as xa

from pathlib import Path
from time import strftime

from corner import corner
from tqdm import tqdm
from matplotlib.pyplot import subplots, setp
from numpy import array, arange, min, max, sqrt, inf, floor, diff, percentile, median, isin, zeros, full_like
from numpy.random import permutation
from astropy.stats import mad_std
from astropy.table import Table
from astropy.io import fits as pf

from muscat2ph.phdata import PhotometryData
from muscat2ta.lc import M2LCSet, M2LightCurve
from muscat2ta.lpf import StudentLSqLPF, GPLPF, NormalLSqLPF

class TransitAnalysis:
    def __init__(self, datadir, target, date, tid, cids, etime=30., mjd_start=-inf, mjd_end=inf, flux_lims=(-inf, inf),
                 model='pb_independent_k', npop=100, pbs=('g', 'r', 'i', 'z_s'), fit_wn=True, **kwargs):
        self.ddata = dd = Path(datadir)
        self.target = target
        self.date = date
        self.tid = tid
        self.cids = cids
        self.model = model
        self.npop = npop
        self.etime = etime
        self.models = None
        self.pbs = pbs
        self.flux_lims = flux_lims
        self.fit_wn = fit_wn

        self.phs = [PhotometryData(dd.joinpath('{}_{}_{}.nc'.format(target, date, pb)), tid, cids, objname=target,
                                   mjd_start=mjd_start, mjd_end=mjd_end, **kwargs)
                    for pb in self.pbs]
        self._init_lcs()

        self.lm_pv = None
        self.gp_pv = None


    def _init_lcs(self):
        self.lcs = M2LCSet([M2LightCurve(pbi, ph.bjd, ph.relative_flux, ph.relative_error, ph.aux, array(ph.aux.quantity)) for pbi,ph in enumerate(self.phs)])
        self.lcs.mask_outliers()
        self.lcs.mask_covariate_outliers()
        self.lcs.mask_limits(self.flux_lims)

        try:
            for ph, lc in zip(self.phs, self.lcs):
                et = float(ph._aux.loc[:, 'exptime'].mean())
                lc.downsample_time(self.etime)
        except ValueError:
            print("Couldn't initialise the light curves")
            return

        try:
            self.lmlpf = NormalLSqLPF(self.target, self.lcs, self.pbs, model=self.model)
            self.gplpf = GPLPF(self.target, self.lcs, self.pbs, model=self.model, fit_wn=self.fit_wn)
            self.models = {'linear':self.lmlpf, 'gp':self.gplpf}
        except ValueError:
            print("Couldn't initialise the LPFs")


    def optimize_comparison_stars(self, n_stars=1, start_id=0, end_id=10, start_apt=0, end_apt=None):
        for ph in self.phs:
            ph._rset.select_best(n_stars, start_id, end_id, start_apt, end_apt)

    def print_ptp_scatter(self):
        r1s = [res.std() for res in self.gplpf.residuals(self.gp_pv)]
        r2s = [(res - pre).std() for res, pre in zip(self.gplpf.residuals(self.gp_pv), self.gplpf.predict(self.gp_pv))]
        for r1, r2, pb in zip(r1s, r2s, 'g r i z'.split()):
            print('{} {:5.0f} ppm -> {:5.0f} ppm'.format(pb, 1e6 * r1, 1e6 * r2))

    def set_prior(self, i, p):
        for lpf in (self.lmlpf, self.gplpf):
            lpf.ps[i].prior = p

    def add_t14_prior(self, m, s):
        for lpf in (self.lmlpf, self.gplpf):
            lpf.add_t14_prior(m, s)

    def add_ldtk_prior(self, teff, logg, z, uncertainty_multiplier=3, pbs=('g', 'r', 'i', 'z')):
        from ldtk import LDPSetCreator
        from ldtk.filters import sdss_g, sdss_r, sdss_i, sdss_z
        fs = {n: f for n, f in zip('g r i z'.split(), (sdss_g, sdss_r, sdss_i, sdss_z))}
        filters = [fs[k] for k in pbs]
        self.ldsc = LDPSetCreator(teff, logg, z, filters)
        self.ldps = self.ldsc.create_profiles(1000)
        self.ldps.resample_linear_z()
        self.ldps.set_uncertainty_multiplier(uncertainty_multiplier)
        for lpf in (self.lmlpf, self.gplpf):
            lpf.ldsc = self.ldsc
            lpf.ldps = self.ldps
            lpf.lnpriors.append(lambda pv:self.ldps.lnlike_tq(pv[lpf._sl_ld]))


    def detrend_polynomial(self, npol=20):
        time, fcorr, fsys, fbase = [], [], [], []
        for lc in self.lcs:
            tm, fc, fs, fb = lc.detrend_poly(npol)
            time.append(tm)
            fcorr.append(fc)
            fsys.append(fs)
            fbase.append(fb)
        return time, fcorr, fsys, fbase


    def optimize(self, model, niter=100, pop=None):
        assert model in self.models.keys()
        if model == 'linear':
            self.lmlpf.optimize_global(niter, self.npop, pop, label='Optimizing linear model')
            self.lm_pv = self.lmlpf.de.minimum_location
        elif model == 'gp':
            if pop is None and self.gplpf.de is None and self.lmlpf.de is not None:
                pop = self.gplpf.create_pv_population(self.npop)
                stop = self.lmlpf.ps.blocks[2].stop
                pop[:,:stop] = self.lmlpf.de.population[:,:stop]
            self.gplpf.optimize_global(niter, self.npop, pop, label='Optimizing GP model')
            self.gp_pv = self.gplpf.de.minimum_location


    def sample(self, model, niter=100, thin=5, reset=False):
        assert model in self.models.keys()
        self.models[model].sample_mcmc(niter, thin=thin, label='Sampling linear model', reset=reset)


    def mask_flux_outliers(self, sigma=4):
        assert self.lm_pv is not None, "Need to run global optimization before calling outlier masking"
        self.lmlpf.mask_outliers(sigma=sigma, means=self.lmlpf.flux_model(self.lm_pv))


    def learn_gp_hyperparameters(self, pv=None, joint_fit=True, method='L-BFGS-B'):
        pv = pv if pv is not None else self.lm_pv
        assert pv is not None, 'Must carry out linear model optimisation before GP hyperparameter optimisation'
        if joint_fit:
            self.gplpf.optimize_hps_jointly(pv, method=method)
        else:
            self.gplpf.optimize_hps(pv, method=method)

    def posterior_samples(self, burn=0, thin=1, model='gp', include_ldc=False):
        assert model in ('linear', 'gp')
        lpf = self.lmlpf if model == 'linear' else self.gplpf
        ldstart = lpf._slld[0].start
        fc = lpf.sampler.chain[:,burn::thin,:].reshape([-1, lpf.de.n_par])
        if include_ldc:
            return pd.DataFrame(fc, columns=lpf.ps.names)
        else:
            return pd.DataFrame(fc[:,:ldstart], columns=lpf.ps.names[:ldstart])

    def plot_mcmc_chains(self, pid=0, model='gp', alpha=0.1, thin=1, ax=None):
        assert model in ('linear', 'gp')
        lpf = self.lmlpf if model == 'linear' else self.gplpf
        fig, ax = (None, ax) if ax is not None else subplots()
        ax.plot(lpf.sampler.chain[:,::thin,pid].T, 'k', alpha=alpha)
        fig.tight_layout()
        return fig

    def plot_basic_posteriors(self, model='gp', burn=0, thin=1):
        df = self.posterior_samples(burn, thin, model, False)
        df['k'] = sqrt(df.k2)
        df.drop(['k2'], axis=1, inplace=True)
        return corner(df)


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


    @property
    def savefile_name(self):
        return '{}_{}_{}_fit.nc'.format(self.target, self.ddata.absolute().name, self.model)

    def load(self):
        ds = xa.open_dataset(self.savefile_name).load()
        ds.close()
        return ds

    def save(self):
        delm = None
        if self.lmlpf.de:
            delm = xa.DataArray(self.lmlpf.de.population, dims='pvector lm_parameter'.split(),
                                coords={'lm_parameter': self.lmlpf.ps.names})

        degp = None
        if self.gplpf.de:
            degp = xa.DataArray(self.gplpf.de.population, dims='pvector gp_parameter'.split(),
                                coords={'gp_parameter': self.gplpf.ps.names})

        gphp = None
        if self.gplpf.gphps is not None:
            if self.fit_wn:
                gphpls = 'log_var sky airmass xy_amplitude xy_scale entropy'.split()
            else:
                gphpls = 'sky airmass xy_amplitude xy_scale entropy'.split()

            gphp = xa.DataArray(self.gplpf.gphps, dims='filter gp_hyperparameter'.split(),
                                coords={'filter': 'g r i z'.split(),
                                        'gp_hyperparameter':gphpls})

        lmmc = None
        if self.lmlpf.sampler is not None:
            lmmc = xa.DataArray(self.lmlpf.sampler.chain, dims='pvector step lm_parameter'.split(),
                                coords={'lm_parameter': self.lmlpf.ps.names},
                                attrs={'ndim': self.lmlpf.de.n_par, 'npop': self.lmlpf.de.n_pop})

        gpmc = None
        if self.gplpf.sampler is not None:
            gpmc = xa.DataArray(self.gplpf.sampler.chain, dims='pvector step gp_parameter'.split(),
                                coords={'gp_parameter': self.gplpf.ps.names},
                                attrs={'ndim': self.gplpf.de.n_par, 'npop': self.gplpf.de.n_pop})

        ds = xa.Dataset(data_vars={'de_population_lm': delm, 'de_population_gp': degp,
                                   'gp_hyperparameters': gphp, 'lm_mcmc': lmmc, 'gp_mcmc': gpmc},
                        attrs={'created': strftime('%Y-%m-%d %H:%M:%S'), 'obsnight':self.ddata.absolute().name,
                               'tid':self.tid, 'cids':self.cids, 'target':self.target})

        ds.to_netcdf(self.savefile_name)


    def save_fits(self, model='linear', npoly=10):
        assert model in ('linear', 'gp', 'linpoly')
        phdu = pf.PrimaryHDU()
        phdu.header.update(target=self.target, night=self.date)
        hdul = pf.HDUList(phdu)

        if model == 'linpoly':
            times, fcorr, fsys, fbase = self.detrend_polynomial(npoly)
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
                cnames = lpf.datasets[0]._covnames
            else:
                lpf = self.gplpf
                pv = self.gp_pv
                trends = lpf.predict(pv)
                cnames = lpf.datasets[0]._covnames[2:]

            transit = lpf.transit_model(pv)

            for i, pb in enumerate('g r i z'.split()):
                df = Table(np.transpose([lpf.times[i], lpf.fluxes[i] - trends[i], trends[i], transit[i]]),
                           names='time flux trend model'.split(),
                           meta={'extname': pb, 'filter': pb, 'trends': model, 'wn':lpf.wn})
                hdul.append(pf.BinTableHDU(df))

            for i, pb in enumerate('g r i z'.split()):
                df = Table(lpf.covariates[i], names=cnames, meta={'extname': 'aux_' + pb})
                hdul.append(pf.BinTableHDU(df))

            fname = '{}_{}_{}_{}.fits'.format(self.target, self.date, model, self.model)
        hdul.writeto(fname, overwrite=True)


class MultiTransitAnalysis(TransitAnalysis):
    def __init__(self, datadir, target, ftemplate, model='pb_independent_k', npop=100, pbs=('g', 'r', 'i', 'z_s'), fit_wn=True):
        self.ddata = Path(datadir)
        self.target = target
        self.ftemplate = ftemplate
        self.pbs = pbs
        self.fit_wn = fit_wn
        self.model = model
        self.npop = npop
        self.models = None
        self.lm_pv = None
        self.gp_pv = None
        self._init_lcs()

    def _init_lcs(self):
        files = sorted(glob(self.ftemplate))
        lcs = []
        for fname in files:
            with pf.open(fname) as f:
                for i in range(4):
                    d, h = f[i + 1].data, f[i + 1].header
                    aux = pd.DataFrame(f[4 + i + 1].data)
                    time, flux = d['time'], d['flux'] + d['trend']
                    error = full_like(time, h['wn'])
                    lcs.append(M2LightCurve(i, time, flux, error, aux.values[:, 1:], aux.columns[1:]))
        self.lcs = M2LCSet(lcs)

        try:
            self.lmlpf = NormalLSqLPF(self.target, self.lcs, self.pbs, model=self.model)
            self.gplpf = GPLPF(self.target, self.lcs, self.pbs, model=self.model, fit_wn=self.fit_wn)
            self.models = {'linear': self.lmlpf, 'gp': self.gplpf}
        except ValueError:
            print("Couldn't initialise the LPFs")

    @property
    def savefile_name(self):
        return '{}_multi_{}_fit.nc'.format(self.target, self.model)

    def save(self):
        delm = None
        if self.lmlpf.de:
            delm = xa.DataArray(self.lmlpf.de.population, dims='pvector lm_parameter'.split(),
                                coords={'lm_parameter': self.lmlpf.ps.names})

        degp = None
        if self.gplpf.de:
            degp = xa.DataArray(self.gplpf.de.population, dims='pvector gp_parameter'.split(),
                                coords={'gp_parameter': self.gplpf.ps.names})

        gphp = None
        if self.gplpf.gphps is not None:
            if self.fit_wn:
                gphpls = 'log_var sky airmass xy_amplitude xy_scale entropy'.split()
            else:
                gphpls = 'sky airmass xy_amplitude xy_scale entropy'.split()

            gphp = xa.DataArray(self.gplpf.gphps, dims='light_curve gp_hyperparameter'.split(),
                                coords={'light_curve': range(self.gplpf.nlc),
                                        'gp_hyperparameter': gphpls})

        lmmc = None
        if self.lmlpf.sampler is not None:
            lmmc = xa.DataArray(self.lmlpf.sampler.chain, dims='pvector step lm_parameter'.split(),
                                coords={'lm_parameter': self.lmlpf.ps.names},
                                attrs={'ndim': self.lmlpf.de.n_par, 'npop': self.lmlpf.de.n_pop})

        gpmc = None
        if self.gplpf.sampler is not None:
            gpmc = xa.DataArray(self.gplpf.sampler.chain, dims='pvector step gp_parameter'.split(),
                                coords={'gp_parameter': self.gplpf.ps.names},
                                attrs={'ndim': self.gplpf.de.n_par, 'npop': self.gplpf.de.n_pop})

        ds = xa.Dataset(data_vars={'de_population_lm': delm, 'de_population_gp': degp,
                                   'gp_hyperparameters': gphp, 'lm_mcmc': lmmc, 'gp_mcmc': gpmc},
                        attrs={'created': strftime('%Y-%m-%d %H:%M:%S'), 'obsnight': self.ddata.absolute().name,
                               'target': self.target})

        ds.to_netcdf(self.savefile_name)


    def plot_light_curve(self, model='linear', method='de', detrend_obs=False, detrend_mod=True,
                         include_mod=True, figsize=(11, 6), xwidth=None):
        assert model in ('linear', 'gp')
        assert method in ('de', 'mcmc')

        lpf = self.lmlpf if model == 'linear' else self.gplpf
        posterior_limits = None
        pv = lpf.de.minimum_location

        npb = len(self.pbs)
        nlc = self.lcs.size
        nni = nlc // npb

        if model == 'linear':
            baseline = lpf.baseline(pv)
            trends = lpf.trends(pv)
            fluxes_obs = [lpf.fluxes[i] - trends[i] for i in range(nlc)]
            fluxes_trm = lpf.transit_model(pv)
            fluxes_mod = lpf.transit_model(pv) if detrend_mod else lpf.flux_model(pv)
        else:
            baseline = lpf.predict(pv)
            fluxes_obs = [lpf.fluxes[i] - baseline[i] for i in range(4)] if detrend_obs else lpf.fluxes
            fluxes_trm = lpf.transit_model(pv)
            fluxes_mod = lpf.flux_model(pv)

        xwidth = xwidth or max([t.ptp() for t in self.lcs.times])
        tn = array([(t.mean() - pv[0]) / pv[1] for t in self.lcs.times]).round()
        tc = pv[0] + tn * pv[1]

        fig, axs = subplots(npb, nni, figsize=figsize, sharex='col', sharey=True)
        for iax, ax in enumerate(axs.T.flat):
            ax.plot(lpf.times[iax], fluxes_obs[iax], '.', alpha=0.2)
            ax.plot(lpf.times[iax], fluxes_mod[iax], 'k')
            setp(ax, xlim=(tc[iax] - 0.5 * xwidth, tc[iax] + 0.5 * xwidth))

        fig.tight_layout()

