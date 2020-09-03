#  MuSCAT2 photometry and transit analysis pipeline
#  Copyright (C) 2019  Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from pathlib import Path

import astropy.io.fits as pf
import pandas as pd
import xarray as xa
from astropy.stats import sigma_clip
from astropy.table import Table
from matplotlib.artist import setp
from matplotlib.pyplot import subplots
from numpy import arange, sort, log10, sqrt, diff, unique, percentile, squeeze, array, concatenate, median, ndarray
from numpy.random import uniform, permutation, normal
from pytransit import LinearModelBaseline
from pytransit.orbits import epoch

from .m2baselpf import M2BaseLPF, downsample_time


def read_reduced_m2(datadir, pattern='*.fits', remove_trends=True, downsample=None):
    files = sorted(Path(datadir).glob(pattern))
    times, fluxes, residuals, pbs, wns, covs, vars = [], [], [], [], [], [], []
    for f in files:
        with pf.open(f) as hdul:
            npb = (len(hdul)-1)//2
            for ipb in range(npb):
                hdu = hdul[1 + ipb]
                if remove_trends:
                    fobs = hdu.data['flux'].astype('d').copy()
                else:
                    fobs = hdu.data['flux'].astype('d').copy() * hdu.data['baseline'].astype('d').copy()
                fmod = hdu.data['model'].astype('d').copy()
                time = hdu.data['time_bjd'].astype('d').copy()
                mask = ~sigma_clip(fobs-fmod, sigma=3).mask

                pbs.append(hdu.header['filter'])
                wns.append(hdu.header['wn'])

                if downsample is None:
                    times.append(time[mask])
                    fluxes.append(fobs[mask])
                    residuals.append((fobs-fmod)[mask])
                    vars.append((fobs - fmod)[mask].var())
                    covs.append(Table.read(f, 1 + npb + ipb).to_pandas().values[mask, 1:])
                else:
                    cov = Table.read(f, 1 + npb + ipb).to_pandas().values[mask, 1:]
                    tb, fb = downsample_time(time[mask], fobs[mask], downsample)
                    _, rb = downsample_time(time[mask], (fobs - fmod)[mask], downsample)
                    _, cb = downsample_time(time[mask], cov, downsample)
                    times.append(tb)
                    fluxes.append(fb)
                    residuals.append(rb)
                    vars.append(rb.var())
                    covs.append(cb)

    return times, fluxes, pbs, wns, covs, array(vars), residuals


class M2MultiNightLPF(M2BaseLPF):
    def __init__(self, target: str, use_opencl: bool = False, n_legendre: int = 0,
                 with_transit=True, with_contamination=False,
                 radius_ratio: str = 'achromatic', noise_model='white', klims=(0.005, 0.25),
                 datadir='results', filename_pattern='*.fits',
                 contamination_model: str = 'physical',
                 contamination_reference_passband: str = "r'",
                 use_linear_baseline_model: bool = True,
                 downsampling: float = None):

        self.datadir = datadir
        self.pattern = filename_pattern
        self._reduction_residuals = None
        self.downsampling = downsampling

        super().__init__(target, use_opencl, n_legendre, with_transit, with_contamination, radius_ratio,
                         noise_model=noise_model, klims=klims,
                         contamination_model=contamination_model,
                         contamination_reference_passband=contamination_reference_passband,
                         use_linear_baseline_model=use_linear_baseline_model)

    def _read_data(self):
        times, fluxes, pbs, wns, covs, vars, residuals = read_reduced_m2(self.datadir, self.pattern, downsample=self.downsampling)
        pbs = pd.Categorical(pbs, categories='g r i z_s'.split(), ordered=True).remove_unused_categories()
        pbnames = pbs.categories.values
        pbids = pbs.codes
        self._residual_vars = vars
        self._reduction_residuals = residuals
        return pbnames, times, fluxes, covs, wns, pbids, arange(len(fluxes))

    def create_pv_population(self, npv: int = 50) -> ndarray:
        pvp = self.ps.sample_from_prior(npv)
        if self.use_linear_baseline_model:
            for p in self.ps[self._sl_lm]:
                if 'lm_i' in p.name:
                    pvp[:, p.pid] = 0.01 * (pvp[:, p.pid] - 1.0) + 1.0
                else:
                    pvp[:, p.pid] *= 0.01
        return pvp

    # def create_pv_population_b(self, npop=50):
    #     pattern = Path(self.pattern).with_suffix('.nc').name
    #     files = sorted(Path(self.datadir).glob(pattern))
    #     pvp = self.ps.sample_from_prior(npop)
    #
    #     for i, f in enumerate(files):
    #         with xa.open_dataset(f) as ds:
    #             fc = array(ds.lm_mcmc).reshape([-1, ds.lm_mcmc.shape[-1]])
    #         if i == 0:
    #             pvp[:, 1:8] = fc[:npop, 1:8]
    #
    #     c = 0
    #     for ilc in range(self.nlc):
    #         for icov in range(self.ncov[ilc] + 1):
    #             pvp[:, self._start_lm + c] = normal(0, sqrt(self._residual_vars[ilc]), size=npop)
    #             c += 1
    #
    #     pvv = uniform(size=(npop, 2 * self.npb))
    #     pvv[:, ::2] = sort(pvv[:, ::2], 1)[:, ::-1]
    #     pvv[:, 1::2] = sort(pvv[:, 1::2], 1)[:, ::-1]
    #     pvp[:, self._sl_ld] = pvv
    #
    #     for i in range(self.nlc):
    #         wn = diff(self.ofluxa).std() / sqrt(2)
    #         pvp[:, self._start_wn] = log10(uniform(0.5 * wn, 2 * wn, size=npop))
    #     return pvp

    def downsample(self, exptime: float) -> None:
        raise NotImplementedError('Downsampling after data reading is not yet supported.')
        bts, bfs, brs, bcs = [], [], [], []
        for i in range(len(self.times)):
            bt, bf = downsample_time(self.times[i], self.fluxes[i], exptime)
            _,  br = downsample_time(self.times[i], self._reduction_residuals[i], exptime)
            _,  bc = downsample_time(self.times[i], self.covariates[i], exptime)
            bts.append(bt)
            bfs.append(bf)
            brs.append(br)
            bcs.append(bc)
        wns = [diff(f).std() / sqrt(2) for f in bfs]
        self._init_data(bts, bfs, pbids=self.pbids, covariates=bcs, wnids=self.noise_ids)
        self._reduction_residuals = brs
        self.ncov = array([c.shape[1] for c in self.covariates])  # Number of covariates per light curve
        self.cova = concatenate([c.ravel() for c in self.covariates])  # Flattened covariate vector

    def plot_light_curves(self, method='de', width: float = 3., max_samples: int = 100, figsize=None,
                          data_alpha=0.5, ylim=None, remove_baseline=True):
        if method == 'mcmc':
            df = self.posterior_samples(derived_parameters=False, add_tref=False)
            t0, p = df.tc.median(), df.p.median()
            pvs = permutation(df.values)[:max_samples]

            if self.noise_model == 'white':
                fbasel = self.baseline(pvs)
            else:
                fbasel = self._lnlikelihood_models[0].predict_baseline(median(pvs, 0)).reshape([1, -1])

            fmodel = self.flux_model(pvs)
            if remove_baseline:
                fobs = self.ofluxa/median(fbasel, 0)
            else:
                fobs = self.ofluxa
                fmodel *= fbasel
            fmperc = percentile(fmodel, [50, 16, 84, 2.5, 97.5], 0)

        else:
            t0, p = self.de.minimum_location[0], self.de.minimum_location[1]
            if self.noise_model == 'white':
                fbasel = squeeze(self.baseline(self.de.minimum_location))
            else:
                fbasel = squeeze(self._lnlikelihood_models[0].predict_baseline(self.de.minimum_location))

            if remove_baseline:
                fobs = self.ofluxa/fbasel
                fmodel = squeeze(self.transit_model(self.de.minimum_location))
            else:
                fobs = self.ofluxa
                fmodel = squeeze(self.flux_model(self.de.minimum_location))*fbasel
            fmperc = None

        epochs = [epoch(t.mean(), t0, p) for t in self.times]
        n_epochs = unique(epochs).size
        epoch_to_row = {e: i for i, e in enumerate(unique(epochs))}

        fig, axs = subplots(n_epochs, 4, figsize=figsize, sharey='all', sharex='all')
        for i in range(self.nlc):
            sl = self.lcslices[i]
            e = epochs[i]
            irow = epoch_to_row[e]
            icol = self.pbids[i]
            ax = axs[irow, icol]

            tc = t0 + e*p
            time = self.timea[sl] - tc

            ax.plot(time, fobs[sl], '.', alpha=data_alpha)

            if method == 'de':
                ax.plot(time, fmodel[sl], 'w', lw=4)
                ax.plot(time, fmodel[sl], 'k', lw=1)
            else:
                ax.fill_between(time, *fmperc[3:5, sl], alpha=0.55)
                ax.fill_between(time, *fmperc[1:3, sl], alpha=0.75)
                ax.plot(time, fmperc[0, sl], 'k')

        setp(axs, xlim=(-width/2/24, width/2/24))
        setp(axs[:, 0], ylabel='Normalised flux')
        setp(axs[-1], xlabel=f'Time - T$_c$ [d]')

        if ylim is not None:
            setp(axs, ylim=ylim)

        [ax.set_title(pb) for ax, pb in zip(axs[0], "g' r' i' z'".split())]
        setp([a.get_xticklabels() for a in axs.flat[:-4]], visible=False)
        setp([a.get_yticklabels() for a in axs[:, 1:].flat], visible=False)
        fig.subplots_adjust(hspace=0, wspace=0, left=0.05, right=0.98, bottom=0.05, top=0.95)
        return fig