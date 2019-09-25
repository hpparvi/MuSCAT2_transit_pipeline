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
from numpy import arange, newaxis, atleast_2d, zeros, sort, log10, sqrt, diff, unique, percentile, squeeze, array, \
    isfinite, digitize, nan, ceil, full
from numpy.random import uniform, permutation

from pytransit import BaseLPF, LinearModelBaseline
from pytransit.lpf.lpf import map_pv, map_ldc
from pytransit.orbits import epoch
from pytransit.param.parameter import PParameter, UniformPrior as UP

from .m2lpf import change_depth

def downsample_time(time, values, inttime=30.):
    if values.ndim == 1:
        values = atleast_2d(values).T
    duration = 24. * 60. * 60. * time.ptp()
    nbins = int(ceil(duration / inttime))
    bins = arange(nbins)
    edges = time[0] + bins * inttime / 24 / 60 / 60
    bids = digitize(time, edges) - 1
    bt, bv = full(nbins, nan), zeros((nbins, values.shape[1]))
    for i, bid in enumerate(bins):
        bmask = bid == bids
        if bmask.sum() > 0:
            bt[i] = time[bmask].mean()
            bv[i,:] = values[bmask,:].mean(0)
    m = isfinite(bt)
    return bt[m], squeeze(bv[m])

def read_reduced_m2(datadir, pattern='*.fits'):
    files = sorted(Path(datadir).glob(pattern))
    times, fluxes, pbs, wns, covs = [], [], [], [], []
    for f in files:
        with pf.open(f) as hdul:
            npb = (len(hdul)-1)//2
            masks = []
            for hdu in hdul[1:1+npb]:
                fobs = hdu.data['flux'].astype('d').copy() * hdu.data['baseline'].astype('d').copy()
                fmod = hdu.data['model'].astype('d').copy()
                time = hdu.data['time_bjd'].astype('d').copy()
                mask = ~sigma_clip(fobs-fmod, sigma=5).mask
                masks.append(mask)
                times.append(time[mask])
                fluxes.append(fobs[mask])
                pbs.append(hdu.header['filter'])
                wns.append(hdu.header['wn'])
            for i in range(npb):
                covs.append(Table.read(f, 1+npb+i).to_pandas().values[masks[i],:])
    return times, fluxes, pbs, wns, covs


class M2MultiNightLPF(LinearModelBaseline, BaseLPF):
    def __init__(self, name: str, min_exptime=None, datadir='results', filename_pattern='*.fits', result_dir='results'):
        self.datadir = datadir
        self.pattern = filename_pattern

        times, fluxes, pbs, wns, covs = read_reduced_m2(datadir, filename_pattern)

        if min_exptime is not None:
            bts, bfs, bcs = [], [], []
            for i in range(len(times)):
                bt, bf = downsample_time(times[i], fluxes[i], 60.)
                bt, bc = downsample_time(times[i], covs[i], 60.)
                bts.append(bt)
                bfs.append(bf)
                bcs.append(bc)
            times, fluxes, covs = bts, bfs, bcs
            wns = [diff(f).std()/sqrt(2) for f in fluxes]

        pbs = pd.Categorical(pbs, categories='g r i z_s'.split(), ordered=True).remove_unused_categories()
        pbnames = pbs.categories.values
        pbids = pbs.codes
        BaseLPF.__init__(self, name, pbnames, times, fluxes, pbids=pbids, wnids=arange(len(pbs)),
                         covariates=covs, result_dir=result_dir)

    def _init_p_planet(self):
        pk2 = [PParameter(f'k2_{pb}', f'area_ratio {pb}', 'A_s', UP(0.01 ** 2, 0.25 ** 2), (0.01 ** 2, 0.25 ** 2)) for
               pb in self.passbands]
        self.ps.add_passband_block('k2', 1, self.npb, pk2)
        self._pid_k2 = arange(self.npb) + self.ps.blocks[-1].start
        self._start_k2 = self.ps.blocks[-1].start
        self._sl_k2 = self.ps.blocks[-1].slice

    def transit_model(self, pv, copy=True):
        pv = atleast_2d(pv)
        mean_ar = pv[:, self._sl_k2].mean(1)
        pvv = zeros((pv.shape[0], pv.shape[1] - self.npb + 1))
        pvv[:, :4] = pv[:, :4]
        pvv[:, 4] = mean_ar
        pvv[:, 5:] = pv[:, 4 + self.npb:]
        pvp = map_pv(pvv)
        ldc = map_ldc(pvv[:, 5:5 + 2 * self.npb])
        flux = self.tm.evaluate_pv(pvp, ldc, copy)
        rel_ar = pv[:, self._sl_k2] / mean_ar[:, newaxis]
        flux = change_depth(rel_ar, flux, self.lcids, self.pbids)
        return flux

    def create_pv_population(self, npop=50):
        pattern = Path(self.pattern).with_suffix('.nc').name
        files = sorted(Path(self.datadir).glob(pattern))
        pvp = self.ps.sample_from_prior(npop)

        ibla = 0
        for i, f in enumerate(files):
            with xa.load_dataset(f) as ds:
                fc = array(ds.lm_mcmc).reshape([-1, ds.lm_mcmc.shape[-1]])
                blcs = [p for p in array(ds.lm_parameter) if 'bl' in p]
                bl = ds.lm_mcmc.loc[:, :, blcs]
                bl = array(bl).reshape([-1, bl.shape[-1]])

            if i == 0:
                pvp[:, 1:8] = fc[:npop, 1:8]

            for ibl in range(bl.shape[1]):
                pvp[:, self._start_bl + ibla + ibl] = bl[:npop, ibl]
            ibla += bl.shape[1]

        pvv = uniform(size=(npop, 2 * self.npb))
        pvv[:, ::2] = sort(pvv[:, ::2], 1)[:, ::-1]
        pvv[:, 1::2] = sort(pvv[:, 1::2], 1)[:, ::-1]
        pvp[:, self._sl_ld] = pvv

        for i in range(self.nlc):
            wn = diff(self.ofluxa).std() / sqrt(2)
            pvp[:, self._start_err] = log10(uniform(0.5 * wn, 2 * wn, size=npop))
        return pvp


    def plot_light_curves(self, method='de', width: float = 3., max_samples: int = 100, figsize=None, data_alpha=0.5,
                          ylim=None):
        if method == 'mcmc':
            df = self.posterior_samples(derived_parameters=False)
            t0, p = df.tc.median(), df.p.median()
            fmodel = self.flux_model(permutation(df.values)[:max_samples])
            fmperc = percentile(fmodel, [50, 16, 84, 2.5, 97.5], 0)
        else:
            fmodel = squeeze(self.flux_model(self.de.minimum_location))
            t0, p = self.de.minimum_location[0], self.de.minimum_location[1]
            fmperc = None

        epochs = [epoch(t.mean(), t0, p) for t in self.times]
        n_epochs = unique(epochs).size
        epoch_to_row = {e: i for i, e in enumerate(unique(epochs))}

        fig, axs = subplots(n_epochs, 4, figsize=figsize, sharey='all', sharex='all')
        for i in range(self.nlc):
            e = epochs[i]
            irow = epoch_to_row[e]
            icol = self.pbids[i]
            ax = axs[irow, icol]

            tc = t0 + e * p
            time = self.times[i] - tc

            ax.plot(time, self.fluxes[i], '.', alpha=data_alpha)

            if method == 'de':
                ax.plot(time, fmodel[self.lcslices[i]], 'w', lw=4)
                ax.plot(time, fmodel[self.lcslices[i]], 'k', lw=1)
            else:
                ax.fill_between(time, *fmperc[3:5, self.lcslices[i]], alpha=0.15)
                ax.fill_between(time, *fmperc[1:3, self.lcslices[i]], alpha=0.25)
                ax.plot(time, fmperc[0, self.lcslices[i]])

        setp(axs, xlim=(-width / 2 / 24, width / 2 / 24))
        setp(axs[:, 0], ylabel='Normalised flux')
        setp(axs[-1], xlabel=f'Time - T$_c$ [d]')

        if ylim is not None:
            setp(axs, ylim=ylim)

        [ax.set_title(pb) for ax, pb in zip(axs[0], "g' r' i' z'".split())]
        setp([a.get_xticklabels() for a in axs.flat[:-4]], visible=False)
        setp([a.get_yticklabels() for a in axs[:, 1:].flat], visible=False)
        fig.subplots_adjust(hspace=0, wspace=0, left=0.05, right=0.98, bottom=0.05, top=0.95)
        return fig
