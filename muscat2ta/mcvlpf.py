#  MuSCAT2 photometry and transit analysis pipeline
#  Copyright (C) 2021  Hannu Parviainen
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
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, List

import seaborn as sb
import matplotlib as mpl
import pandas as pd
from astropy.io import fits as pf
from astropy.stats import sigma_clip
from astropy.table import Table
from ldtk import LDPSetCreator
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import subplots, setp, subplot, figure
from muscat2ph.catalog import get_toi
from numpy import arange, isfinite, zeros, diff, concatenate, sqrt, array, ndarray, inf, atleast_2d, sum, median, where, floor, nanmedian, \
    squeeze, argsort, isnan, linspace, percentile
from pytransit import RoadRunnerModel
from muscat2ta.filters import PYTRANSIT_FILTERS, get_ldtk_filters, ALL_PASSBANDS
from pytransit.contamination import Instrument, SMContamination
from pytransit.lpf.cntlpf import contaminate, PhysContLPF
from pytransit.lpf.tess.tgclpf import BaseTGCLPF
from pytransit.lpf.tesslpf import downsample_time
from pytransit.orbits import as_from_rhop, i_from_ba
from pytransit.utils.downsample import downsample_time_1d, downsample_time_2d
from pytransit.utils.keplerlc import KeplerLC
from pytransit.utils.misc import fold
from uncertainties import ufloat, nominal_value


@dataclass
class Star:
    radius: ufloat
    teff: ufloat
    logg: ufloat
    z: ufloat

    @staticmethod
    def from_toi(toi):
        from astroquery.mast import Catalogs
        tb = Catalogs.query_object(f"TIC {toi.tic}", radius=.002, catalog="TIC")[0]
        radius = ufloat(*tb['rad e_rad'.split()])
        teff = ufloat(*tb['Teff e_Teff'.split()])
        logg = ufloat(*tb['logg e_logg'.split()])

        z = ufloat(*tb['MH e_MH'.split()])
        if isnan(z.n):
            z = ufloat(0, 0.01)
        return Star(radius, teff, logg, z)


@dataclass
class Photometry:
    times: List
    fluxes: List
    covariates: List
    passbands: List
    noise: List
    instrument: List
    piis: List
    exptimes: List
    nsamples: List

    def __add__(self, other):
        return Photometry(self.times+other.times, self.fluxes+other.fluxes, self.covariates+other.covariates,
                          self.passbands+other.passbands, self.noise+other.noise, self.instrument+other.instrument,
                          self.piis+other.piis, self.exptimes+other.exptimes, self.nsamples+other.nsamples)


def read_tess_data(dfiles, zero_epoch: float, period: float, use_pdc: bool = False,
              transit_duration_d: float = 0.1, baseline_duration_d: float = 0.3):
    times, fluxes, ins, piis, exptimes, nsamples = [], [], [], [], [], []
    for dfile in dfiles:
        tb = Table.read(dfile)

        if 'PDCSAP_FLUX' in tb.colnames:
            source = 'SPOC'
            fcol = 'PDCSAP_FLUX' if use_pdc else 'SAP_FLUX'
        elif 'KSPSAP_FLUX' in tb.colnames:
            source = 'QLP'
            fcol = 'KSPSAP_FLUX'

        bjdrefi = tb.meta['BJDREFI']
        df = tb.to_pandas().dropna(subset=['TIME', fcol])
        time = df.TIME.values + bjdrefi
        flux = df[fcol].values
        flux /= nanmedian(flux)
        m = flux > 0.9
        lc = KeplerLC(time[m], flux[m], zeros(flux[m].size), nominal_value(zero_epoch), nominal_value(period), transit_duration_d, baseline_duration_d)
        times.extend(copy(lc.time_per_transit))
        cfluxes = copy(lc.normalized_flux_per_transit)
        if use_pdc and 'CROWDSAP' in tb.meta:
            contamination = 1 - tb.meta['CROWDSAP']
            cfluxes = [contamination + (1 - contamination) * f for f in cfluxes]
        fluxes.extend(cfluxes)
        exptimes.extend(len(cfluxes)*[0.0 if source == 'SPOC' else 0.021])
        nsamples.extend(len(cfluxes)*[1 if source == 'SPOC' else 10])

    ins = len(times) * ["TESS"]
    piis = list(arange(len(times)))
    return Photometry(times, fluxes, len(times) * [array([[]])], len(times) * ['tess'], [diff(concatenate(fluxes)).std() / sqrt(2)],
                      ins, piis, exptimes, nsamples)


def read_m2_data(files, downsample=None, passbands=None, heavy_baseline: bool = True):
    if passbands is None:
        passbands = ALL_PASSBANDS
    times, fluxes, pbs, wns, covs = [], [], [], [], []
    for inight, f in enumerate(files):
        with pf.open(f) as hdul:
            npb = (len(hdul)-1)//2
            for ipb in range(npb):
                hdu = hdul[1 + ipb]
                pb = hdu.header['filter']
                if pb in passbands:
                    fobs = hdu.data['flux'].astype('d').copy()
                    fmod = hdu.data['model'].astype('d').copy()
                    time = hdu.data['time_bjd'].astype('d').copy()
                    mask = ~sigma_clip(fobs-fmod, sigma=5).mask

                    wns.append(hdu.header['wn'])
                    pbs.append(pb)

                    if downsample is None:
                        times.append(time[mask])
                        fluxes.append(fobs[mask])
                        covs.append(Table.read(f, 1+npb+ipb).to_pandas().values[mask,1:])
                    else:
                        cov = Table.read(f, 1+npb+ipb).to_pandas().values[mask,1:]
                        tb, fb, eb = downsample_time_1d(time[mask], fobs[mask], downsample / 24 / 60)
                        _,  cb, _ = downsample_time_2d(time[mask], cov, downsample / 24 / 60)
                        m = isfinite(tb)
                        times.append(tb[m])
                        fluxes.append(fb[m])
                        covs.append(cb[m])
    if not heavy_baseline:
        covs = len(times) * [array([[]])]
    ins = len(times)*["M2"]
    piis = list(arange(len(times)))
    exptimes = len(times)*[0.0]
    nsamples = len(times)*[1]
    return Photometry(times, fluxes, covs, pbs, wns, ins, piis, exptimes, nsamples)


color = sb.color_palette()[0]
color_rgb = mpl.colors.colorConverter.to_rgb(color)
colors = [sb.utils.set_hls_values(color_rgb, l=l) for l in linspace(1, 0, 12)]
cmap = sb.blend_palette(colors, as_cmap=True)

color = sb.color_palette()[1]
color_rgb = mpl.colors.colorConverter.to_rgb(color)
colors = [sb.utils.set_hls_values(color_rgb, l=l) for l in linspace(1, 0, 12)]
cmap2 = sb.blend_palette(colors, as_cmap=True)


def _jplot(xs, y, xlabels=None, ylabel='', figsize=None, nb=30, gs=25, **kwargs):
    nx = len(xs)
    figsize = figsize or (13, 13/nx)
    fig = figure(figsize=figsize)
    gs_ct = GridSpec(2, nx + 1, bottom=0.2, top=1, left=0.1, right=1, hspace=0.05, wspace=0.05,
                     height_ratios=[0.15, 0.85], width_ratios=nx*[1] + [0.2])

    ylim = percentile(y, [0.5, 99.5])
    yper = percentile(y, [50, 75, 95])
    axs_j = []
    axs_m = []
    for i, x in enumerate(xs):
        xlim = percentile(x, [1, 99])
        aj = subplot(gs_ct[1, i])
        am = subplot(gs_ct[0, i])
        aj.hexbin(x, y, gridsize=gs, cmap=cmap, extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
        # for yp in yper:
        #    aj.axhline(yp, lw=1, c='k', alpha=0.2)
        am.hist(x, bins=nb, alpha=0.5, range=xlim)
        setp(aj, xlim=xlim, ylim=ylim)
        setp(am, xlim=aj.get_xlim())
        setp(am, xticks=[], yticks=[])

        if i > 0:
            setp(aj.get_yticklabels(), visible=False)
        else:
            setp(aj, ylabel=ylabel)
        if xlabels is not None:
            setp(aj, xlabel=xlabels[i])
        axs_j.append(aj)
        axs_m.append(am)

    am = subplot(gs_ct[1, -1])
    am.hist(y, bins=nb, alpha=0.5, range=ylim, orientation='horizontal')
    setp(am, ylim=ylim, xticks=[])
    setp(am.get_yticklabels(), visible=False)

    [sb.despine(ax=ax, left=True, offset=0.1) for ax in axs_m]
    [sb.despine(ax=ax) for ax in axs_j]
    sb.despine(ax=am, bottom=True)
    return fig

# Define the log posterior functions
# ----------------------------------
# The `pytransit.lpf.tess.BaseTGCLPF` class can be used directly to model TESS photometry together with ground-based
# multicolour photometry with physical contamination on the latter. The only thing that needs to be implemented is the
# `BaseTGCLPF.read_data()` method that reads and sets up the data. However, we also use the `_post_initialisation`
# hook to set the parameter priors for the so that we don't need to remember to set these later.



class MCVLPF(BaseTGCLPF):
    def __init__(self, toi: float, star: Optional[Star] = None,
                 zero_epoch: Optional[ufloat] = None, period: Optional[ufloat] = None,
                 use_ldtk: bool = True, use_opencl: bool = False, use_pdc: bool = True,
                 heavy_baseline: bool = False, downsample: Optional[float] = None,
                 m2_passbands: Iterable = None):

        name = f"TOI-{toi}"
        self.toi = toi = get_toi(toi)
        self.zero_epoch = zero_epoch or ufloat(toi.epoch[0], toi.epoch[1])
        self.period = period or ufloat(toi.period[0], toi.period[1])
        self.star = star or Star.from_toi(self.toi)

        self.use_pdc = use_pdc
        self.use_opencl = use_opencl
        self.heavy_baseline = heavy_baseline
        self.downsample = downsample
        self.m2_passbands = m2_passbands if m2_passbands is not None else ALL_PASSBANDS
        tm = RoadRunnerModel('power-2-pm', small_planet_limit=0.005, parallel=True)

        self.result_dir = Path('.')
        self._stess = None
        self._ntess = None
        self.data: Optional[Photometry] = None

        times, fluxes, pbnames, pbs, wns, covs = self.read_data()
        pbids = pd.Categorical(pbs, categories=pbnames).codes
        wnids = arange(len(times))
        tref = floor(concatenate(times).min())

        self.wns = wns
        PhysContLPF.__init__(self, name, passbands=pbnames, times=times, fluxes=fluxes, pbids=pbids, wnids=wnids,
                             covariates=covs, tref=tref, tm=tm, nsamples=self.data.nsamples, exptimes=self.data.exptimes)

        if use_ldtk:
            self.set_ldtk_priors()


    def read_data(self):
        ddata = Path('photometry')
        dtess = ddata/'tess'
        tess_files = sorted(dtess.glob('*.fits'))
        dm2 = ddata/'m2'
        m2_files = sorted(dm2.glob('*.fits'))
        dtess = read_tess_data(tess_files, self.zero_epoch.n, self.period.n, baseline_duration_d=0.3, use_pdc=self.use_pdc)
        dm2 = read_m2_data(m2_files, downsample=self.downsample, passbands=self.m2_passbands, heavy_baseline=self.heavy_baseline)
        pbnames = ['tess'] + list(dm2.passbands)
        self._stess = len(dtess.times)
        self._ntess = sum([t.size for t in dtess.times])
        self.data = data = dtess + dm2
        data.fluxes = [f / median(f) for f in data.fluxes]
        data.covariates = [(c-c.mean(0)) / c.std(0) for c in data.covariates]
        return data.times, data.fluxes, pbnames, data.passbands, data.noise, data.covariates

    def _init_instrument(self):
        """Set up the instrument and contamination model."""
        m2_pbs = [pb for pb in self.passbands if pb != 'tess']
        m2_filters = [PYTRANSIT_FILTERS[pb] for pb in m2_pbs]
        self.instrument = Instrument('MuSCAT2', m2_filters)
        self.cm = SMContamination(self.instrument, m2_filters[-1].name)
        self.add_prior(lambda pv: where(pv[:, 4] < pv[:, 5], 0, -inf))
        self.add_prior(lambda pv: where(pv[:, 8] < pv[:, 5], 0, -inf))

    def _post_initialisation(self):
        if self.use_opencl:
            self.tm = self.tm.to_opencl()
        self.set_prior('tc', 'NP', self.zero_epoch.n, 2*self.zero_epoch.s)
        self.set_prior('p', 'NP', self.period.n, 2*self.period.s)
        self.set_prior('rho', 'UP', 1, 35)
        self.set_prior('k2_app', 'UP', round(0.5*self.toi.depth[0]*1e-6, 5), round(1.5*self.toi.depth[0]*1e-6, 5))
        self.set_prior('k2_true', 'UP', 0.02 ** 2, 0.95 ** 2)
        self.set_prior('k2_app_tess', 'UP', round(0.5*self.toi.depth[0]*1e-6, 5), round(1.5*self.toi.depth[0]*1e-6, 5))
        self.set_prior('teff_h', 'NP', self.star.teff.n, self.star.teff.s)
        self.set_prior('teff_c', 'UP', 2500, 12000)

    def set_ldtk_priors(self):
        from ldtk import tess
        star = self.star
        m2_pbs = [pb for pb in self.passbands if pb != 'tess']
        filters = [tess] + get_ldtk_filters(m2_pbs)
        sc = LDPSetCreator((star.teff.n, star.teff.s), (star.logg.n, star.logg.s), (star.z.n, star.z.s), filters)
        ps = sc.create_profiles(500)
        ps.set_uncertainty_multiplier(5)
        ldc, lde = ps.coeffs_p2mp()
        for i, p in enumerate(self.ps[self._sl_ld]):
            self.set_prior(p.name, 'NP', round(ldc.flat[i], 5), round(lde.flat[i], 5))

    def create_pv_population(self, npv: int = 50) -> ndarray:
        pvp = super().create_pv_population(npv)
        for p in self.ps[self._sl_lm]:
            if 'lm_i' in p.name:
                pvp[:, p.pid] = 0.01 * (pvp[:, p.pid] - 1.0) + 1.0
            else:
                pvp[:, p.pid] *= 0.01
        return pvp

    def transit_model(self, pvp):
        pvp = atleast_2d(pvp)
        cnt = zeros((pvp.shape[0], self.npb))
        zero_epoch = pvp[:,0] - self._tref
        period = pvp[:,1]
        smaxis = as_from_rhop(pvp[:, 2], period)
        inclination  = i_from_ba(pvp[:, 3], smaxis)
        radius_ratio = sqrt(pvp[:,5:6])
        ldc = pvp[:, self._sl_ld].reshape([-1, self.npb, 2])
        flux = self.tm.evaluate(radius_ratio, ldc, zero_epoch, period, smaxis, inclination)
        cnt[:, 0] = 1 - pvp[:, 8] / pvp[:, 5]
        cnref = 1. - pvp[:, 4] / pvp[:, 5]
        cnt[:, 1:] = self.cm.contamination(cnref, pvp[:, 6], pvp[:, 7])
        return contaminate(flux, cnt, self.lcids, self.pbids)

    def plot_folded_tess_transit(self, solution: str = 'de', pv: ndarray = None, binwidth: float = 1,
                                 plot_model: bool = True, plot_unbinned: bool = True, plot_binned: bool = True,
                                 xlim: tuple = None, ylim: tuple = None, ax=None, figsize: tuple = None):

        if pv is None:
            if solution.lower() == 'local':
                pv = self._local_minimization.x
            elif solution.lower() in ('de', 'global'):
                pv = self.de.minimum_location
            elif solution.lower() in ('mcmc', 'mc'):
                pv = self.posterior_samples().median().values
            else:
                raise NotImplementedError("'solution' should be either 'local', 'global', or 'mcmc'")

        if ax is None:
            fig, ax = subplots(figsize=figsize)
        else:
            fig, ax = None, ax

        ax.autoscale(enable=True, axis='x', tight=True)

        etess = self._ntess
        t = self.timea[:etess]
        fo = self.ofluxa[:etess]
        fm = squeeze(self.transit_model(pv))[:etess]
        bl = squeeze(self.baseline(pv))[:etess]

        phase = 24 * pv[1] * (fold(t, pv[1], pv[0], 0.5) - 0.5)
        sids = argsort(phase)
        phase = phase[sids]
        bp, bf, be = downsample_time(phase, (fo / bl)[sids], binwidth / 60)
        if plot_unbinned:
            ax.plot(phase, (fo / bl)[sids], 'k.', alpha=1, ms=2)
        if plot_binned:
            ax.errorbar(bp, bf, be, fmt='ko', ms=3)
        if plot_model:
            ax.plot(phase, fm[sids], 'k')
        setp(ax, ylim=ylim, xlim=xlim, xlabel='Time - T$_c$ [h]', ylabel='Normalised flux')

        if fig is not None:
            fig.tight_layout()
        return fig


    def plot_joint_marginals(self, figsize=None, nb=30, gs=25, with_contamination=False, **kwargs):
        df = self.posterior_samples()
        if with_contamination:
            xlabels = ['$\Delta$ T$_\mathrm{Eff}$ [K]', 'Apparent radius ratio', 'Ref. pb. contamination',
                   'TESS contamination', 'Impact parameter', 'Stellar density [g/cm$^3$]']
            return _jplot([df.teff_c-df.teff_h, df.k_app, df.cref, df.ctess, df.b, df.rho], df.k_true,
                      xlabels, 'True radius ratio', figsize, nb, gs, **kwargs)
        else:
            xlabels = ['$\Delta$ T$_\mathrm{Eff}$ [K]', 'Apparent radius ratio',
                       'Impact parameter', 'Stellar density [g/cm$^3$]']
            return _jplot([df.teff_c-df.teff_h, df.k_app, df.b, df.rho], df.k_true,
                          xlabels, 'True radius ratio', figsize, nb, gs, **kwargs)