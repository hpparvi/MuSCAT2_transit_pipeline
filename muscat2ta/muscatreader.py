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

import re
from glob import glob
from os.path import join

import pandas as pd
from matplotlib.pyplot import errorbar, plot
from numpy import median, sqrt, diff, inf, unique, argmin


def estimate_white_noise(flux):
    df = diff(flux)
    return median(abs(df - median(df))) * 1.4826 / sqrt(2.)


class MuSCATReader:
    allowed_filters = 'g r i z'.split()

    def __init__(self, datadir):
        self.ddir = datadir
        self.dfiles = sorted(glob(join(self.ddir, 'lcf*dat')))
        self.dfilters = [re.search('msct_([rgz])_', fn).group(1) for fn in self.dfiles]
        self.dfiles = {flt: [self.dfiles[i] for i, f in enumerate(self.dfilters) if f == flt]
                       for flt in unique(self.dfilters)}
        self.filters = self.dfiles.keys()
        self.filters = pd.Categorical(self.filters, categories=self.allowed_filters, ordered=True).sort_values()

        self.datasets = {}
        for flt, files in self.dfiles.items():
            dfs = [self.read_file(f) for f in files]
            df = dfs[0]['bjd airmass sky dx dy fwhm peak'.split()].copy()
            flux, ferr = self.select_light_curve(dfs)
            df['flux'] = flux
            df['ferr'] = ferr
            self.datasets[flt] = df

    def read_file(self, fname):
        df = pd.read_csv(fname, sep=' ', )
        df = pd.DataFrame(df.values[:, :-2], columns=df.columns[1:-1])
        df.iloc[:, 0] = df.iloc[:, 0] + float(df.columns[0].split('-')[1])
        df.drop('frame', 1, inplace=True)
        cols = 'bjd airmass sky dx dy fwhm peak'.split()
        for c in df.columns:
            if 'flux' in c:
                cols.append('flux_' + (re.search('r=(..)', c).group(1)))
            elif 'err' in c:
                cols.append('ferr_' + (re.search('r=(..)', c).group(1)))
        df.columns = cols
        for c in df.columns:
            df[c] = df[c].astype('d')
        return df

    def select_aperture(self, df):
        fluxes = [c for c in df.columns if 'flux' in c]
        ferrs = [c for c in df.columns if 'ferr' in c]
        wn = list(map(estimate_white_noise, df[fluxes].values.T))
        return (df[fluxes[argmin(wn)]].values.astype('d'),
                df[ferrs[argmin(wn)]].values.astype('d'),
                min(wn))
                min(wn))

    def select_light_curve(self, dfs):
        wn = inf
        for i, df in enumerate(dfs):
            f, e, wnt = self.select_aperture(df)
            if wnt < wn:
                flux, ferr, wn = f, e, wnt
        return flux, ferr

    def plot_dataset(self, flt, errors=False):
        df = self.datasets[flt]

        if errors:
            errorbar(df.bjd, df.flux, df.ferr)
        else:
            plot(df.bjd, df.flux)