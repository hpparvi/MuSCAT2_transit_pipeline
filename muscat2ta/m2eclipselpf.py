#  MuSCAT2 photometry and transit analysis pipeline
#  Copyright (C) 2020  Hannu Parviainen
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
from typing import Union, Optional

from numpy import floor, atleast_2d, where, diff, inf, all, arange
from pandas import Categorical
from pytransit import LinearModelBaseline
from pytransit.lpf.eclipselpf import EclipseLPF

from muscat2ta.m2mnlpf import read_reduced_m2


class M2EclipseLPF(EclipseLPF):
    def __init__(self, name: str, datadir: Union[str, Path] = 'results', pattern: str = '*.fits',
                 downsample: Optional[float] = None, model_baseline: bool = True):
        times, fluxes, pbs, wns, covs, vars, residuals = read_reduced_m2(datadir, pattern=pattern,
                                                                         downsample=downsample)
        pbs = Categorical(pbs, categories='g r i z_s'.split(), ordered=True).remove_unused_categories()
        pbnames = pbs.categories.values
        pbids = pbs.codes
        wnids = arange(pbids.size)
        tref = floor(times[0].min())
        self._model_baseline = model_baseline
        super().__init__(name, pbnames, times, fluxes, pbids=pbids, covariates=covs, wnids=wnids, tref=tref)

    def _post_initialisation(self):
        super()._post_initialisation()

        # Force the flux ratio to be monotonically increasing
        # ---------------------------------------------------
        # Add a prior that forces the planet-star flux ratio to increase
        # towards red wavelengths monotonically. This should be the case
        # always when dealing with secondary eclipses by a planet.
        def frprior(pv):
            pv = atleast_2d(pv)
            return where(all(diff(pv[:, self._sl_fr]) > 0., 1), 0, -inf)

        self.add_prior(frprior)

        # Force a circular orbit by default
        # ---------------------------------
        # Here we set tight zero-centered priors on sqrt(e) cos(w) and
        # sqrt(e) sin(w) to force a circular orbit.  These priors should
        # be overridden if we either have good estimates for e and w, or
        # if we there is a high chance that the orbit is not circular
        # (which actually is the case very often).
        self.set_prior('secw', 'NP', 0.0, 1e-6)
        self.set_prior('sesw', 'NP', 0.0, 1e-6)

    def _init_baseline(self):
        if self._model_baseline:
            self._add_baseline_model(LinearModelBaseline(self))