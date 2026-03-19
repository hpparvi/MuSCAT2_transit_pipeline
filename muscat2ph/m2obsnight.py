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
import logging

from pathlib import Path
from typing import Union, Optional, List

from astropy.time import Time

glob_patterns = {'M1':'MSCT?_*.fits', 'M2':'MCT2?_*.fits', 'M3':'ogg2*fits.fz'}

BROADBAND_FILTERS = ['g', 'r', 'i', 'z_s']
NARROWBAND_FILTERS = ['na_d', 'g_narrow', 'i_narrow', 'z_narrow']
FILTER_SETS = {'broadband': BROADBAND_FILTERS, 'narrowband': NARROWBAND_FILTERS}

class M2ObservationNight:

    def __init__(self, root: Union[Path, str], obj: Optional[str] = None, passbands: Optional[List] = None):
        self.root = Path(root).resolve()
        self.night = self.root.absolute().name
        self.date = Time.strptime(self.night, '%y%m%d')
        if passbands is not None:
            self.pbs = passbands
        else:
            self.pbs = self._detect_filters()
        if obj:
            self.objects = [obj]
        else:
            self.objects = [o.name for o in list(self.root.joinpath('obj').glob('*'))]

    def _detect_filters(self) -> List[str]:
        """Auto-detect filter set by checking which filter subdirectories exist."""
        obj_dir = self.root / 'obj'
        if not obj_dir.exists():
            return BROADBAND_FILTERS
        subdirs = set()
        for obj_path in obj_dir.iterdir():
            if obj_path.is_dir():
                for child in obj_path.iterdir():
                    if child.is_dir():
                        subdirs.add(child.name)
        if subdirs & set(NARROWBAND_FILTERS):
            return NARROWBAND_FILTERS
        return BROADBAND_FILTERS

class M2ObservationData:

    def __init__(self, night, obj):
        self.night = night
        self.obj = obj

        self.instrument = None
        self.files = {}
        self.files_with_wcs = {}

        for pb in night.pbs:
            ddir = night.root.joinpath('obj', obj, pb)
            if ddir.exists():
                for instrument, pattern in glob_patterns.items():
                    files = sorted(list(ddir.glob(pattern)))
                    if files:
                        self.files[pb] = files
                        self.instrument = instrument
                        break
                self.files_with_wcs[pb] = list(filter(lambda f: f.with_suffix('.wcs').exists(), self.files[pb]))
        self.pbs = list(self.files.keys())

        if sum([len(f) for f in self.files.values()]) == 0:
            logging.warning("No files to process")