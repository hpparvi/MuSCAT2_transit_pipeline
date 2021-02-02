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
from typing import Union, Optional


class M2ObservationNight:

    def __init__(self, root: Union[Path, str], obj: Optional[str] = None, passbands=None):
        self.root = Path(root).resolve()
        self.date = self.root.absolute().name

        if passbands == 'all' or passbands is None:
            self.pbs = 'r g i z_s'.split()
        else:
            self.pbs = passbands

        if obj:
            self.objects = [obj]
        else:
            self.objects = [o.name for o in list(root.joinpath('obj').glob('*'))]


class M2ObservationData:

    def __init__(self, night, obj):
        self.night = night
        self.obj = obj

        self.files = {}
        self.files_with_wcs = {}

        for pb in night.pbs:
            ddir = night.root.joinpath('obj', obj, pb)
            if ddir.exists():
                self.files[pb] = sorted(list(ddir.glob('MCT2?_*.fits')))
                self.files_with_wcs[pb] = list(filter(lambda f: f.with_suffix('.wcs').exists(), self.files[pb]))
        self.pbs = list(self.files.keys())

        if sum([len(f) for f in self.files.values()]) == 0:
            logging.warning("No files to process")