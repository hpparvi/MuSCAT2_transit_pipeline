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

from difflib import get_close_matches
from pathlib import Path

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from pkg_resources import resource_filename

m2_catalog_file = Path(resource_filename('muscat2ph', '../data/m2_catalog.csv')).resolve()

def read_m2_catalog():
    with open(m2_catalog_file, 'r') as f:
        t = f.readlines()
        names = t[0].lower().replace('decl','dec').strip().split(',')
        names.remove('comments')
        ccol = 9
        ncols = len(names)
        ids = []
        targets = []
        for i,l in enumerate(t[1:]):
            items = l.strip().strip('"').split(';')
            items = items[:ccol] + items[-(ncols-ccol):]
            items[1] = items[1].lower()
            ids.append(items[0])
            targets.append(items[1:])
        return pd.DataFrame(targets, index=ids, columns=names[1:])

def get_m2_coords(name):
    cat = read_m2_catalog()
    name = get_close_matches(name.lower(), cat.name, 1)[0]
    target = cat[cat.name==name]
    return SkyCoord(float(target.ra), float(target.dec), frame='fk5', unit=(u.deg, u.deg))
