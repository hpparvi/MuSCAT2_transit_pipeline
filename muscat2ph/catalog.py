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
from collections import namedtuple

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from numpy import remainder, array
from pkg_resources import resource_filename

TOI = namedtuple('TOI', 'tic toi tmag ra dec epoch period duration depth'.split())

m2_catalog_file = Path(resource_filename('muscat2ph', '../data/m2_catalog.csv')).resolve()
toi_catalog_file = Path(resource_filename('muscat2ph', '../data/toi_catalog.csv')).resolve()


def update_toi_catalog(remove_fp: bool = False, remove_known_planets: bool = False) -> None:
    """Download TOI list from TESS Alert/TOI Release.

    Parameters
    ----------
    remove_fp : bool
        remove false positive from the catalog if `True`
    remove_known_planets: bool
        remove known planets from the catalog if `True`
    """
    dl_link = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
    fp = Path('toi_catalog.csv')

    d = pd.read_csv(dl_link)
    ntois = len(d)

    if remove_fp:
        d = d[d['TFOPWG Disposition'] != 'FP']
        print(f'Removed {ntois - len(d)} TOIs marked as FP')
        ntois = len(d)

    if remove_known_planets:
        planet_keys = ['WASP', 'SWASP', 'HAT', 'HATS', 'KELT', 'QATAR', 'K2', 'Kepler']
        keys = []
        for key in planet_keys:
            idx = ~array(d['Comments'].str.contains(key).tolist(), dtype=bool)
            d = d[idx]
            if idx.sum() > 0:
                keys.append(key)
        print(f'Removed {ntois - len(d)} TOIs marked as known planets')

    d.to_csv(toi_catalog_file, index=False)


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


def parse_toi(toi):
    try:
        toi = float(toi.lower().strip('toi')) if isinstance(toi, str) else float(toi)
        if abs(toi % 1) < 1e-6:
            toi += 0.01
        return toi
    except ValueError as e:
        raise ValueError(f'Cannot parse "{toi}" into a TOI number')


def get_toi(toi):
    toi = parse_toi(toi)
    df = pd.read_csv(toi_catalog_file, sep=',')
    try:
        dtoi = df[df.TOI == toi]

        zero_epoch = dtoi[['Epoch (BJD)', 'Epoch (BJD) err']].values[0]
        period = dtoi[['Period (days)', 'Period (days) err']].values[0]
        duration = dtoi[['Duration (hours)', 'Duration (hours) err']].values[0]
        depth = dtoi[['Depth (ppm)', 'Depth (ppm) err']].values[0]
        return TOI(*dtoi['TIC ID, TOI, TESS Mag, RA, Dec'.split(', ')].values[0], epoch=zero_epoch,
                   period=period, duration=duration, depth=depth)
    except IndexError:
        raise ValueError(f'Cannot find TOI {toi} from the catalog')


def get_toi_or_tic(toi_or_tic):
    df = pd.read_csv(toi_catalog_file, sep=',')
    if abs(remainder(toi_or_tic, 1)) < 1e-5:
        toi = float(df[df['TIC ID'] == int(toi_or_tic)]['TOI'])
    else:
        toi = toi_or_tic
    return get_toi(toi)

def get_m2_coords(name):
    """
    Returns the sky coordinates of a target based on the internal MuSCAT2 target catalog.

    Parameters
    ----------
    name

    Returns
    -------
    Astropy SkyCoord object
    """
    cat = read_m2_catalog()
    name = get_close_matches(name.lower(), cat.name, 1)[0]
    target = cat[cat.name==name]
    return SkyCoord(float(target.ra), float(target.dec), frame='fk5', unit=(u.deg, u.deg))


def get_toi_or_tic_coords(toi_or_tic):
    """
    Returns the sky coordinates of a TOI based on the TOI catalog.

    Parameters
    ----------
    toi

    Returns
    -------
    Astropy SkyCoord object
    """
    toi = get_toi_or_tic(toi_or_tic)
    return SkyCoord(toi.ra, toi.dec, frame='fk5', unit=(u.hourangle, u.deg))

def get_coords(target: str):
    try:
        return get_toi_or_tic_coords(parse_toi('toi00680.01'))
    except ValueError:
        return get_m2_coords(target)