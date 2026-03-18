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

import ssl
from collections import namedtuple
from difflib import get_close_matches
from importlib.resources import files
from pathlib import Path
from typing import Optional

import astropy.units as u
import pandas as pd
import httpx
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.simbad import Simbad
from numpy import remainder, array, squeeze

TOI = namedtuple('TOI', 'tic toi tmag ra dec epoch period duration depth'.split())


def _get_data_path(filename: str) -> Path:
    """Get the path to a data file in the muscat2ph.data package."""
    return Path(str(files("muscat2ph.data").joinpath(filename)))


m2_catalog_file = _get_data_path("m2_catalog.csv")
toi_catalog_file = _get_data_path("toi_catalog.csv")


def update_m2_catalog(password: str) -> None:
    login_url = 'https://research.iac.es/proyecto/muscat/users/login'
    csv_url = 'https://research.iac.es/proyecto/muscat/stars/export'
    data = {'username': 'observer', 'password': password}

    context = ssl.create_default_context()
    context.set_ciphers('DEFAULT:@SECLEVEL=1')

    with httpx.Client(verify=context) as client:
        client.post(login_url, data=data)
        result = client.get(csv_url)

    if "access not permitted" in result.text.lower():
        raise ValueError("Wrong password")

    with open(m2_catalog_file, "w") as fout:
        fout.write(result.text)


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
    df = pd.read_csv(m2_catalog_file, sep=";", quotechar='"', on_bad_lines='skip')
    df.columns = [s.lower() for s in df.columns]
    df['name'] = df.name.str.lower()
    return df


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
    return SkyCoord(float(target.ra.values[0]), float(target.decl.values[0]), frame='fk5', unit=(u.deg, u.deg))


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


def get_coords(target: str, obsdate: Optional[Time] = None):
    try:
        simbad = Simbad()
        simbad.add_votable_fields('pmra')
        simbad.add_votable_fields('pmdec')
        try:
            tbl = simbad.query_object(target)
        except Exception:
            tbl = None
            
        if tbl is None:
            raise KeyError
        else:
            tbl = tbl.filled(0.0)
            coo = squeeze(SkyCoord(tbl['RA'], tbl['DEC'], unit=(u.hourangle, u.deg)))
            if obsdate is None:
                return coo
            else:
                epoch = Time('2000-01-01')
                dt = (obsdate - epoch).to(u.yr)
                ra = coo.ra + dt * tbl['PMRA'].data * u.mas / u.yr
                dec = coo.dec + dt * tbl['PMDEC'].data * u.mas / u.yr
                return squeeze(SkyCoord(ra, dec))
    except KeyError:
        try:
            return get_toi_or_tic_coords(parse_toi(target))
        except ValueError:
            return get_m2_coords(target)
