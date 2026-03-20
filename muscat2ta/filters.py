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

from pytransit.contamination.filter import sdss_g, sdss_r, sdss_i, sdss_z
from pytransit.contamination.filter import BoxcarFilter as PTBoxcarFilter

ALL_PASSBANDS = ('g', 'r', 'i', 'z_s', 'na_d', 'g_narrow', 'i_narrow', 'z_narrow')

# Pytransit filter objects for contamination modelling
# ----------------------------------------------------
# Broadband filters use the standard SDSS definitions from pytransit.
# Narrow-band filters use BoxcarFilter with approximate wavelength ranges in nm.

_na_d_pt = PTBoxcarFilter('na_d', 585, 600)
_g_narrow_pt = PTBoxcarFilter('g_narrow', 480, 520)
_i_narrow_pt = PTBoxcarFilter('i_narrow', 740, 760)
_z_narrow_pt = PTBoxcarFilter('z_narrow', 850, 870)

PYTRANSIT_FILTERS = {
    'g': sdss_g,
    'r': sdss_r,
    'i': sdss_i,
    'z_s': sdss_z,
    'na_d': _na_d_pt,
    'g_narrow': _g_narrow_pt,
    'i_narrow': _i_narrow_pt,
    'z_narrow': _z_narrow_pt,
}


def get_ldtk_filters(passbands):
    """Return ldtk filter objects for the given passband names.

    Parameters
    ----------
    passbands : iterable of str
        Passband names (e.g., ('g', 'r', 'na_d')).

    Returns
    -------
    list
        List of ldtk filter objects.
    """
    from ldtk import sdss_g as lg, sdss_r as lr, sdss_i as li, sdss_z as lz
    from ldtk import BoxcarFilter as LDBoxcarFilter

    ldtk_broadband = {
        'g': lg,
        'r': lr,
        'i': li,
        'z_s': lz,
    }

    ldtk_narrowband = {
        'na_d': LDBoxcarFilter('na_d', 585, 600),
        'g_narrow': LDBoxcarFilter('g_narrow', 480, 520),
        'i_narrow': LDBoxcarFilter('i_narrow', 740, 760),
        'z_narrow': LDBoxcarFilter('z_narrow', 850, 870),
    }

    filters = []
    for pb in passbands:
        if pb in ldtk_broadband:
            filters.append(ldtk_broadband[pb])
        elif pb in ldtk_narrowband:
            filters.append(ldtk_narrowband[pb])
        else:
            raise ValueError(f"Unknown passband '{pb}' for ldtk filter lookup.")
    return filters
