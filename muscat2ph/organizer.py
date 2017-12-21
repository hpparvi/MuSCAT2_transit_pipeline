from os.path import basename
from shutil import copy
from pathlib import Path

import logging

import astropy.io.fits as pf

from tqdm import tqdm

caltypes = {'dark': 'dark',
            'flat': 'flat',
            'domeflat': 'flat'}


class M2NightOrganizer:
    def __init__(self, obsdir, rootdir):
        self.obsdir = od = Path(obsdir)
        self.rootdir = rd = Path(rootdir)

        if (not od.exists()) or (not rd.exists()):
            raise FileNotFoundError

        self.files = sorted(self.obsdir.glob('*.fits'))

        self.orgdir = rd.joinpath(od.name)
        self.objdir = self.orgdir.joinpath('obj')
        self.caldir = self.orgdir.joinpath('calibs')

        logging.info('Found %i files', len(self.files))
        logging.info('Original directory: %s', self.orgdir.absolute())
        logging.info('Object directory: %s', self.objdir.absolute())
        logging.info('Calibration directory: %s', self.caldir.absolute())

        for p in (self.orgdir, self.objdir, self.caldir):
            p.mkdir(exist_ok=True)

    def create_path(self, f):
        h = pf.getheader(f)
        obj = h['object'].lower()
        flt = h['filter'].lower()
        if obj in caltypes.keys():
            return self.caldir.joinpath(caltypes[obj], flt)
        else:
            return self.objdir.joinpath(obj, flt)

    def organize(self, dry_run=False, overwrite=True):
        if not dry_run:
            for f in tqdm(self.files, desc='Organizing files'):
                try:
                    npath = self.create_path(f)
                    if overwrite or not npath.exists():
                        npath.mkdir(parents=True, exist_ok=True)
                        copy(str(f), str(npath))
                except IOError:
                    logging.warning("Warning: skipping a corrupted file %s",str(f))

