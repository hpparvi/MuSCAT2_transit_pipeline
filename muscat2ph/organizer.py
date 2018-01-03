import logging
import astropy.io.fits as pf

from shutil import copy
from pathlib import Path
from numpy import array, in1d
from tqdm import tqdm

caltypes = {'dark': 'dark',
            'flat': 'flat',
            'domeflat': 'flat'}

M2FSIZE = 2105280 # Standard MuSCAT2 file size

class M2NightOrganizer:
    def __init__(self, obsdir, rootdir, skip_existing=False):
        self.obsdir = od = Path(obsdir)
        self.rootdir = rd = Path(rootdir)

        if (not od.exists()) or (not rd.exists()):
            raise FileNotFoundError

        self.orgdir = rd.joinpath(od.name)
        self.objdir = self.orgdir.joinpath('obj')
        self.caldir = self.orgdir.joinpath('calibs')

        logging.info('Original directory: %s', self.orgdir.absolute())
        logging.info('Object directory: %s', self.objdir.absolute())
        logging.info('Calibration directory: %s', self.caldir.absolute())

        for p in (self.orgdir, self.objdir, self.caldir):
            p.mkdir(exist_ok=True)

        self.files, self.corrupted = self.gather_files(skip_existing)


    def gather_files(self, skip_existing=False):
        files = array(sorted(self.obsdir.glob('*.fits')))
        sizes = array([f.stat().st_size for f in files])
        cmask = sizes != M2FSIZE
        corrupted = files[cmask]
        files = files[~cmask]
        logging.info('Found %i good files and %i corrupted files', len(files), len(corrupted))

        if skip_existing:
            n_raw = array(sorted([l.name for l in files]))
            n_existing = array(sorted([l.name for l in self.orgdir.rglob('*.fits')]))
            emask = ~in1d(n_raw, n_existing)
            files = files[emask]
            logging.info('Skipped existing %i files', (~emask).sum())

        return list(files), list(corrupted)


    def create_path(self, f):
        h = pf.getheader(f)
        obj = h['object'].lower()
        flt = h['filter'].lower()
        if obj in caltypes.keys():
            return self.caldir.joinpath(caltypes[obj], flt)
        else:
            return self.objdir.joinpath(obj, flt)


    def organize(self):
        for f in tqdm(self.files, desc='Organizing files'):
            ndir = self.create_path(f)
            if not ndir.exists():
                ndir.mkdir(parents=True)
            copy(str(f), str(ndir.joinpath(f.name)))


    def check(self, verbose=False):
        cfiles = array(sorted(self.orgdir.rglob("*.fits")))
        logging.info('Final check: found %i good files in raw directory and %i files under the organized tree',
                     len(self.files), len(cfiles))
        if verbose:
            print('\n m2organize check')
            print(' ----------------')
            print(' - %i good fits files in the raw directory'%len(self.files))
            print(' - %i fits files under the organized tree\n'%len(cfiles))
            print(' - %i corrupted fits files in the raw directory\n'%len(self.corrupted))

