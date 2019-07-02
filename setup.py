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

from setuptools import setup, find_packages

setup(name='MuSCAT2TA',
      version='0.1',
      description='Package for MuSCAT2 transit analysis.',
      long_description='Package for MuSCAT2 transit analysis.',
      author='Hannu Parviainen',
      author_email='hpparvi@gmail.com',
      url='https://github.com/hpparvi/muscat2ta',
      packages=find_packages(),
      install_requires=["numpy", "scipy>=0.16", "astropy", "tqdm", "traitlets", 'pandas', 'xarray', 'photutils', 'matplotlib', 'astroquery',
                        'corner', 'seaborn', 'numba', 'george', 'pytransit', 'ldtk', 'uncertainties'],
      include_package_data=True,
      scripts=['bin/m2organize', 'bin/m2photometry', 'bin/m2astrometry', 'bin/m2nbtemplate', 'bin/m2init'],
      license='GPLv2',
      classifiers=[
          "Topic :: Scientific/Engineering",
          "Intended Audience :: Science/Research",
          "Intended Audience :: Developers",
          "Development Status :: 5 - Production/Stable",
          "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
          "Operating System :: OS Independent",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3"
      ],
      keywords= 'astronomy astrophysics exoplanets'
      )
