from setuptools import setup, find_packages

setup(name='MuSCAT2TA',
      version='0.1',
      description='Package for MuSCAT2 transit analysis.',
      long_description='Package for MuSCAT2 transit analysis.',
      author='Hannu Parviainen',
      author_email='hpparvi@gmail.com',
      url='https://github.com/hpparvi/muscat2ta',
      packages=find_packages(),
      install_requires=["numpy", "scipy>=0.16", "astropy", "tqdm", "traitlets", 'pandas'],
      include_package_data=True,
      scripts=['bin/m2organize', 'bin/m2photometry', 'bin/m2astrometry', 'bin/m2fit'],
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
