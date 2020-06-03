import xarray as xa
import pandas as pd

from typing import Union
from pathlib import Path
from numpy import array

from .m2eclipselpf import M2EclipseLPF

def load_mcmc_samples(fname: Union[str,Path]) -> pd.DataFrame:
    with xa.load_dataset(fname) as ds:
        df = pd.DataFrame(array(ds.lm_mcmc).reshape([-1, ds.lm_mcmc.shape[-1]]), columns=ds.name)
    return df

__all__ = "M2EclipseLPF load_mcmc_samples".split()