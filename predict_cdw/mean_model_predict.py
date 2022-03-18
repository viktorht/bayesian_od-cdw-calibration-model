
import os

import arviz as az
import numpy as np
import pandas as pd
import xarray

from matplotlib import pyplot as plt


RESULTS_DIR = os.path.join("results", "runs")
MODEL_NANE = "mean_model"
posterior_file = os.path.join(RESULTS_DIR, MODEL_NANE, "posterior.nc")

od = np.array([1, 60, 50, 100])

posterior = az.from_netcdf(posterior_file)

def posterior_predictive_dist(od: np.array)->np.ndarray:
    assert od.shape.__len__() == 1, 'od array is not 1 dimentional'

    od = od.flatten() # has no effect arary is already flat

    beta1 = posterior.posterior['beta1'].values.reshape(-1, 1)
    beta2 = posterior.posterior['beta2'].values.reshape(-1, 1)

    return beta1 + beta2 * od

