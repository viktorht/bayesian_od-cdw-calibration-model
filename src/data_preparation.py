"""Provides functions prepare_data_x.

These functions should take in a dataframe of measurements and return a
PreparedData object.

Note that you can change the input arbitrarily - for example if you want to take
in two dataframes, a dictionary etc. However in this case you will need to edit
the corresponding code in the file prepare_data.py accordingly.

"""

from functools import partial
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.prepared_data import PreparedData
from src.util import (
    CoordDict,
    StanInput,
    check_is_df,
    make_columns_lower_case,
    stanify_dict,
)

NEW_COLNAMES = {'CDW (g/L)' : 'y',
                'OD' : 'x',
                'Dilution name': 'Dilution_name'
            }


N_CV_FOLDS = 2
DIMS = {
    "b": ["covariate"],
    "y": ["observation"],
    "yrep": ["observation"],
    "llik": ["observation"],
}

def prepare_data_means_model(measurements_raw: pd.DataFrame) -> PreparedData:
    """Prepare data cdw measurements"""
    x_cols = ['x']
    measurements_cdw = process_measurements_cdw_mean_model(measurements_raw['cdw'])
    measurements_od = process_measurements_od_mean_model(measurements_raw['od'])

    measurements = pd.merge(measurements_cdw, measurements_od, on = 'Dilution_name')

    return PreparedData(
        name="mean_model",
        coords=CoordDict(
            {"covariate": x_cols, "observation": measurements.index.tolist()}
        ),
        dims=DIMS,
        measurements=measurements,
        number_of_cv_folds=N_CV_FOLDS,
        stan_input_function=partial(get_stan_input, x_cols=x_cols),
    )

def process_measurements_cdw_mean_model(measurements: pd.DataFrame) -> pd.DataFrame:
    """Process CDW measurements to mean model.
    Cleans up df and calculate the mean values
    """
    out = (measurements
        .rename(columns=NEW_COLNAMES)
        .filter(['Dilution_name', 'y'])
        .groupby('Dilution_name')
        .agg('mean')
    ).copy()

    check_is_df(out)
    return out


def process_measurements_od_mean_model(measurements: pd.DataFrame) -> pd.DataFrame:
    """Process od measurements to mean model.
    Cleans up df and calculate the mean values
    """
    out = (measurements
        .rename(columns=NEW_COLNAMES)
        .filter(['Dilution_name', 'x'])
        .groupby('Dilution_name')
        .agg('mean')
    ).copy()

    check_is_df(out)
    return out


def get_stan_input(
    measurements: pd.DataFrame,
    x_cols: List[str],
    likelihood: bool,
    train_ix: List[int],
    test_ix: List[int],
) -> StanInput:
    """Turn a processed dataframe into a Stan input."""
    return stanify_dict(
        {
            "N": len(measurements),
            "N_train": len(train_ix),
            "N_test": len(test_ix),
            "K": len(x_cols),
            "x": measurements[x_cols],
            "y": measurements["y"],
            "likelihood": int(likelihood),
            "ix_train": [i + 1 for i in train_ix],
            "ix_test": [i + 1 for i in test_ix],
            "y": measurements["y"],
            "likelihood": int(likelihood),
        }
    )


def get_stan_inputs(
    prepared_data: PreparedData,
) -> Tuple[StanInput, StanInput, List[StanInput]]:
    """Get Stan input dictionaries for all modes from a PreparedData object."""
    ix_all = list(range(len(prepared_data.measurements)))
    stan_input_prior, stan_input_posterior = (
        prepared_data.stan_input_function(
            measurements=prepared_data.measurements,
            train_ix=ix_all,
            test_ix=ix_all,
            likelihood=likelihood,
        )
        for likelihood in (False, True)
    )
    stan_inputs_cv = []
    kf = KFold(prepared_data.number_of_cv_folds, shuffle=True)
    for train, test in kf.split(prepared_data.measurements):
        stan_inputs_cv.append(
            prepared_data.stan_input_function(
                measurements=prepared_data.measurements,
                likelihood=True,
                train_ix=list(train),
                test_ix=list(test),
            )
        )
    return stan_input_prior, stan_input_posterior, stan_inputs_cv
