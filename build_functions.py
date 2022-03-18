#%%
from src.data_preparation import *


# %%
cdw = pd.read_csv('data/raw/od_cdw_calibration-cdw.csv', index_col=0)
cdw.head()

NEW_COLNAMES = {'CDW (g/L)' : 'CDW',
                'Dilution name': 'Dilution_name'
                #'CDW rel error (g/L)' : 'CDW_relerror', 
                #'Volume of dilution rel error (uL)' : 'volume_relerror'
            }

N_CV_FOLDS = 10
DIMS = {
    "b": ["covariate"],
    "y": ["observation"],
    "yrep": ["observation"],
    "llik": ["observation"],
}

# %%
def prepare_data_means_model(measurements_raw: pd.DataFrame) -> PreparedData:
    """Prepare data cdw measurements"""
    x_cols = ['OD']
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
        .filter(['Dilution_name', 'CDW'])
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
        .filter(['Dilution_name', 'OD'])
        .groupby('Dilution_name')
        .agg('mean')
    ).copy()

    check_is_df(out)
    return out
# %%
