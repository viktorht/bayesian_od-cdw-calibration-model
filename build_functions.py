#%%
from src.data_preparation import *


# %%
cdw = pd.read_csv('data/raw/od_cdw_calibration-cdw.csv', index_col=0)
cdw.head()

NEW_COLNAMES = {'CDW (g/L)' : 'CDW', 
                #'CDW rel error (g/L)' : 'CDW_relerror', 
                #'Volume of dilution rel error (uL)' : 'volume_relerror'
            }

DROP_COLS = [
    'CDW rel error (g)',
    'CDW rel error (g/L)', 
    'Volume of dilution rel error (uL)',
    'CDW_replicate'
    ]

# %%
def prepare_data_cdw(measurements_raw: pd.DataFrame) -> PreparedData:
    """Prepare data cdw measurements"""
    x_cols = ["x1", "x2", "x1:x2"]
    measurements = process_measurements(measurements_raw)
    return PreparedData(
        name="interaction",
        coords=CoordDict(
            {"covariate": x_cols, "observation": measurements.index.tolist()}
        ),
        dims=DIMS,
        measurements=measurements,
        number_of_cv_folds=N_CV_FOLDS,
        stan_input_function=partial(get_stan_input, x_cols=x_cols),
    )

def process_measurements_cdw(measurements: pd.DataFrame) -> pd.DataFrame:
    """Process the measurements.

    This is to illustrate how you might want to do common table manipulation
    tasks like filtering, changing column names and adding new columns.

    Note that if you want, you can use different measurement processing
    functions for different prepare_data functions

    Contains check_is_df a lot because many pandas methods have return signatures
    including None, but we want to raise an error unless a DataFrame is returned.

    """
    out = check_is_df(
        check_is_df(
            check_is_df(measurements.rename(columns=NEW_COLNAMES))
            .drop(columns=DROP_COLS, axis=0)
        )
    ).copy()

    return out

# %%
