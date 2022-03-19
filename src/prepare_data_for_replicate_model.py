# make stan data
import json
from cmdstanpy import write_stan_json
import pandas as pd
import os

OUTPUT_DIR = os.path.join('data','prepared','replicate_model')

## Preparing data
od = pd.read_csv(os.path.join('data', 'raw', 'od_cdw_calibration-od.csv'), index_col = 0)
cdw = pd.read_csv(os.path.join('data', 'raw', 'od_cdw_calibration-cdw.csv'), index_col = 0)

cdw_clean = cdw.query("`CDW rel error (g/L)` < 0.5")
od_clean = od.query("`Dilution name`.isin(@cdw_clean['Dilution name'])")

assert od_clean.shape[0] == cdw_clean.shape[0], 'There are not an equal amount of measurements'

# Organize stan input
stan_data = {
    'N' : od_clean.shape[0],
    'x_meas' : od_clean['OD'].to_numpy().tolist(),
    'y_meas' : cdw_clean['CDW (g/L)'].to_numpy().tolist(),
    'D' : od_clean['Dilution name'].nunique(),
    'likelihood' : 1,
    'dilution_x' : (pd.factorize(od_clean['Dilution name'])[0] + 1).tolist(), # stan does not accept 0 for categorical mapping therefore + 1
    'dilution_y' : (pd.factorize(cdw_clean['Dilution name'])[0] + 1).tolist(),
    'prior_mean_sigma_eps' : 2,
    'prior_std_sigma_eps' : 1,
    'sigma_x' : 0.5,
    'sigma_y' : 0.3 / 2
}

# output
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

write_stan_json(os.path.join(OUTPUT_DIR,'stan_input_posterior.json'), stan_data)
