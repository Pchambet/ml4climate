# Import modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
import xarray as xr

# Data directory
DATA_DIR = Path('data')

# Filename to geopotential height at 500hPa from MERRA-2 reanalysis
filename_z500 = 'merra2_analyze_height_500_month_19800101-20190101.nc'
filename_u10 = 'merra2_area_selection_output_zonal_wind_merra2_2010-2019_daily.csv'
filename_v10 = 'merra2_area_selection_output_meridional_wind_merra2_2010-2019_daily.csv'

# Read geopotential height dataset with xarray
filepath = Path(DATA_DIR, filename_z500)
ds = xr.load_dataset(filepath)

# Select geopotential height variable
z500_label = list(ds)[0]
da_z500_hr = ds[z500_label]

# Downsample
N_GRID_AVG = 4
da_z500 = da_z500_hr.coarsen(lat=N_GRID_AVG, boundary='trim').mean().coarsen(
    lon=N_GRID_AVG, boundary='trim').mean()

REMOVE_SEASONAL_CYCLE = True
if REMOVE_SEASONAL_CYCLE:
    gp_z500_cycle = da_z500.groupby('time.month')
    da_z500_anom = gp_z500_cycle - gp_z500_cycle.mean('time')
else:
    da_z500_anom = da_z500

# Convert to bandas with grid points as columns
df_z500_anom = da_z500_anom.stack(latlon=('lat', 'lon')).to_dataframe()[
    z500_label].unstack(0).transpose()

# Read wind data with pandas
read_csv_kwargs = dict(index_col=0, parse_dates=True)
filepath = Path(DATA_DIR, filename_u10)
df_u_daily = pd.read_csv(filepath, **read_csv_kwargs)
filepath = Path(DATA_DIR, filename_v10)
df_v_daily = pd.read_csv(filepath, **read_csv_kwargs)

# Compute wind speed
df_wind_daily = np.sqrt(df_u_daily**2 + df_v_daily**2)

# Resample to monthly
df_wind = df_wind_daily.resample('MS').mean()

# Select region
# region_name = 'Bretagne'
region_name = 'National'
if region_name == 'National':
    df_wind_reg = df_wind.mean('columns')
    df_wind_reg.name = region_name
else:
    df_wind_reg = df_wind[region_name]

if REMOVE_SEASONAL_CYCLE:
    da_wind_reg = df_wind_reg.to_xarray()
    gp_wind_cycle = da_wind_reg.groupby('time.month')
    da_wind_anom = gp_wind_cycle - gp_wind_cycle.mean('time')
    df_wind_anom = da_wind_anom.drop('month').to_dataframe()[region_name]
else:
    df_wind_anom = df_wind_reg

# Select common index
idx = df_z500_anom.index.intersection(df_wind_anom.index)
df_z500_anom = df_z500_anom.loc[idx]
df_wind_anom = df_wind_anom.loc[idx]

# Number of years in dataset
time = df_wind_anom.index
n_years = time.year.max() - time.year.min() + 1

# Estimator choice
reg_class = linear_model.Ridge

# Configure estimator
fit_intercept = True

# Prepare input and target
X = df_z500_anom.values
y = df_wind_anom.values

# Set number of splits for cross-validation
N_SPLITS = n_years

# Define array of regularization-parameter values
ALPHA_RNG = np.logspace(3, 8, 20)

# Declare empty arrays in which to store r2 scores and coefficients
r2_validation = np.empty(ALPHA_RNG.shape)
coefs = np.empty((len(ALPHA_RNG), X.shape[1]))

# Estimator choice
reg_class = linear_model.Ridge

# Loop over regularization-parameter values
for k, alpha in enumerate(ALPHA_RNG):
    # Define the Ridge estimator for particular regularization-parameter value
    reg = reg_class(alpha=alpha, fit_intercept=fit_intercept)

    # Get r2 test scores from k-fold cross-validation
    r2_validation_arr = cross_val_score(reg, X, y, cv=N_SPLITS)

    # Get r2 expected prediction score by averaging over test scores
    r2_validation[k] = r2_validation_arr.mean()

    # Save coefficients
    reg.fit(X, y)
    coefs[k] = reg.coef_

# Get best value of the regularization parameter
i_best = np.argmax(r2_validation)
alpha_best = ALPHA_RNG[i_best]
r2_best = r2_validation[i_best]

# Plot validation curve
alpha_label = r'$\alpha$ (°C2/MWh)'
plt.figure()
plt.plot(ALPHA_RNG, r2_validation)
plt.xscale('log')
plt.xlabel(alpha_label)
plt.ylabel(r'$r^2$')
plt.ylim(0., 1.)
title = (region_name +
         r'. Best $r^2$: {:.2} for $\alpha$ = {:.1e}'.format(
             r2_best, alpha_best))
_ = plt.title(title)

plt.show(block=False)
