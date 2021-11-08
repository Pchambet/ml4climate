from setuptools import setup

name = 'machine_learning_for_climate_and_energy'
reqs = ['bokeh >=2.2.3', 'hvplot', 'matplotlib >=3.2.2', 'netcdf4', 'numpy',
        'pandas >=1.0.5', 'panel >=0.10.3', 'scipy', 'scikit-learn >=1.0',
        'xarray']

setup(name=name, install_requires=reqs)
