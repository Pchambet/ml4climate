from setuptools import setup

name = 'machine_learning_for_climate_and_energy'
reqs = ['bokeh >=2.2.3', 'holoviews >=1.14.9', 'hvplot', 'matplotlib',
        'netcdf4 ==1.5.6', 'numpy', 'pandas >=1.2.4', 'panel >=0.10.3',
        'scipy', 'scikit-learn >=1.0', 'xarray']

setup(name=name, install_requires=reqs)
