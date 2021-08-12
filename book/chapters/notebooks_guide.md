# Notebooks Guide

## General guidelines for MEC666

  - Each session includes a course chapter (correspondant: Philippe Drobinski) and a tutorial (correspondant: Alexis Tantet).
  - We use Moodle to share course/tutorial material with you (course book, this guide, links, etc.).
  - Some of the tutorials are based on notebooks to be run interactively on the LMD's JupyterHub:
    - Instructions on how to run the notebooks interactively are given below (Section 2).
	- Your credentials to login to the JupyterHub have been sent to you by e-mail.
	- If not, ask your correspondant before the tutorial.
	- A link to a webpage corresponding to the notebook accessible by anyone but without interaction is also provided on the session's section on Moodle.
  - Otherwise, a PDF of the session's tutorial is provided on Moodle.
    Please download it before the tutorial.

## Getting a notebook ready for the tutorial on the JupyterHub from IPSL

To get the notebooks on the JupyterHub ready for the tutorials, please follow the instructions bellow with the help of the following figures:

  - Log in to your account at [https://www1.lmd.polytechnique.fr/jupyterhub/](https://www1.lmd.polytechnique.fr/jupyterhub/).
	This may take a few minutes (refresh the page from time to time to make sure that you are redirected to the hub).

:::{figure} logging_in
<img src="../images/notebooks_guide/login.png" alt="Logging in" width="800px">

Logging in.
:::

  - Open a Terminal from the launcher (if needed, choose `File/New Launcher` in the top menu).

:::{figure} open_terminal
<img src="../images/notebooks_guide/open_terminal.png" alt="Open a terminal" width="800px">

Open a terminal.
:::


  - Ensure that you are in the home directory by entering this command in the terminal:

```
cd ~
```

  - Enter this command in the terminal to install the Python packages (it may take a while):

```
pip install --src=. -e git+https://gitlab.in2p3.fr/energy4climate/public/education/climate_change_and_energy_transition.git#egg=simclimat
```

:::{figure} install_packages
<img src="../images/notebooks_guide/install_packages.png" alt="Install pakages" width="800px">

Install Python packages.
:::

<!--   - Enter this command in the terminal to activate the pyviz jupyterlab extension (this may take a while): -->

<!-- ``` -->
<!-- jupyter labextension install @pyviz/jupyterlab_pyviz -->
<!-- ``` -->

<!-- :::{figure} install_extensions -->
<!-- <img src="../images/notebooks_guide/install_extension.png" alt="Install extensions" width="800px"> -->

<!-- Install `pyviz` extension for jupyterlab. -->
<!-- ::: -->

<!--   - Log out (`File/Log Out` in the top menu) and log in again. -->
  - Open the notebook by following the path `simclimat/book/notebooks/**/*.ipynb` in the side bar on the left (replace the stars by the directory and the name of the notebook).

:::{figure} open_notebook
<img src="../images/notebooks_guide/open_notebook.png" alt="Open a notebook" width="800px">

Open a notebook.
:::

  - `Run/Restart Kernel and Run All Cells` to run all cells while making sure that the last installation of the package is used.
  - Select the first cell and type `Shift Enter` or press the `Play` button in the toolbar repeatedly to execute one cell after the other.

## Getting a notebook ready for the tutorial on your machine using `conda`

To get the notebooks on your computer ready for the tutorials using `conda`, please make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual) for Python 3 installed on your machine and follow the instructions bellow:

  - Use one of the following methods to get the package on your machine:
    - Download with you browser using this link as shown on the following figure and decompress the package: [https://gitlab.in2p3.fr/energy4climate/public/education/climate_change_and_energy_transition](https://gitlab.in2p3.fr/energy4climate/public/education/climate_change_and_energy_transition)
    - Clone the package using `git` by entering the following command in a terminal:

```
git clone https://gitlab.in2p3.fr/energy4climate/public/education/climate_change_and_energy_transition.git
```

:::{figure} open_notebook
<img src="../images/notebooks_guide/download_with_browser.png" alt="Download package with browser" width="800px">

Download the package with your browser.
:::

  - Create an automatic conda environment for the tutorials:

```
conda env create -f environment.yml
```

  - Activate the environment:

```
conda activate climate_energy_tutorials
```

  - Install `jupyter`:

```
conda install jupyter
```

  - Run `jupyter` in the `book/notebooks/` directory:

```
jupyter notebook book/notebooks/
```
