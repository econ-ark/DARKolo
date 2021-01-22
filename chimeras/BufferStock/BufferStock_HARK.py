# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Buffer Stock Saving in HARK

# %% [markdown]
# This notebook constructs solutions to a standard buffer stock saving model obtained by the [Econ-ARK/HARK](https://github.com/econ-ark/HARK) toolkit

# %% {"code_folding": [0]}
# This cell does some setup and imports generic tools
# Much of the material below is unnecessary for present purposes;
# It is inherited from a notebook designed to work on many systems and accomplish many goals
# most of which are not relevant here
# For "ultimately I shoud delete these files purposes:" It was derived from BufferStock-HARK-vs-dolo.ipynb in llorracc/dolo/examples/notebooks

Generator = True  # Is this notebook the master or is it generated?
# Import related generic python packages
import numpy as np
from time import clock

mystr = lambda number: "{:.4f}".format(number)

# This is a jupytext paired notebook that autogenerates BufferStockTheory.py
# which can be executed from a terminal command line via "ipython BufferStockTheory.py"
# But a terminal does not permit inline figures, so we need to test jupyter vs terminal
# Google "how can I check if code is executed in the ipython notebook"

from IPython import get_ipython  # In case it was run from python instead of ipython


def in_ipynb():
    try:
        if (
            str(type(get_ipython()))
            == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>"
        ):
            return True
        else:
            return False
    except NameError:
        return False


# Determine whether to make the figures inline (for spyder or jupyter)
# vs whatever is the automatic setting that will apply if run from the terminal
if in_ipynb():
    # %matplotlib inline generates a syntax error when run from the shell
    # so do this instead
    get_ipython().run_line_magic("matplotlib", "inline")
else:
    get_ipython().run_line_magic("matplotlib", "auto")
    print("You appear to be running from a terminal")
    print("By default, figures will appear one by one")
    print("Close the visible figure in order to see the next one")

# Import the plot-figure library matplotlib

import matplotlib.pyplot as plt

# In order to use LaTeX to manage all text layout in our figures, we import rc settings from matplotlib.
from matplotlib import rc

plt.rc("font", family="serif")

# LaTeX is huge and takes forever to install on mybinder
# so if it is not installed then do not use it
from distutils.spawn import find_executable

iflatexExists = False
if find_executable("latex"):
    iflatexExists = True

plt.rc("font", family="serif")
plt.rc("text", usetex=iflatexExists)

# The warnings package allows us to ignore some harmless but alarming warning messages
import warnings

warnings.filterwarnings("ignore")

# The tools for navigating the filesystem
import sys
import os

sys.path.insert(
    0, os.path.abspath("../../lib")
)  # REMARKs directory is two down from root

from HARK.utilities import plotFuncsDer, plotFuncs
from copy import copy, deepcopy

# Define (and create, if necessary) the figures directory "Figures"
if Generator:
    my_file_path = os.path.dirname(
        os.path.abspath("BufferStockTheory.py")
    )  # Find pathname to this file:
    Figures_HARK_dir = os.path.join(
        my_file_path, "Figures/"
    )  # LaTeX document assumes figures will be here
    Figures_HARK_dir = os.path.join(
        my_file_path, "/tmp/Figures/"
    )  # Uncomment to make figures outside of git path
    if not os.path.exists(Figures_HARK_dir):
        os.makedirs(Figures_HARK_dir)

# %% [markdown]
# # HARK
# The [Econ-ARK/HARK](https://github.com/econ-ark/HARK) toolkit's solution to this problem is part of the $\texttt{ConsIndShockModel.py}$ module in the $\texttt{ConsumptionSaving}$ directory of tools.  For an introduction to this module, see the [ConsIndShockModel.ipynb](https://econ-ark.org/notebooks) notebook at the [Econ-ARK](https://econ-ark.org) website.

# %%
import HARK.ConsumptionSaving.ConsumerParameters as Params
import BufferStockParameters

# %% {"code_folding": [0]}
# Define a parameter dictionary with baseline parameter values

# Set the baseline parameter values
PermGroFac = 1.03
Rfree = 1.04
DiscFac = 0.96
CRRA = 2.00
UnempPrb = 0.00
IncUnemp = 0.0
PermShkStd = 0.1
TranShkStd = 0.1
# Import default parameter values
import HARK.ConsumptionSaving.ConsumerParameters as Params

# Make a dictionary containing all parameters needed to solve the model
base_params = Params.init_idiosyncratic_shocks

# Set the parameters for the baseline results in the paper
# using the variable values defined in the cell above
base_params["PermGroFac"] = [PermGroFac]  # Permanent income growth factor
base_params["Rfree"] = Rfree  # Interest factor on assets
base_params["DiscFac"] = DiscFac  # Time Preference Factor
base_params["CRRA"] = CRRA  # Coefficient of relative risk aversion
base_params[
    "UnempPrb"
] = UnempPrb  # Probability of unemployment (e.g. Probability of Zero Income in the paper)
base_params["IncUnemp"] = IncUnemp  # Induces natural borrowing constraint
base_params["PermShkStd"] = [
    PermShkStd
]  # Standard deviation of log permanent income shocks
base_params["TranShkStd"] = [
    TranShkStd
]  # Standard deviation of log transitory income shocks

# Some technical settings that are not interesting for our purposes
base_params["LivPrb"] = [1.0]  # 100 percent probability of living to next period
base_params["CubicBool"] = True  # Use cubic spline interpolation
base_params["T_cycle"] = 1  # No 'seasonal' cycles
base_params["BoroCnstArt"] = None  # No artificial borrowing constraint

from HARK.utilities import plotFuncsDer, plotFuncs
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType

# %% {"code_folding": [0]}
# Create a model identical to the dolo model
# Start with the HARK baseline parameters and modify
# to be like the dolo model
base_params_dolo = dict(base_params)
base_params_dolo["BoroCnstArt"] = 0.0  # Liquidity constraint at 0
base_params_dolo["UnempPrb"] = 0  # No point-mass on unemployment state
base_params_dolo["TranShkCount"] = 25  # Default number of nodes in dolo
base_params_dolo["PermShkCount"] = 25
base_params_dolo["aXtraMax"] = BufferStockParameters.Common[
    "max_m"
]  # Use same maximum as dolo
base_params_dolo["aXtraCount"] = 100  # How dense to make the grid
base_params_dolo["DiscFac"] = 0.96
# base_params_dolo['CubicBool']    = False
model_HARK = IndShockConsumerType(
    **base_params_dolo, cycles=0
)  # cycles=0 indicates infinite horizon

# %% {"code_folding": [0]}
# Solve the HARK model
model_HARK.updateIncomeProcess()
model_HARK.solve()
model_HARK.UnempPrb = 0.05
model_HARK.unpackcFunc()

# %%
