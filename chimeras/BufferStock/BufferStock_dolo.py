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
# # Dolo
#
# [Installation instructions](https://github.com/EconForge/dolo/wiki/Installation) for dolo involve a number of dependencies, including the dolo language "dolang."  This notebook assumes all these have been installed.

# %%
import dolang
from dolo import *
import BufferStockParameters

# %% [markdown]
# Dolo defines models using "model files" whose syntax is specified in the documentation

# %%
import sys
import os
os.getcwd()

# %%
model_dolo = yaml_import("./bufferstock.yaml")
print( model_dolo )

# %% [markdown]
# ## Explaining the Dolo Model File 
#
# Each item in the `bufferstock.yaml` file corresponds to an equation in the formal mathematics of the BufferStockTheory problem.
#
# ### arbitrage:
#
# The first such item we want to consider is labelled `arbitrage`.  In Dolo, the arbitrage equation represents a condition that should be equal to zero for an optimizing agent (perhaps it should be called the `no-arbitrage` equation since the idea is that the amount by which you could make yourself better is zero).  
#
# See the dolo/dolark documentation to absorb syntactic conventions like the fact that time offsets are signified by parens; e.g., `c(-1)` is the variable `c` one period in the past.
#
# If the model is stochastic, equations that involve a future-dated variable are calculated in expectation.
#
# Dolo's specification of the model defines `perm` as the logarithm of what is called $\psi$, so its exponential `exp(perm(1))` is equivalent to $\psi_{t+1}$ in the companion notebook's notation.  Finally, the expression `| 0.0 <= c <= m` corresponds to the liquidity constraint mentioned above (mathematically, $0 \leq c \leq m$). Thus, the arbitrage equation:
#
#     arbitrage:
#         - (R*β*((c(1)*exp(perm(1))*Γ)/c)^(-ρ)-1 ) | 0.0<=c<=m
#
# is equivlent to the Euler equation in the HARK notebook, because:
#
# \begin{align*}
# 0 & = R \beta \mathbb{E}_{t}[(\psi c_{t+1}\Gamma /c_{t})^{-\rho})]-1 \\ 
# 1 & = R \beta \mathbb{E}_{t}[(\Gamma \psi \frac{c_{t+1}}{c_{t}})^{-\rho})] 
# \\c_{t}^{-\rho} & = R \beta \mathbb{E}_{t}[(\Gamma \psi c_{t+1})^{-\rho})] \\
# \end{align*}
#
# ### transition:
#
# The second equation in the `dolang` model is:
#
#     transition:
#         - m = exp(tran) + (m(-1)-c(-1))*(R/(Γ*exp(perm)))
#
# Recall that the dynamic budget constraints in the HARK problem are:
# \begin{eqnarray*}
# a_t &=& m_t - c_t \\
# m_{t+1} &=& R/(\Gamma \psi_{t+1}) a_t + \theta_{t+1} \\
# \end{eqnarray*}
#
# Substituting the first condition, defining $a_t$, into the second, and rearranging, we get:
# \begin{eqnarray}
# m_{t+1} &=& \theta_{t+1} +  (m_t - c_t) R/(\Gamma \psi_{t+1})
# \end{eqnarray}
# and $\theta_{t+1}$ is equivalent to the exponential of the lognormally-distributed stochastic variable `tran`.  (In dolo, the equation's timing is shifted back a period because its expectation is not being taken)

# %% {"code_folding": []}
# Set a maximum value of the market resources ratio m for use in both models
max_m = BufferStockParameters.Common['max_m']
model_dolo.data['calibration']['max_m'] = max_m

# Obtain the decision rule by time iteration
dr = time_iteration(model_dolo,tol=1e-08,verbose=True)
