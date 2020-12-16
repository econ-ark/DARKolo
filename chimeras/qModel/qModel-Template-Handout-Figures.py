# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Structural Changes in the q Model: lecture notes experiments
#
# This notebook simulates the thought experiments discussed in Christopher D. Carroll's graduate
# Macroeconomics [lecture notes](http://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/Investment/qModel/):
# productivity, corporate tax rate, and investment tax credit changes. For each experiment, the the figure from the lecture notes is reproduced.
#
# For each change I display the behavior of the model in two different
# contexts:
# * The change takes place at $t=0$ without notice.
# * The change is announced at $t=0$ but takes place at $t=5$.

# %% {"code_folding": []}
# Preamble
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['axes.labelsize'] = 20

from copy import deepcopy
from scipy import optimize

from dolo import *
import dolo.algos.perfect_foresight as pf
import dolo.algos.value_iteration as vi

import pandas as pd

# Since the Qmod class is in other folder we need to
# change the path
from Qmod.Q_investment import Qmod, pathValue

# %% [markdown]
# Since the plots for every experiment have the same format, I first define a function that carries out the analysis given a path for the exogenous variables.
# %% {"code_folding": []}
def structural_change(mod1,mod2,k0,t_change,T_sim,npoints = 300, figname = None):
    """
    Computes (optimal) capital and lambda dynamics in face of a structural
    change in the Q investment model.

    Parameters:
        - mod1    : Qmod object representing the parameter values prevailing
                    from time 0 to t_change-1.
        - mod2    : Qmod object representing the parameter values prevailing
                    from time t_change onwards.
        - k0      : initial value for capital.
        - t_change: time period at which the structural change takes place. It
                    is assumed that the change is announced at period 0.
        - T_sim   : final time period of the simulation.
        - npoints : number of points in the capital grid to be used for phase
                    diagram plots.
    """

    # If the change is announced with anticipation, the optimal path of
    # investment from 0 to t_change-1 is computed, as it does not correspond to
    # the usual policy rule.
    if t_change > 0:
        fobj = lambda x: -1*pathValue(x,mod1,mod2,k0,t_change)
        inv = optimize.minimize(fobj,x0 = np.ones(t)*mod1.kss*mod2.delta,
                                options = {'disp': True},
                                tol = 1e-16).x

    # Find paths of capital and lambda
    k = np.zeros(T_sim)
    lam = np.zeros(T_sim)
    invest = np.zeros(T_sim)
    
    k[0] = k0
    for i in range(0,T_sim-1):

        if i < t_change:
            # Before the change, investment follows the optimal
            # path computed above.
            k[i+1] = k[i]*(1-mod1.delta) + inv[i]
            lam[i] = mod1.findLambda(k[i],k[i+1])
            invest[i] = inv[i]
            
        else:
            # After the change, investment follows the post-change policy rule.
            k[i+1] = mod2.k1Func(k[i])
            lam[i] = mod2.findLambda(k[i],k[i+1])
            invest[i] = k[i+1] - (1-mod2.delta)*k[i]
            
    # Compute final period lambda and investment
    lam[T_sim-1] = mod2.findLambda(k[T_sim-1],mod2.k1Func(k[T_sim-1]))
    invest[T_sim-1] = mod2.k1Func(k[T_sim-1]) - (1-mod2.delta)*k[T_sim-1]
    
    # Get a vector with the post-itc price of capital, to calculate q
    Pcal = np.array([1-mod1.zeta]*t_change + [1-mod2.zeta]*(T_sim-t_change))
    
    # Compute q
    q = lam/Pcal
    
    # Create a figure with phase diagrams and dynamics.
    fig, ax = plt.subplots(3, 2, figsize=(15,12))
    
    # 1st plot: lambda phase diagrams
    
    # Plot k,lambda path.
    ax[0,0].plot(k,lam,'.k')
    ax[0,0].plot(k[t_change],lam[t_change],'.r',label = 'Change takes effect')

    # Plot the loci of the pre and post-change models.
    k_range = np.linspace(0.1*min(mod1.kss,mod2.kss),2*max(mod1.kss,mod2.kss),
                          npoints)
    mods = [mod1,mod2]
    colors = ['r','b']
    labels = ['Pre-change','Post-change']
    for i in range(2):

        # Plot k0 locus
        ax[0,0].plot(k_range,mods[i].P*np.ones(npoints),
                     linestyle = '--', color = colors[i],label = labels[i])
        # Plot lambda0 locus
        ax[0,0].plot(k_range,[mods[i].lambda0locus(x) for x in k_range],
                     linestyle = '--', color = colors[i])
        # Plot steady state
        ax[0,0].plot(mods[i].kss,mods[i].P,marker = '*', color = colors[i])

    ax[0,0].set_xlabel('$k$')
    ax[0,0].set_ylabel('$\\lambda$')
    ax[0,0].legend()
    
    # 2nd plot: q phase diagrams
    
    # Plot k,lambda path.
    ax[0,1].plot(k,q,'.k')
    ax[0,1].plot(k[t_change],q[t_change],'.r',label = 'Change takes effect')

    # Plot the loci of the pre and post-change models.
    mods = [mod1,mod2]
    colors = ['r','b']
    labels = ['Pre-change','Post-change']
    for i in range(2):

        # Plot k0 locus
        ax[0,1].plot(k_range,np.ones(npoints),
                     linestyle = '--', color = colors[i],label = labels[i])
        # Plot q0 locus
        ax[0,1].plot(k_range,[mods[i].lambda0locus(x)/mods[i].P for x in k_range],
                     linestyle = '--', color = colors[i])
        # Plot steady state
        ax[0,1].plot(mods[i].kss,1,marker = '*', color = colors[i])

    ax[0,1].set_xlabel('$k$')
    ax[0,1].set_ylabel('$q$')
    ax[0,1].legend()
    
    # 3rd plot: capital dynamics
    time = range(T_sim)
    ax[1,0].plot(time,k,'.k')
    ax[1,0].set_xlabel('$t$')
    ax[1,0].set_ylabel('$k_t$')
    
    # 4rd plot: lambda dynamics
    ax[1,1].plot(time,lam,'.k')
    ax[1,1].set_xlabel('$t$')
    ax[1,1].set_ylabel('$\\lambda_t$')
    
    # 5th plot: investment dynamics
    ax[2,0].plot(time,invest,'.k')
    ax[2,0].set_xlabel('$t$')
    ax[2,0].set_ylabel('$i_t$')
    
    # 6th plot: q dynamics
    ax[2,1].plot(time,q,'.k')
    ax[2,1].set_xlabel('$t$')
    ax[2,1].set_ylabel('$q_t$')
    
    if figname is not None:
        fig.savefig('./Figures/'+figname+'.svg')
        fig.savefig('./Figures/'+figname+'.png')
        fig.savefig('./Figures/'+figname+'.pdf')
    
    return({'k':k, 'lambda':lam})
# %%
# Base parameters

# Discount factor and return factor
beta = 0.98
R = 1/beta

# Tax rate
tau = 0.05

# Share of capital in production
alpha = 0.33

# Adjustment costs
omega = 1

# Investment tax credit
zeta = 0

# Depreciation rate
delta = 0.1

# Technological factor
psi = 1


## Qmod python class
Qmodel = Qmod(beta, tau, alpha, omega, zeta, delta, psi)
Qmodel.solve()

# %% [markdown]
# ## Examples:
#
# ## 1. An unanticipated increase in productivity
# %% {"code_folding": [0]}
figname = 'ProductivityIncrease'

# Total simulation time
T = 20
# Time the change occurs
t = 0
# Initial level of capital
k0 = Qmodel.kss

# Productivity in the "new" state
psi_new = 1.3

## Qmod class

# Copy the initial model, set the higher psi and re-solve
Q_high_psi = deepcopy(Qmodel)
Q_high_psi.psi = psi_new
Q_high_psi.solve()

sol = structural_change(mod1 = Qmodel, mod2 = Q_high_psi,
                        k0 = k0, t_change = t,T_sim=T,
                       figname = figname)
fig = plt.gcf() # Get the figure in order to save it
# %% [markdown]
# ## 2. An increase in productivity announced at t=0 but taking effect at t=5
# %% {"code_folding": []}
figname = 'ProductivityIncrease_ant'

# Repeat the calculation now assuming the change happens at t=5
t = 5

# Qmod class
sol = structural_change(mod1 = Qmodel, mod2 = Q_high_psi,
                        k0 = k0, t_change = t,T_sim=T,
                        figname = figname)
# %% [markdown]
# ## 3. An unanticipated corporate tax-cut
# %% {"code_folding": [0]}
figname = 'CorporateTaxReduction'

# Set the taxes of the 'high-tax' scenario
tau_high = 0.4
# Set time of the change
t = 0

# Qmod class

# Copy the initial model, set a higher psi and re-solve
Q_high_tau = deepcopy(Qmodel)
Q_high_tau.tau = tau_high
Q_high_tau.solve()

# Capital will start at it steady state in the
# high-tax scenario
k0 = Q_high_tau.kss

sol = structural_change(mod1 = Q_high_tau, mod2 = Qmodel,
                        k0 = k0, t_change = t,T_sim=T,
                        figname = figname)
# %% [markdown]
# ## 4. A corporate tax cut announced at t=0 but taking effect at t=5
# %% {"code_folding": [0]}
figname = 'CorporateTaxReduction_ant'

# Modify the time of the change
t = 5

# Qmod class
sol = structural_change(mod1 = Q_high_tau, mod2 = Qmodel,
                        k0 = k0, t_change = t,T_sim=T,
                        figname = figname)
# %% [markdown]
# ## 5. An unanticipated ITC increase
# %% {"code_folding": [0]}
figname = 'ITCIncrease'

# Set time of the change
t=0
# Set investment tax credit in the high case
itc_high = 0.2
# Set initial value of capital
k0 = Qmodel.kss

# Qmod class

# Copy the initial model, set a higher psi and re-solve
Q_high_itc = deepcopy(Qmodel)
Q_high_itc.zeta = itc_high
Q_high_itc.solve()

sol = structural_change(mod1 = Qmodel, mod2 = Q_high_itc,
                        k0 = k0, t_change = t,T_sim=T,
                        figname = figname)
# %% [markdown]
# ## 6. An ITC increase announced at t=0 but taking effect at t=5
# %% {"code_folding": [0]}
figname = 'ITCIncrease_ant'

# Modify time of the change
t = 5

# Qmod class
sol = structural_change(mod1 = Qmodel, mod2 = Q_high_itc,
                        k0 = k0, t_change = t,T_sim=T,
                        figname = figname)
