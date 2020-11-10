# -*- coding: utf-8 -*-
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
# # The q-Model in HARK and Dolo
#
# This notebook illustrates and compares two implementations of the q-Model of investment:
# - The class QMod, which will be the basis of HARK's implementation of the model.
# - A Dolo model, represented in a yaml file.
#
# Both implementations follow Christopher D. Carroll's graduate
# Macroeconomics [lecture notes](http://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/Investment/qModel/).

# %% {"code_folding": []}
# Preamble
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

###############################
# QMod class definition #######
###############################

from scipy import interpolate
from scipy import optimize

class Qmod:
    """
    A class representing the Q investment model.
    The class follows the model's version discussed in Christopher D. Carroll's
    lecture notes:
    http://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/Investment/qModel/
    """
    
    def __init__(self,beta = 0.98,tau = 0.05,alpha = 0.33,omega = 1,zeta = 0,
                 delta = 0.1, psi = 1):
        """
        Parameters:
        - Beta: utility discount factor.
        - Tau: corporate tax rate.
        - Alpha: output elasticity with respect to capital.
        - Omega: adjustment cost parameter.
        - Zeta: investment tax credit.
        - Delta: capital depreciation rate.
        - Psi: total productivity augmenting factor.
        """
        
        # Assign parameter values
        self.beta = beta
        self.tau = tau
        self.alpha = alpha
        self.omega = omega
        self.zeta = zeta
        self.delta = delta
        self.psi = psi
        
        # Initialize
        self.P = None
        
        # Create empty consumption function
        self.k1Func = None
        
        #  Initialize steady state capital
        self.kss = None
    
    # Output
    def f(self,k):
        return(self.psi*k**self.alpha)
    
    # Marginal productivity of capital
    def f_k(self,k):
        return(self.psi*self.alpha*k**(self.alpha-1))
    
    # Revenue:
    def pi(self,k):
        return((1-self.tau)*self.f(k))
    
    # Investment adjustment cost
    def j(self,i,k):
        return(k/2*((i-self.delta*k)/k)**2*self.omega)
    
    # Expenditure:
    def expend(self,k,i):
        return((i+self.j(i,k))*self.P*self.beta)
    
    # Flow utility
    def flow(self,k,i):
        return(self.pi(k) - self.expend(k,i))
        
    # Value function: maximum expected discounted utility given initial caputal
    def value_func(self,k,tol = 10**(-2)):
        """
        Parameters:
            - k  : (current) capital.
            - tol: absolute distance to steady state capital at which the model
                   will be considered to have reached its steady state.
        """
        
        if abs(k-self.kss) > tol:
            # If steady state has not been reached, find the optimal capital
            # for the next period and continue computing the value recursively.
            k1 = self.k1Func(k)
            i = k1 - k*(1-self.delta)
            return(self.flow(k,i) + self.beta*self.value_func(k1,tol))
        
        else:
            # If steady state is reached return present discounted value
            # of all future flows (which will be identical)
            return(self.flow(self.kss,self.kss*self.delta)/(1-self.beta))
        
    # Derivative of adjustment cost with respect to investment
    def j_i(self,i,k):
        iota = i/k - self.delta
        return(iota*self.omega)
    
    # Derivative of adjustment cost with respect to capital.
    def j_k(self,i,k):
        iota = i/k - self.delta
        return(-(iota**2/2+iota*self.delta)*self.omega)
    
    # Error in the euler Equation implied by a k_0, k_1, k_2 triad.
    # This can be solved to obtain the triads that are consistent with the
    # equation.
    def eulerError(self,k0,k1,k2):
        
        # Compute implied investments at t=0 and t=1.
        i0 = k1 - (1-self.delta)*k0
        i1 = k2 - (1-self.delta)*k1
        
        # Compute implied error in the Euler equation
        error = (1+self.j_i(i0,k0))*self.P -\
                ((1-self.tau)*self.f_k(k1) +\
                 ((1-self.delta) +\
                  (1-self.delta)*self.j_i(i1,k1) - self.j_k(i1,k1)
                 )*self.P*self.beta
                )
        
        return(error)
    
    # Find the k_2 implied by the euler equation for an initial k_0, k_1.
    def k2(self,k0,k1):
        
        # Find the k2 that is consistent with the Euler equation
        sol = optimize.root_scalar(lambda x: self.eulerError(k0,k1,x),
                                   x0=k0, x1=self.kss)
        
        # Return exception if no compatible capital is found
        if sol.flag != "converged":
            raise Exception('Could not find capital value satisfying Euler equation')
        
        return(sol.root)
    
    # Find the capital trajectory implied by the euler equation for
    # an initial k_0, k_1.
    def shoot(self,k0,k1,t):
        """
        Parameters:
            - k0, k1: initial values for capital.
            - t     : number of periods to be simulated.
        """
        # Initialize k
        k = np.zeros(t)
        k[0] = k0
        k[1] = k1
        
        # Simulate capital dynamics
        for i in range(2,t):
            
            try:
                k[i] = self.k2(k[i-2],k[i-1])
            except:
                # If at some point no solution can be found stop simulation.
                k[i:] = k[i]
                return(k)
                
            if k[i]<0 or (abs(k[i]-self.kss) > 2*abs(k0-self.kss)):
                # If a negative or diverging capital is obtained, stop
                # simulation
                k[i:] = k[i]
                return(k)
            
        return(k)
    
    # Shooting algorithm to find k_1 given k_0.
    def find_k1(self,k0,T=30,tol = 10**(-3),maxiter = 200):
        """
        Parameters:
            - k0     : initial value of capital.
            - T      : number of time periods to be simulated for every
                       candidate solution.
            - tol    : distance between k(T) and steady state capital at which
                       a solution is satisfactory
            - maxiter: maximum number of iterations.
        """
        # Initialize interval over which a solution is searched.
        top = max(self.kss,k0)
        bot = min(self.kss,k0)
        
        for k in range(maxiter):
            
            # Simulate capital dynamics at the midpoint of the
            # current interval.
            init = (top+bot)/2
            path = self.shoot(k0,init,T)
            
            # Check the final value of capital
            k_f = path[-1]
            
            if np.isnan(k_f):
                bot = init
            else:
                if abs(k_f - self.kss)<tol:
                    # Stop if capital reaches and stays at
                    # the steady state
                    return(init)
                else:
                    if k_f >= self.kss:
                        # If capital ends up above steady state,
                        # we are underestimating k_1.
                        top = init
                    else:
                        # If capital ends up below steady state,
                        # we are overestimating k_1
                        bot = init
            
        return(init)
    
    # Construction of the policy rule by solving for k_1 given
    # k_0 over a grid of points and then finding an interpolating
    # function
    def solve(self,k_min=10**(-4), n_points = 50):
        """
        Parameters:
            - k_min   : minimum value of capital at which the policy rule will
                        be solved for.
            - n_points: number of points at which to numerically solve for the
                        policy rule.
        """
        
        # Set the price of capital after ITC
        self.P = (1-self.zeta)
        
        # First find steady state capital (in case parameters were changed)
        self.kss = ((1-(1-self.delta)*self.beta)*self.P/((1-self.tau)*
                     self.alpha*self.psi))**(1/(self.alpha-1))
        
        # Create k_0 grid
        k_max = 4*self.kss
        k0 = np.linspace(k_min,k_max,n_points)
        k1 = np.zeros(len(k0))
        
        # Find k_0 at each point in the grid
        for i in range(len(k0)):
            
            k1[i] = self.find_k1(k0[i])
        
        # Interpolate over the grid to get a continuous
        # function
        self.k1Func = interpolate.interp1d(k0,k1)
    
    # Simulation of capital dynamics from a starting k_0 for a number of
    # periods t
    def simulate(self,k0,t):
        k = np.zeros(t)
        k[0]=k0
        for i in range(1,t):
            k[i] = self.k1Func(k[i-1])
        return(k)
    
    # Net investment ratio at t, as a function of marginal value of capital at
    # t+1.
    def iota(self,lam_1):
        iota = ( lam_1/self.P - 1)/self.omega
        return(iota)
    
    # Detivative of adjustment costs as a function of lambda(t+1), assuming
    # optimal investment.
    def jkl(self,lam_1):
        iota = self.iota(lam_1)
        jk = -(iota**2/2+iota*self.delta)*self.omega
        return(jk)
    
    # Plot the marginal value of capital at t implied by the envelope condition,
    # as a function of the marginal value at t+1, at a given level of capital.
    def plotEnvelopeCond(self,k, npoints = 10):
        
        # Create grid for lambda(t+1)
        lam_1 = np.linspace(0,2,npoints)
        
        # Compute each component of the envelope condition
        prod = np.ones(npoints)*(1-self.tau)*self.f_k(k)
        iota = (lam_1/self.P - 1)/self.omega
        jk = - (iota**2/2+iota*self.delta)*self.omega
        inv_gain = -jk*self.beta*self.P
        fut_val = (1-self.delta)*self.beta*lam_1
        
        # Plot lambda(t) as a function of lambda(t+1)
        plt.plot(lam_1,prod+inv_gain+fut_val, label = "Env. Condition value")
        plt.plot(lam_1,lam_1, linestyle = '--', color = 'k', label = "45° line")
        
        plt.legend()
        plt.title('$\\lambda (t)$ vs $\lambda (t+1)$ at $k =$ %1.2f' %(k))
        plt.xlabel('$\\lambda (t+1)$')
        plt.ylabel('$\\lambda (t)$')
    
    # Solve for the value of lambda(t) that implies lambda(t)=lambda(t+1) at
    # a given level of capital.
    def lambda0locus(self,k):
        
        # Set the initial solution guess acording to the level of capital. This
        # is important given that the equation to be solved is quadratic.
        if k > self.kss:
            x1 = 0.5*self.P
        else:
            x1 = 1.5*self.P
            
        bdel = self.beta*(1-self.delta)
        
        # Lambda solves the following equation:
        error = lambda x: (1-bdel)*x - (1-self.tau)*self.f_k(k) +\
                          self.jkl(x)*self.beta*self.P
        
        # Search for a solution. The locus does not exist at all k.
        sol = optimize.root_scalar(error, x0 = self.P, x1 = x1)
        if sol.flag != 'converged':
            return( np.float('nan') )
        else:
            return(sol.root)
    
    # Compute marginal value of capital at t using k0,k1 and the envelope
    # condition
    def findLambda(self,k0,k1):
        
        # Implied investment at t
        i = k1 - (1-self.delta)*k0
        iota = i/k0 - self.delta
        
        q1 = iota*self.omega + 1
        lam1 = q1*self.P 
        
        # Envelope equation
        lam = (1-self.tau)*self.f_k(k0) - self.j_k(i,k0)*self.beta*self.P +\
              self.beta*(1-self.delta)*lam1
              
        return(lam)
    
    # Plot phase diagram of the model
    def phase_diagram(self, k_min = 0.1, k_max = 2,npoints = 200,
                      stableArm = False,
                      Qspace = False):
        """
        Parameters:
            - [k_min,k_max]: minimum and maximum levels of capital for the
                             diagram, expressed as a fraction of the steady
                             state capital.
            - npoints      : number of points in the grid of capital for which
                             the loci are plotted.
            - stableArm    : enables/disables plotting of the model's stable
                             arm.
            - Qspace       : boolean indicating whether the diagram should be
                             in Q space instead of lambda space.
        """
        
        # Create capital grid.
        k = np.linspace(k_min*self.kss,k_max*self.kss,npoints)
        
        # Define normalization factor in case we are in Qspace
        fact = 1
        yLabel = '\\lambda'
        if Qspace:
            fact = 1/self.P
            yLabel = 'q'
            
        # Plot 
        plt.figure()
        # Plot k0 locus
        plt.plot(k,self.P*np.ones(npoints) * fact,
                 label = '$\\dot{k}=0$ locus')
        # Plot lambda0 locus
        plt.plot(k,[self.lambda0locus(x)*fact for x in k],
                 label = '$\\dot{'+yLabel+'}=0$ locus')
        # Plot steady state
        plt.plot(self.kss,self.P*fact,'*r', label = 'Steady state')
        
        # PLot stable arm
        if stableArm:
            
            if self.k1Func is None:
                raise Exception('Solve the model first to plot the stable arm!')
            else:
                lam = np.array([self.findLambda(k0 = x, k1 = self.k1Func(x))
                                for x in k])
                plt.plot(k,lam*fact, label = 'Stable arm')
        
        # Labels
        plt.title('Phase diagram')
        plt.xlabel('$k$')
        plt.ylabel('$'+yLabel+'$')
        plt.legend()
        
        plt.show()


##############################################################################
# Definition of additional related utilities
##############################################################################

def pathValue(invest,mod1,mod2,k0,t):
    '''
    Computes the value of taking investment decisions [i(0),i(1),...,i(t-1)]
    starting at capital k0 and knowing that the prevailing model will switch
    from mod1 to mod2 at time t.

    Parameters:
        - invest: vector/list with investment values for periods 0 to t-1
        - mod1  : Qmod object representing the parameter values prevailing from
                  time 0 to t-1.
        - mod2  : Qmod object representing the parameter values prevailing from
                  time t onwards.
        - k0    : capital at time 0.
        - t     : time of the structural change.
    '''

    # Initialize capital and value (utility)
    k = np.zeros(t+1)
    k[0] = k0
    value = 0

    # Compute capital and utility flows until time t-1
    for i in range(t):
        flow = mod1.flow(k[i],invest[i])
        value += flow*mod1.beta**i
        k[i+1] = k[i]*(1-mod1.delta) + invest[i]

    # From time t onwards, model 2 prevails and its value function can be used.
    value += (mod1.beta**t)*mod2.value_func(k[t])

    return(value)

def structural_change(mod1,mod2,k0,t_change,T_sim,npoints = 300):
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
        inv = optimize.minimize(fobj,x0 = np.ones(t_change)*mod1.kss*mod2.delta,
                                options = {'disp': True},
                                tol = 1e-16).x

    # Find paths of capital and lambda
    k = np.zeros(T_sim)
    lam = np.zeros(T_sim)
    k[0] = k0
    for i in range(0,T_sim-1):

        if i < t_change:
            # Before the change, investment follows the optimal
            # path computed above.
            k[i+1] = k[i]*(1-mod1.delta) + inv[i]
            lam[i] = mod1.findLambda(k[i],k[i+1])
        else:
            # After the change, investment follows the post-change policy rule.
            k[i+1] = mod2.k1Func(k[i])
            lam[i] = mod2.findLambda(k[i],k[i+1])

    lam[T_sim-1] = mod2.findLambda(k[T_sim-1],mod2.k1Func(k[T_sim-1]))

    # Create a figure with phase diagrams and dynamics.
    plt.figure()

    # Plot k,lambda path.
    plt.plot(k,lam,'.k')
    plt.plot(k[t_change],lam[t_change],'.r',label = 'Change takes effect')

    # Plot the loci of the pre and post-change models.
    k_range = np.linspace(0.1*min(mod1.kss,mod2.kss),2*max(mod1.kss,mod2.kss),
                          npoints)
    mods = [mod1,mod2]
    colors = ['r','b']
    labels = ['Pre-change','Post-change']
    for i in range(2):

        # Plot k0 locus
        plt.plot(k_range,mods[i].P*np.ones(npoints),
                 linestyle = '--', color = colors[i],label = labels[i])
        # Plot lambda0 locus
        plt.plot(k_range,[mods[i].lambda0locus(x) for x in k_range],
                 linestyle = '--', color = colors[i])
        # Plot steady state
        plt.plot(mods[i].kss,mods[i].P,marker = '*', color = colors[i])

    plt.title('Phase diagrams and model dynamics')
    plt.xlabel('K')
    plt.ylabel('Lambda')
    plt.legend()

    return({'k':k, 'lambda':lam})

# %% [markdown]
# ## 1. Basic features of the Qmod class
#
# We first illustrate how to create and use the main features of the Qmod class.

# %% [markdown]
# ### Creating a model
#
# We first create an instance of the model with the default parameter values.

# %%
# Create model object
Qexample = Qmod()

# %% [markdown]
# ### Model solution and policy rule.
#
# Now that we have created the model, we can solve it. To solve the model is to find its policy rule: a function specifying what is the optimal value for capital at $t+1$ given capital at $t$ (implicitly defining optimal investment). Solving the model also finds is steady state value of capital.
#
# We now illustrate these two features.

# %%
# Solve the model
Qexample.solve()

# Print its steady state
print('The steady state value of capital is %f' % (Qexample.kss))

# Plot policy rule
k = np.linspace(1,3*Qexample.kss,20)

plt.figure()
plt.plot(k,[Qexample.k1Func(x) for x in k], label = "Optimal capital")
plt.plot(k,k, linestyle = '--', color = 'k', label = "45° line")
plt.plot(Qexample.kss,Qexample.kss,'*r', label = "Steady state")
plt.title('Policy Rule')
plt.xlabel('$k(t)$')
plt.ylabel('$k(t+1)$')
plt.legend()
plt.show()
# %% [markdown]
# ### Simulation of capital dynamics.
#
# The class can also compute the dynamic adjustment of capital from a given starting level.
#
# We can use this to see how adjustment costs affect the speed of adjustment.

# %%
# Create and solve two instances, one with high and one with low adjustment costs omega
Qlow  = Qmod(omega =  0.1)
Qhigh = Qmod(omega =  0.9)

Qlow.solve()
Qhigh.solve()

# Simulate adjustment from an initial capital level
k0 = 2*Qhigh.kss
t = 50
k_low = Qlow.simulate(k0,t)
k_high = Qhigh.simulate(k0,t)

# Plot
plt.figure()
plt.plot(k_low, label = 'Low $\\omega$')
plt.plot(k_high, label = 'High $\\omega$')
plt.axhline(y = Qhigh.kss,linestyle = '--',color = 'k', label = 'Steady state ${k}$')
plt.title('Capital')
plt.xlabel('$t$')
plt.ylabel('$k(t)$')
plt.legend()
plt.show()
# %% [markdown]
# ### Phase diagram.
#
# The class can plot a model's phase diagram. The model has to be solved if the stable arm is to be displayed.

# %%
# Create and solve model object
Qexample = Qmod()
Qexample.solve()
# Generate its phase diagram
Qexample.phase_diagram(stableArm = True)

# %% [markdown]
# ## 2. Structural Changes Using Qmod and Dolo
#
# The tools in this repository can also be used to analyze the models optimal dynamic response to structural changes.
#
# To illustrate this capabilities, I simulate the changes discussed in Christopher D. Carroll's graduate
# Macroeconomics [lecture notes](http://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/Investment/qModel/):
# productivity, corporate tax rate, and investment tax credit changes.
#
# For each change I display the behavior of the model under two different assumptions:
# * The change takes place at $t=0$ without notice.
# * The change is announced at $t=0$ but takes place at $t=5$.
#
# I find the optimal responses using both Qmod and the Dolo implementation of the q-Model. Thus, I first load the required tools that Dolo uses.

# %% {"code_folding": []}
from dolo import *
import dolo.algos.perfect_foresight as pf
import dolo.algos.value_iteration as vi
import pandas as pd

# %% [markdown]
# Now I create a base model parametrization using both the Qmod class and the Dolo implementation.

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

## Dolo

QDolo = yaml_import("./Q_model.yaml")
# We do not pass psi, tau, or zeta since they are handled not as parameters
# but exogenous variables.
QDolo.set_calibration(R = R, alpha = alpha, delta = delta, omega = omega)


# %% [markdown]
# I use "structural_change" (defined above in the preamble), a function that computes and presents optimal dynamics in face of structural changes in the Qmod implementation.
# %% [markdown]
# Now, I define another function to easily simulate parameter changes in the Dolo
# implementation
# %% {"code_folding": [0]}
def simul_change_dolo(model, k0,  exog0, exog1, t_change, T_sim):

    # The first step is to create time series for the exogenous variables
    exog = np.array([exog1,]*(T_sim - t_change))
    if t_change > 0:
        exog = np.concatenate((np.array([exog0,]*(t_change)),
                               exog),
                              axis = 0)
    exog = pd.DataFrame(exog, columns = ['R','tau','itc_1','psi'])

    # Simpulate the optimal response
    dr = pf.deterministic_solve(model = model,shocks = exog, T=T_sim,
                                verbose=True, s1 = k0)

    # Dolo uses the first period to report the steady state
    # so we ommit it.
    return(dr[1:])

# %% [markdown]
# ### Examples:
#
# We are now ready to simulate structural changes.
# %% [markdown]
# #### 2.1. An unanticipated increase in productivity

# %% {"code_folding": [0]}
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
                        k0 = k0, t_change = t,T_sim=T)

## Dolo

soldolo = simul_change_dolo(model = QDolo, k0 = np.array([k0]),
                            exog0 = [R,tau,zeta,psi],
                            exog1 = [R,tau,zeta,psi_new],
                            t_change = t, T_sim = T)

# Plot the path of capital under both solutions
time = range(T)
plt.figure()
plt.plot(time, sol['k'], 'x', label = 'Qmod', alpha = 0.8)
plt.plot(time, soldolo['k'], '+', label = 'Dolo', alpha = 0.8)
plt.legend()
plt.title('Capital dynamics')
plt.ylabel('$k_t$ : capital')
plt.xlabel('$t$ : time')
# %% [markdown]
# #### 2.2. An increase in productivity announced at t=0 but taking effect at t=5
# %% {"code_folding": []}
# Repeat the calculation now assuming the change happens at t=5
t = 5

# Qmod class
sol = structural_change(mod1 = Qmodel, mod2 = Q_high_psi,
                        k0 = k0, t_change = t,T_sim=T)

# Dolo
soldolo = simul_change_dolo(model = QDolo, k0 = np.array([k0]),
                            exog0 = [R,tau,zeta,psi],
                            exog1 = [R,tau,zeta,psi_new],
                            t_change = t, T_sim = T)

# Plot the path of capital under both solutions
time = range(T)
plt.figure()
plt.plot(time, sol['k'], 'x', label = 'Qmod', alpha = 0.8)
plt.plot(time, soldolo['k'], '+', label = 'Dolo', alpha = 0.8)
plt.legend()
plt.title('Capital dynamics')
plt.ylabel('$k_t$ : capital')
plt.xlabel('$t$ : time')
# %% [markdown]
# #### 2.3. An unanticipated corporate tax-cut
# %% {"code_folding": [0]}
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
                        k0 = k0, t_change = t,T_sim=T)

# Dolo
soldolo = simul_change_dolo(model = QDolo, k0 = np.array([k0]),
                            exog0 = [R,tau_high,zeta,psi],
                            exog1 = [R,tau,zeta,psi],
                            t_change = t, T_sim = T)

# Plot the path of capital under both solutions
time = range(T)
plt.figure()
plt.plot(time, sol['k'], 'x', label = 'Qmod', alpha = 0.8)
plt.plot(time, soldolo['k'], '+', label = 'Dolo', alpha = 0.8)
plt.legend()
plt.title('Capital dynamics')
plt.ylabel('$k_t$ : capital')
plt.xlabel('$t$ : time')
# %% [markdown]
# #### 2.4. A corporate tax cut announced at t=0 but taking effect at t=5
# %% {"code_folding": [0]}
# Modify the time of the change
t = 5

# Qmod class
sol = structural_change(mod1 = Q_high_tau, mod2 = Qmodel,
                        k0 = k0, t_change = t,T_sim=T)

# Dolo
soldolo = simul_change_dolo(model = QDolo, k0 = np.array([k0]),
                            exog0 = [R,tau_high,zeta,psi],
                            exog1 = [R,tau,zeta,psi],
                            t_change = t, T_sim = T)

# Plot the path of capital under both solutions
time = range(T)
plt.figure()
plt.plot(time, sol['k'], 'x', label = 'Qmod', alpha = 0.8)
plt.plot(time, soldolo['k'], '+', label = 'Dolo', alpha = 0.8)
plt.legend()
plt.title('Capital dynamics')
plt.ylabel('$k_t$ : capital')
plt.xlabel('$t$ : time')
# %% [markdown]
# #### 2.5. An unanticipated ITC increase
# %% {"code_folding": [0]}
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
                        k0 = k0, t_change = t,T_sim=T)

# Dolo
soldolo = simul_change_dolo(model = QDolo, k0 = np.array([k0]),
                            exog0 = [R,tau,zeta,psi],
                            exog1 = [R,tau,itc_high,psi],
                            t_change = t, T_sim = T)

# Plot the path of capital under both solutions
time = range(T)
plt.figure()
plt.plot(time, sol['k'], 'x', label = 'Qmod', alpha = 0.8)
plt.plot(time, soldolo['k'], '+', label = 'Dolo', alpha = 0.8)
plt.legend()
plt.title('Capital dynamics')
plt.ylabel('$k_t$ : capital')
plt.xlabel('$t$ : time')
# %% [markdown]
# #### 2.6. An ITC increase announced at t=0 but taking effect at t=5
# %% {"code_folding": [0]}
# Modify time of the change
t = 5

# Qmod class
sol = structural_change(mod1 = Qmodel, mod2 = Q_high_itc,
                        k0 = k0, t_change = t,T_sim=T)

# Dolo
soldolo = simul_change_dolo(model = QDolo, k0 = np.array([k0]),
                            exog0 = [R,tau,zeta,psi],
                            exog1 = [R,tau,itc_high,psi],
                            t_change = t+1, T_sim = T)

# Plot the path of capital under both solutions
time = range(T)
plt.figure()
plt.plot(time, sol['k'], 'x', label = 'Qmod', alpha = 0.8)
plt.plot(time, soldolo['k'], '+', label = 'Dolo', alpha = 0.8)
plt.legend()
plt.title('Capital dynamics')
plt.ylabel('$k_t$ : capital')
plt.xlabel('$t$ : time')


# %% [markdown]
# ## 3.The advantages of Dolo
#
# Currently, Qmod represents a model in which the interest rate, taxes, and productivity are considered parameters. I compute the transitional dynamics of structural changes by approximating the final value function and then simultaneously optimizing over the transitional investment decisions. This approach works for simple changes that are not far into the future, but even then it can be imprecise and slow (see e.g. Experiment 2.2. above).
#
# On the other hand, the Dolo implementation considers taxes, interest rates, and productivity as exogenous dynamic variables, and solves the problem of transitional dynamics using dynamic optimization tools. This makes it able to easily handle more complicated changes and paths for these variables in the future.
#
# This section illustrates simulations of structural changes in Dolo that would be hard to handle using Qmod. The premise, as before, is that the firm is sitting in steady state and, at time $t=0$ it learns that the exogenous variables will follow the represented paths in the future. It then incorporates this information and reacts optimally.

# %% [markdown]
# I first define a function that, given a future path for the exogenous variables, uses Dolo to solve for the firm's optimal response and plots the results.

# %%
# Define a function to handle plots
def plotQmodel(model, exog, returnDF = False):
    
    # Simpulate the optimal response
    dr = pf.deterministic_solve(model = model,shocks = exog,verbose=True)
    
    # Plot exogenous variables
    ex = ['R','tau','itc_1','psi']
    fig, axes = plt.subplots(1,len(ex), figsize = (10,3))
    axes = axes.flatten()
    
    for i in range(len(ex)):
        ax = axes[i]
        ax.plot(dr[ex[i]],'.')
        ax.set_xlabel('Time')
        ax.set_ylabel(ex[i])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    fig.suptitle('Exogenous variables', fontsize=16)
    
    # Plot optimal response variables
    fig, axes = plt.subplots(2,2, figsize = (10,6))
    axes = axes.flatten()
    opt = ['k','i','lambda_1','q_1']
    
    for i in range(len(opt)):
        ax = axes[i]
        ax.plot(dr[opt[i]],'.')
        ax.set_xlabel('Time')
        ax.set_ylabel(opt[i])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    fig.suptitle('Endogenous response', fontsize=16)
    
    if returnDF:
        return(dr)


# %% [markdown]
# I now produce various simulations.

# %% [markdown]
# ### 3.1. Interest rates that change multiple times.

# %%
# First define the paths of exogenous variables

# Create empty dataframe for exog. variables
exog = pd.DataFrame(columns = ['R','tau','itc_1','psi'])

# Generate an interest rate process
exog.R = np.concatenate((np.repeat(1.03,20),
                         np.repeat(1.05,10),
                         np.repeat(0.97,10),
                         np.repeat(1.07,20),
                         np.repeat(1.03,10)))

# Leave tau at 0
exog.tau = 0
# Leave itc at 0
exog.itc_1 = 0
# Leave psi at 1
exog.psi = 1

# Solve for the optimal response and plot the results  
plotQmodel(QDolo,exog)

# %% [markdown]
# ### 3.2. Multiple parameters changing at different times

# %%
# Create empty dataframe for exog. variables
exog = pd.DataFrame(columns = ['R','tau','itc_1','psi'])

# Generate future tax dynamics
exog.tau = np.concatenate((np.repeat(0.2,20),
                           np.repeat(0,20)))

# Generate future itc dynamics
exog.itc_1 = np.concatenate((np.repeat(0,15),
                             np.repeat(0.2,25)))

# Generate future productivity dynamics
exog.psi= np.concatenate((np.repeat(1,10),
                          np.repeat(1.1,20),
                          np.repeat(1,10)))

# Leave R at 1.02
exog.R = 1.02

# Solve for the optimal response and plot the results  
plotQmodel(QDolo,exog)
