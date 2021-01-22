# -*- coding: utf-8 -*-
"""
@author: Mateo Velásquez-Giraldo
https://github.com/Mv77
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import optimize


class Qmod:
    """
    A class representing the Q investment model.
    The class follows the model's version discussed in Christopher D. Carroll's
    lecture notes:
    http://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/Investment/qModel/
    """

    def __init__(
        self, beta=0.98, tau=0.05, alpha=0.33, omega=1, zeta=0, delta=0.1, psi=1
    ):
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
    def f(self, k):
        return self.psi * k ** self.alpha

    # Marginal productivity of capital
    def f_k(self, k):
        return self.psi * self.alpha * k ** (self.alpha - 1)

    # Revenue:
    def pi(self, k):
        return (1 - self.tau) * self.f(k)

    # Investment adjustment cost
    def j(self, i, k):
        return k / 2 * ((i - self.delta * k) / k) ** 2 * self.omega

    # Expenditure:
    def expend(self, k, i):
        return (i + self.j(i, k)) * self.P * self.beta

    # Flow utility
    def flow(self, k, i):
        return self.pi(k) - self.expend(k, i)

    # Value function: maximum expected discounted utility given initial caputal
    def value_func(self, k, tol=10 ** (-2)):
        """
        Parameters:
            - k  : (current) capital.
            - tol: absolute distance to steady state capital at which the model
                   will be considered to have reached its steady state.
        """

        if abs(k - self.kss) > tol:
            # If steady state has not been reached, find the optimal capital
            # for the next period and continue computing the value recursively.
            k1 = self.k1Func(k)
            i = k1 - k * (1 - self.delta)
            return self.flow(k, i) + self.beta * self.value_func(k1, tol)

        else:
            # If steady state is reached return present discounted value
            # of all future flows (which will be identical)
            return self.flow(self.kss, self.kss * self.delta) / (1 - self.beta)

    # Derivative of adjustment cost with respect to investment
    def j_i(self, i, k):
        iota = i / k - self.delta
        return iota * self.omega

    # Derivative of adjustment cost with respect to capital.
    def j_k(self, i, k):
        iota = i / k - self.delta
        return -(iota ** 2 / 2 + iota * self.delta) * self.omega

    # Error in the euler Equation implied by a k_0, k_1, k_2 triad.
    # This can be solved to obtain the triads that are consistent with the
    # equation.
    def eulerError(self, k0, k1, k2):

        # Compute implied investments at t=0 and t=1.
        i0 = k1 - (1 - self.delta) * k0
        i1 = k2 - (1 - self.delta) * k1

        # Compute implied error in the Euler equation
        error = (1 + self.j_i(i0, k0)) * self.P - (
            (1 - self.tau) * self.f_k(k1)
            + (
                (1 - self.delta)
                + (1 - self.delta) * self.j_i(i1, k1)
                - self.j_k(i1, k1)
            )
            * self.P
            * self.beta
        )

        return error

    # Find the k_2 implied by the euler equation for an initial k_0, k_1.
    def k2(self, k0, k1):

        # Find the k2 that is consistent with the Euler equation
        sol = optimize.root_scalar(
            lambda x: self.eulerError(k0, k1, x), x0=k0, x1=self.kss
        )

        # Return exception if no compatible capital is found
        if sol.flag != "converged":
            raise Exception("Could not find capital value satisfying Euler equation")

        return sol.root

    # Find the capital trajectory implied by the euler equation for
    # an initial k_0, k_1.
    def shoot(self, k0, k1, t):
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
        for i in range(2, t):

            try:
                k[i] = self.k2(k[i - 2], k[i - 1])
            except:
                # If at some point no solution can be found stop simulation.
                k[i:] = k[i]
                return k

            if k[i] < 0 or (abs(k[i] - self.kss) > 2 * abs(k0 - self.kss)):
                # If a negative or diverging capital is obtained, stop
                # simulation
                k[i:] = k[i]
                return k

        return k

    # Shooting algorithm to find k_1 given k_0.
    def find_k1(self, k0, T=30, tol=10 ** (-3), maxiter=200):
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
        top = max(self.kss, k0)
        bot = min(self.kss, k0)

        for k in range(maxiter):

            # Simulate capital dynamics at the midpoint of the
            # current interval.
            init = (top + bot) / 2
            path = self.shoot(k0, init, T)

            # Check the final value of capital
            k_f = path[-1]

            if np.isnan(k_f):
                bot = init
            else:
                if abs(k_f - self.kss) < tol:
                    # Stop if capital reaches and stays at
                    # the steady state
                    return init
                else:
                    if k_f >= self.kss:
                        # If capital ends up above steady state,
                        # we are underestimating k_1.
                        top = init
                    else:
                        # If capital ends up below steady state,
                        # we are overestimating k_1
                        bot = init

        return init

    # Construction of the policy rule by solving for k_1 given
    # k_0 over a grid of points and then finding an interpolating
    # function
    def solve(self, k_min=10 ** (-4), n_points=50):
        """
        Parameters:
            - k_min   : minimum value of capital at which the policy rule will
                        be solved for.
            - n_points: number of points at which to numerically solve for the
                        policy rule.
        """

        # Set the price of capital after ITC
        self.P = 1 - self.zeta

        # First find steady state capital (in case parameters were changed)
        self.kss = (
            (1 - (1 - self.delta) * self.beta)
            * self.P
            / ((1 - self.tau) * self.alpha * self.psi)
        ) ** (1 / (self.alpha - 1))

        # Create k_0 grid
        k_max = 4 * self.kss
        k0 = np.linspace(k_min, k_max, n_points)
        k1 = np.zeros(len(k0))

        # Find k_0 at each point in the grid
        for i in range(len(k0)):

            k1[i] = self.find_k1(k0[i])

        # Interpolate over the grid to get a continuous
        # function
        self.k1Func = interpolate.interp1d(k0, k1)

    # Simulation of capital dynamics from a starting k_0 for a number of
    # periods t
    def simulate(self, k0, t):
        k = np.zeros(t)
        k[0] = k0
        for i in range(1, t):
            k[i] = self.k1Func(k[i - 1])
        return k

    # Net investment ratio at t, as a function of marginal value of capital at
    # t+1.
    def iota(self, lam_1):
        iota = (lam_1 / self.P - 1) / self.omega
        return iota

    # Detivative of adjustment costs as a function of lambda(t+1), assuming
    # optimal investment.
    def jkl(self, lam_1):
        iota = self.iota(lam_1)
        jk = -(iota ** 2 / 2 + iota * self.delta) * self.omega
        return jk

    # Plot the marginal value of capital at t implied by the envelope condition,
    # as a function of the marginal value at t+1, at a given level of capital.
    def plotEnvelopeCond(self, k, npoints=10):

        # Create grid for lambda(t+1)
        lam_1 = np.linspace(0, 2, npoints)

        # Compute each component of the envelope condition
        prod = np.ones(npoints) * (1 - self.tau) * self.f_k(k)
        iota = (lam_1 / self.P - 1) / self.omega
        jk = -(iota ** 2 / 2 + iota * self.delta) * self.omega
        inv_gain = -jk * self.beta * self.P
        fut_val = (1 - self.delta) * self.beta * lam_1

        # Plot lambda(t) as a function of lambda(t+1)
        plt.plot(lam_1, prod + inv_gain + fut_val, label="Env. Condition value")
        plt.plot(lam_1, lam_1, linestyle="--", color="k", label="45° line")

        plt.legend()
        plt.title("$\\lambda (t)$ vs $\lambda (t+1)$ at $k =$ %1.2f" % (k))
        plt.xlabel("$\\lambda (t+1)$")
        plt.ylabel("$\\lambda (t)$")

    # Solve for the value of lambda(t) that implies lambda(t)=lambda(t+1) at
    # a given level of capital.
    def lambda0locus(self, k):

        # Set the initial solution guess acording to the level of capital. This
        # is important given that the equation to be solved is quadratic.
        if k > self.kss:
            x1 = 0.5 * self.P
        else:
            x1 = 1.5 * self.P

        bdel = self.beta * (1 - self.delta)

        # Lambda solves the following equation:
        error = (
            lambda x: (1 - bdel) * x
            - (1 - self.tau) * self.f_k(k)
            + self.jkl(x) * self.beta * self.P
        )

        # Search for a solution. The locus does not exist at all k.
        sol = optimize.root_scalar(error, x0=self.P, x1=x1)
        if sol.flag != "converged":
            return np.float("nan")
        else:
            return sol.root

    # Compute marginal value of capital at t using k0,k1 and the envelope
    # condition
    def findLambda(self, k0, k1):

        # Implied investment at t
        i = k1 - (1 - self.delta) * k0
        iota = i / k0 - self.delta

        q1 = iota * self.omega + 1
        lam1 = q1 * self.P

        # Envelope equation
        lam = (
            (1 - self.tau) * self.f_k(k0)
            - self.j_k(i, k0) * self.beta * self.P
            + self.beta * (1 - self.delta) * lam1
        )

        return lam

    # Plot phase diagram of the model
    def phase_diagram(
        self, k_min=0.1, k_max=2, npoints=200, stableArm=False, Qspace=False
    ):
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
        k = np.linspace(k_min * self.kss, k_max * self.kss, npoints)

        # Define normalization factor in case we are in Qspace
        fact = 1
        yLabel = "\\lambda"
        if Qspace:
            fact = 1 / self.P
            yLabel = "q"

        # Plot
        plt.figure()
        # Plot k0 locus
        plt.plot(k, self.P * np.ones(npoints) * fact, label="$\\dot{k}=0$ locus")
        # Plot lambda0 locus
        plt.plot(
            k,
            [self.lambda0locus(x) * fact for x in k],
            label="$\\dot{" + yLabel + "}=0$ locus",
        )
        # Plot steady state
        plt.plot(self.kss, self.P * fact, "*r", label="Steady state")

        # PLot stable arm
        if stableArm:

            if self.k1Func is None:
                raise Exception("Solve the model first to plot the stable arm!")
            else:
                lam = np.array([self.findLambda(k0=x, k1=self.k1Func(x)) for x in k])
                plt.plot(k, lam * fact, label="Stable arm")

        # Labels
        plt.title("Phase diagram")
        plt.xlabel("$k$")
        plt.ylabel("$" + yLabel + "$")
        plt.legend()

        plt.show()


# Additional tools to compute simple transitional dynamics
##########################################################


def pathValue(invest, mod1, mod2, k0, t):
    """
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
    """

    # Initialize capital and value (utility)
    k = np.zeros(t + 1)
    k[0] = k0
    value = 0

    # Compute capital and utility flows until time t-1
    for i in range(t):
        flow = mod1.flow(k[i], invest[i])
        value += flow * mod1.beta ** i
        k[i + 1] = k[i] * (1 - mod1.delta) + invest[i]

    # From time t onwards, model 2 prevails and its value function can be used.
    value += (mod1.beta ** t) * mod2.value_func(k[t])

    return value


def structural_change(mod1, mod2, k0, t_change, T_sim, npoints=300):
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
        fobj = lambda x: -1 * pathValue(x, mod1, mod2, k0, t_change)
        inv = optimize.minimize(
            fobj,
            x0=np.ones(t_change) * mod1.kss * mod2.delta,
            options={"disp": True},
            tol=1e-16,
        ).x

    # Find paths of capital and lambda
    k = np.zeros(T_sim)
    lam = np.zeros(T_sim)
    k[0] = k0
    for i in range(0, T_sim - 1):

        if i < t_change:
            # Before the change, investment follows the optimal
            # path computed above.
            k[i + 1] = k[i] * (1 - mod1.delta) + inv[i]
            lam[i] = mod1.findLambda(k[i], k[i + 1])
        else:
            # After the change, investment follows the post-change policy rule.
            k[i + 1] = mod2.k1Func(k[i])
            lam[i] = mod2.findLambda(k[i], k[i + 1])

    lam[T_sim - 1] = mod2.findLambda(k[T_sim - 1], mod2.k1Func(k[T_sim - 1]))

    # Create a figure with phase diagrams and dynamics.
    plt.figure()

    # Plot k,lambda path.
    plt.plot(k, lam, ".k")
    plt.plot(k[t_change], lam[t_change], ".r", label="Change takes effect")

    # Plot the loci of the pre and post-change models.
    k_range = np.linspace(
        0.1 * min(mod1.kss, mod2.kss), 2 * max(mod1.kss, mod2.kss), npoints
    )
    mods = [mod1, mod2]
    colors = ["r", "b"]
    labels = ["Pre-change", "Post-change"]
    for i in range(2):

        # Plot k0 locus
        plt.plot(
            k_range,
            mods[i].P * np.ones(npoints),
            linestyle="--",
            color=colors[i],
            label=labels[i],
        )
        # Plot lambda0 locus
        plt.plot(
            k_range,
            [mods[i].lambda0locus(x) for x in k_range],
            linestyle="--",
            color=colors[i],
        )
        # Plot steady state
        plt.plot(mods[i].kss, mods[i].P, marker="*", color=colors[i])

    plt.title("Phase diagrams and model dynamics")
    plt.xlabel("K")
    plt.ylabel("Lambda")
    plt.legend()

    return {"k": k, "lambda": lam}
