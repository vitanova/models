from scipy.stats import norm
import matplotlib.pyplot as plt
import quantecon as qe # all packages is installed if choose anaconda solution except quantecon
# which can be obtained by pip install commend
import numpy as np
from numba import jit
from numpy.random import normal, uniform
from scipy import special
from datetime import datetime as dt
import pandas as pd
from scipy import integrate
from scipy.optimize import minimize, minimize_scalar
from mpl_toolkits.mplot3d import Axes3D
import collections

"""
In this project, we are going to implement an algorithm of
Bayesian estimation for Discrete Choice Dynamic Programming models
following Imai et al. (2009), Norets (2009), and Ishihara and Ching (2016), 
where the formulation of recursive likelihood is based on Reich (2018).

The simulation is based on the classicial bus engine replacement problem by Rust (1987).
But as Norets (2009), we assume the unobserved variable (which can be thought as the quality)
follow an AR(1) process.

For simplicity, our only parameter of interest is \theta, the marginal cost for additional mile of driving

We first do a brute-force two step procedure, similar to Rust (1987), but substitute the outer
maximization loop for MLE with MCMC for Bayesian estimation.

Then, we consider the integrated approach, which uses past result to approximate the expected value function


"""


beta  = 0.9
rc    = 15.0
theta = 0.5
rho   = 0.7
sigma = 1.0

N_x, N_e0, N_e1 = 10, 10, 10
N_t = 5000
tol = 1e-2


def Epsilon(x, N):
    beta, rc, theta, rho, sigma = x
    sigma_z = sigma/(1-rho**2)**0.5
    Eps = np.zeros(N+1)
    for i_eps in range(N+1):
        Eps[i_eps] = sigma_z * special.ndtri(i_eps/N)
    return Eps

@jit(nopython=True)
def norm_pdf(x):
    return np.exp(-x**2/2)/(2*np.pi)**0.5

@jit(nopython=True)
def Z(x, Eps):
    beta, rc, theta, rho, sigma = x
    sigma_z = sigma/(1-rho**2)**0.5
    N = len(Eps)-1
    Z = np.zeros(N)
    for i in range(N):
        Z[i] = N * sigma_z * (norm_pdf(Eps[i]/sigma_z) - norm_pdf(Eps[i+1]/sigma_z))
    return Z

def AddaCooper(x, N):
    beta, rc, theta, rho, sigma = x
    sigma_z = sigma/(1-rho**2)**0.5
    Eps = Epsilon(x, N)
    ZZ = Z(x, Eps)
    PPi = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            temp = integrate.quad(lambda t: np.exp(-t**2/(2*sigma_z**2)) * (special.ndtr((Eps[j+1]-rho*t)/sigma)
                                                                          -special.ndtr((Eps[j]-rho*t)/sigma)),
                                                                          Eps[i], Eps[i+1])[0]
            PPi[i, j] = temp * N /(2*np.pi*sigma_z**2)**0.5
    return ZZ, PPi

def solveStationary(A):
    """
    x = xA where x is the answer
    x - xA = 0
    x( I - A ) = 0 and sum(x) = 1
    """
    n = A.shape[0]
    a = np.eye( n ) - A
    a = np.vstack( (a.T, np.ones( n )) )
    b = np.matrix( [0] * n + [ 1 ] ).T
    return np.linalg.lstsq( a, b )[0]


x = (beta, rc, theta, rho, sigma)
ZZ, PPi = AddaCooper(x, N_e0)
"""
ssd = solveStationary(PPi)
ssd = ssd.reshape(N_e1)
ssd = np.array(ssd)[0]
"""
ssd = np.ones(N_e1)/N_e1
ZZ1 = ZZ * (1-rho**2)**0.5


VF_old = np.ones((N_x, N_e0, N_e1))*-100
PF_old = np.zeros((N_x, N_e0, N_e1))

@jit(nopython=True)
def update_VF_PF(VF_old, PF_old, theta, rc):
    VF_new = VF_old.copy()
    PF_new = PF_old.copy()
    for i_x in np.arange(0, N_x)[::-1]:
        for i_e0 in range(N_e0):
            if i_x == N_x-1:
                for i_e1 in range(N_e1):
                    temp1 = -rc + beta*ssd@VF_old[0]@ssd
                    VF_new[i_x, i_e0, i_e1] = temp1 + ZZ1[i_e1]
                    PF_new[i_x, i_e0, i_e1] = 1
            else:
                for i_e1 in range(N_e1):
                    EVF = PPi[i_e0]@VF_old[i_x+1]@ssd
                    temp0 = -(i_x+1) * theta + beta * EVF + ZZ[i_e0]
                    temp1 = -rc + beta*ssd@VF_old[0]@ssd+ ZZ1[i_e1]
                    if temp0 < temp1:
                        VF_new[i_x, i_e0, i_e1] = temp1
                        PF_new[i_x, i_e0, i_e1] = 1
                    else:
                        VF_new[i_x, i_e0, i_e1] = temp0
                        PF_new[i_x, i_e0, i_e1] = 0
    return VF_new, PF_new

@jit(nopython=True)
def update_simple(VF_old, PF_old, theta, rc):
    VF_new = VF_old.copy()
    for i_x in np.arange(0, N_x)[::-1]:
        for i_e0 in range(N_e0):
            for i_e1 in range(N_e1):
                if PF_new[i_x, i_e0, i_e1] == 1:
                    VF_new[i_x, i_e0, i_e1] = -rc + beta*ssd@VF_old[0]@ssd+ ZZ1[i_e1]
                else:
                    VF_new[i_x, i_e0, i_e1] = -(i_x+1) * theta + beta * PPi[i_e0]@VF_old[i_x+1]@ssd + ZZ[i_e0]
    return VF_new, PF_old

@jit
def iter_VF_PF(VF_old, PF_old, theta, rc):
    for i_iter in range(2000):
        VF_new, PF_new = update_VF_PF(VF_old, PF_old, theta, rc)
        VF_nmo = VF_new - VF_old
        err = np.sum(np.abs(VF_nmo))
        #print(i_iter, err)
        if err < tol:
            break
        else:
            VF_old = VF_new
            PF_old = PF_new 
    return VF_new, PF_new


def gen_obs(N_t, PF_old, theta, rc):
    mc = qe.MarkovChain(PPi)
    sim_ix  = [0]
    sim_ie0 = [qe.DiscreteRV(ssd).draw(k=1)[0]]
    sim_ie1 = [qe.DiscreteRV(ssd).draw(k=1)[0]]

    for t in range(1, N_t):
        decision = PF_old[sim_ix[t-1], sim_ie0[t-1], sim_ie1[t-1]]
        if decision == 0:
            sim_ix.append(sim_ix[t-1]+1)
            sim_ie0.append(mc.simulate(ts_length=2, init=sim_ie0[t-1])[1])
            sim_ie1.append(qe.DiscreteRV(ssd).draw(k=1)[0])
        else:
            sim_ix.append(0)
            sim_ie0.append(qe.DiscreteRV(ssd).draw(k=1)[0])
            sim_ie1.append(qe.DiscreteRV(ssd).draw(k=1)[0])

    #plt.hist(sim_ix, bins='auto')
    data = collections.Counter(sim_ix)
    data = dict(data)
    #print(data)
    array = np.array(list(data.items()), dtype=int)
    nmax = np.max(array, axis=0)[0]
    addi = np.vstack((np.arange(nmax+1, N_x+1), np.zeros(N_x-nmax-1+1))).T
    array = np.append(array, addi, axis=0)
    df = pd.DataFrame(array, columns=['milage', 'obs'])
    df['rep'] = -df['obs'].diff()
    df = df.fillna(0)
    ret_obs = df.rep.values
    ret_obs = ret_obs[1:]
    return ret_obs

@jit
def log_ret_probs(PF_old):
    inv_PF = PF_old.copy()
    for i in range(len(inv_PF)):
        inv_PF[i] = np.ones((N_e0, N_e1)) - PF_old[i]
    prob_ret = np.zeros(len(inv_PF))
    pnot_ret= PF_old.copy()
    pnot_ret[0] = np.ones((N_e0, N_e1))/(N_e0*N_e1)
    prob_ret[0] = 1 - np.sum(pnot_ret[0])
    #IMPORTANT: 
    #the dist of epsilon_0 for x = 0 is NOT uniform! 
    for i in range(1, len(inv_PF)):
        new_mat = np.sum(pnot_ret[i-1], axis=1)
        new_mat = new_mat.reshape((N_e0, 1))
        updated = np.transpose(PPi) @ new_mat
        expand  = np.repeat(updated, N_e1, axis=1)/N_e1
        not_ret = inv_PF[i] * expand
        ret     = PF_old[i] * expand
        prob_ret[i] = max(np.sum(ret), 1e-16)
        pnot_ret[i] = not_ret
    return np.log(prob_ret)

    
@jit
def log_theta(x):
	# used for MLE
    theta, rc = x
    VF_old = np.ones((N_x, N_e0, N_e1))*-100
    PF_old = np.zeros((N_x, N_e0, N_e1))
    VF_old, PF_old = iter_VF_PF(VF_old, PF_old, theta=theta, rc=rc)
    log_probs = log_ret_probs(PF_old)
    value = log_probs @ ret_obs
    print(theta, rc, -value)
    return -value

#iter_VF_PF(VF_old, PF_old, theta, rc)

"""
Now, consider bayesian estimation

first, generate a sample of observations

"""
@jit
def sample_lik(VF_old, PF_old, ret_obs, theta, rc):
    log_probs = log_ret_probs(PF_old)
    sum_logs = log_probs @ ret_obs
    return sum_logs

@jit
def mcmc(ret_obs, guesses, N_mc):
    theta, rc = guesses
    mc_theta_rc = np.ones((N_mc, 2))
    mc_theta_rc[0] = np.array([theta, rc])

    VF_old = np.ones((N_x, N_e0, N_e1))*-100
    PF_old = np.zeros((N_x, N_e0, N_e1))
    
    VF_old, PF_old = iter_VF_PF(VF_old, PF_old, theta=theta, rc = rc)
    VF_theta = np.zeros((N_mc, N_x, N_e0, N_e1))
    PF_theta = np.zeros((N_mc, N_x, N_e0, N_e1))
    VF_theta[0], PF_theta[0] = VF_old, PF_old
    logliks     = np.ones(N_mc)
    logliks[0]  = sample_lik(VF_old, PF_old, ret_obs, theta=theta, rc = rc)
    
    for i in range(1, N_mc):
        # update theta, holding rc constant
        theta0, rc0 = mc_theta_rc[i-1]
        theta1 = theta0 + 0.01*np.random.normal()
        prop1 = np.array([theta1, rc0])
        ind = np.abs(mc_theta_rc-prop1).sum(axis=1).argmin()
        VF_pre, PF_pre = VF_theta[ind], PF_theta[ind]
        VF_new, PF_new = update_VF_PF(VF_pre, PF_pre, theta=theta1, rc=rc0)

        lik_new = sample_lik(VF_new, PF_new, ret_obs, theta1, rc0)
        lik_old = logliks[i-1]
        ratio = np.exp(lik_new - lik_old)

        if ratio >= 1:
            mc_theta_rc[i, 0] = theta1
        else:
            if np.random.uniform() <= ratio:
                mc_theta_rc[i, 0] = theta1
            else:
                mc_theta_rc[i, 0] = theta0
        
        # update rc, with the updated theta
        rc1 = rc0 + 0.1*np.random.normal()
        theta2 = mc_theta_rc[i, 0]
        prop2 = np.array([theta2, rc1])
        ind = np.abs(mc_theta_rc-prop2).sum(axis=1).argmin()
        VF_pre, PF_pre = VF_theta[ind], PF_theta[ind]
        VF_new, PF_new = update_VF_PF(VF_pre, PF_pre, theta=theta2, rc=rc1)
        VF_theta[i], PF_theta[i] = VF_new, PF_new
        lik_new2 = sample_lik(VF_new, PF_new, ret_obs, theta2, rc1)

        ratio = np.exp(lik_new2 - lik_new)

        if ratio >= 1:
            mc_theta_rc[i, 1] = rc1
            logliks[i] = lik_new2
        else:
            if np.random.uniform() <= ratio:
                mc_theta_rc[i, 1] = rc1
                logliks[i] = lik_new2
            else:
                mc_theta_rc[i, 1] = rc0
                logliks[i] = lik_new
        if i%100==0:
            print(i, mc_theta_rc[i])

    return mc_theta_rc


theta = 1.0
rc    = 15.0
VF_old, PF_old = iter_VF_PF(VF_old, PF_old, theta=theta, rc=rc)
ret_obs = gen_obs(N_t, PF_old, theta, rc)
print(f'true values: {theta}, {rc}')

guess = [0.7, 13.0]
print(f'init guess: {guess[0]}, {guess[1]}')
qe.util.tic()
res = mcmc(ret_obs, guess, 10000)
qe.util.toc()

mc_theta_rc = res
burn = len(mc_theta_rc)//2
mc_t_t = mc_theta_rc[burn:]
mc_theta_t, mc_rc_t = mc_t_t[:, 0], mc_t_t[:, 1]
fig, ax = plt.subplots()
plt.hist(mc_theta_t, bins='auto')
plt.show()
fig, ax = plt.subplots()
plt.hist(mc_rc_t, bins='auto')
plt.show()

mc_t_s = mc_theta_t.copy()
mc_t_s.sort()
mcts90 = mc_t_s[[250, 4750]]
print(mcts90)

mc_r_s = mc_rc_t.copy()
mc_r_s.sort()
mcrs90 = mc_r_s[[250, 4750]]
print(mcrs90)