from scipy.stats import norm
import matplotlib.pyplot as plt
import quantecon as qe # all packages is installed if choose anaconda solution except quantecon
# which can be obtained by pip install commend
import numpy as np
from numba import jit
from numpy.random import normal
from scipy import special
from datetime import datetime as dt
from scipy.optimize import newton


# basic parameters
beta  = 0.9901
alpha = 0.3600
delta = 0.0227
A     = 0.3531
phi_1 = 0.8440
phi_2 = 0.5
sigma = 2
nu    = 0.3659
# simulation params
l_ss = 0.33
k_ss = ((1/beta - 1 + delta)/(alpha*A))**(1/(alpha-1))*l_ss
kmin = 0.75*k_ss
kmax = 1.25*k_ss
grid = 501
K    = np.linspace(kmin, kmax, grid)
# iteration criterions
max_iter=10000 # maximum of iteration
tol     =1e-6 # tolerance

# utility function
@jit(nopython=True)
def u(c, l):
    return (c**nu*(1-l)**(1-nu))**(1-sigma)/(1-sigma)


# (c) Tauchen method
def tauchen(mu, sigma_e, rho, lambda_z):
    # no. grid points
    N_z = 2*lambda_z+1
    # value of grid points
    Z = np.asarray([mu+lam*sigma_e/(1-rho**2)**0.5 for lam in range(-lambda_z, lambda_z+1)])
    # mid points
    M = np.asarray([(Z[i]+Z[i+1])/2 for i in range(N_z-1)])
    # transition matrix
    Pi = np.empty((N_z, N_z))
    # fill in probs
    for i in range(N_z):
        for j in range(N_z):
            if j==0:
                Pi[i, j] = special.ndtr((M[j]-(1-rho)*mu-rho*Z[i])/sigma_e)
            elif j<N_z-1:
                Pi[i, j] = special.ndtr((M[j]-(1-rho)*mu-rho*Z[i])/sigma_e) - special.ndtr((M[j-1]-(1-rho)*mu-rho*Z[i])/sigma_e)
            else:
                Pi[i, j] = 1 - special.ndtr((M[j-1]-(1-rho)*mu-rho*Z[i])/sigma_e)
    return Z, Pi
# parameters for stochastic process
lambda_z = 2
N_z      = 2*lambda_z+1
mu       = 0
sigma_e  = 0.007
rho      = 0.95
ZZ, PPi = tauchen(mu, sigma_e, rho, lambda_z)
print('The discrete states are: ')
print(ZZ)
print('Their transition matrix is: ')
print(PPi)

print("Start deriving optimal labor given z, k and kprime")
# FOC for labor
def H(l, z, k, kprime):
    c     = A*np.exp(z)*k**alpha*l**(1-alpha) + (1-delta)*k - kprime
    temp1 = (1-l)*nu*A*np.exp(z)*k**alpha*l**(-alpha)
    temp2 = c*(1-nu)
    return temp1 -temp2
# for each (iz, ik, ikprime), use newton method to find optimal l
# which is equivalent to finding roots of H holding z, k, kprime fixed
# and store it for convenience
# it runs 501*501*5 loops, and takes 400+s.
LL = np.empty((N_z, grid, grid))
for iz in range(N_z):
    for ik in range(grid):
        for ikprime in range(grid):
            def HH(l):
                return H(l, ZZ[iz], K[ik], K[ikprime])
            try:
                pot_root = newton(HH, 0.5)
            # no sultion!
            except:
                pot_root = 1
            LL[iz, ik, ikprime] = pot_root
# modify the matrix
L_dis = np.empty((N_z, grid, grid))
for iz in range(N_z):
    for ik in range(grid):
        for ikprime in range(grid):
            temp = LL[iz, ik, ikprime]
            if (temp>0) & (temp<1):
                L_dis[iz, ik, ikprime]=temp
            else:
                L_dis[iz, ik, ikprime] = L_dis[iz, ik, ikprime-1]
print('optimal labor function discretization finished!')

# (d) (i)
@jit(nopython=True)
def VFI_basic(v):
    v_old=v.copy()
    ikprime_old = np.zeros((N_z, grid))
    v_new = np.ones((N_z, grid))
    ikprime_new = np.zeros((N_z, grid))
    n=0
    # start loop
    while n<=max_iter+1:
        error = 0
        for iz in range(N_z):
            for ik in range(grid):
                v_max_so_far = -99999
                ikprime_so_far = 0
                for ikprime in range(grid):
                    l = L_dis[iz, ik, ikprime]
                    c = A*np.exp(ZZ[iz])*K[ik]**alpha*l**(1-alpha)+(1-delta)*K[ik]-K[ikprime]
                    if c>0:
                        v_z_kprime = 0
                        for izprime in range(N_z):
                            v_z_kprime += v_old[izprime, ikprime] * PPi[iz, izprime]
                        v_z_k = (1-beta)*u(c, l) + beta*v_z_kprime
                        if v_z_k > v_max_so_far:
                            v_max_so_far = v_z_k
                            ikprime_so_far = ikprime
                # update value and policy function
                v_new[iz, ik] = v_max_so_far
                ikprime_new[iz, ik] = ikprime_so_far
                # now errors
                temp_err = v_old[iz, ik]-v_new[iz, ik]
                temp_err1 = v_new[iz, ik]-v_old[iz, ik]
                if temp_err1 > temp_err:
                    temp_err = temp_err1
                if temp_err > error:
                    error = temp_err
        # print(n, error)
        if error < tol:
            return n, v_new, ikprime_new
            break
        else:
            v_old = v_new.copy()
            ikprime_old = ikprime_new.copy()
            n = n+1


# (d) (ii)
@jit(nopython=True)
def VFI_monotonicity(v):
    v_old=v.copy()
    ikprime_old = np.zeros((N_z, grid))
    v_new = np.ones((N_z, grid))
    ikprime_new = np.zeros((N_z, grid))
    n=0
    # start loop
    while n<=max_iter:
        error = 0
        for iz in range(N_z):
            for ik in range(grid):
                v_max_so_far = -99999
                ik_max_so_far = 0
                # exploit policy function monotonicity
                if ik ==0:
                    ikprime_start = 0
                else:
                    ikprime_start = ikprime_old[iz, ik-1]
                # use this to reduce cells for search
                for ikprime in range(ikprime_start, grid):
                    # replace the l function as its discreted values
                    l = L_dis[iz, ik, ikprime]
                    c = A*np.exp(ZZ[iz])*K[ik]**alpha*l**(1-alpha)+(1-delta)*K[ik]-K[ikprime]
                    if c>0:
                        v_z_kprime = 0
                        for izprime in range(N_z):
                            v_z_kprime += v_old[izprime, ikprime] * PPi[iz, izprime]
                        v_z_k = (1-beta)*u(c, l) + beta*v_z_kprime
                        if v_z_k > v_max_so_far:
                            v_max_so_far = v_z_k
                            ikprime_so_far = ikprime
                # update value and policy function
                v_new[iz, ik] = v_max_so_far
                ikprime_new[iz, ik] = ikprime_so_far
                # now errors
                temp_err = v_old[iz, ik]-v_new[iz, ik]
                temp_err1 = v_new[iz, ik]-v_old[iz, ik]
                if temp_err1 > temp_err:
                    temp_err = temp_err1
                if temp_err > error:
                    error = temp_err
        # print(n, error)
        if error < tol:
            return n, v_new, ikprime_new
            break
        else:
            v_old = v_new.copy()
            ikprime_old = ikprime_new.copy()
            n = n+1


# (d) (iii)
@jit(nopython=True)
def VFI_monotonicity_concavity(v):
    v_old=v.copy()
    ikprime_old = np.zeros((N_z, grid))
    v_new = np.ones((N_z, grid))
    ikprime_new = np.zeros((N_z, grid))
    n=0
    # start loop
    while n<=max_iter:
        error = 0
        for iz in range(N_z):
            for ik in range(grid):
                v_max_so_far = -99999
                ik_max_so_far = 0
                # exploit policy function monotonicity
                if ik ==0:
                    ikprime_start = 0
                else:
                    ikprime_start = ikprime_old[iz, ik-1]
                # use this to reduce cells for search
                for ikprime in range(ikprime_start, grid):
                    l = L_dis[iz, ik, ikprime]
                    c = A*np.exp(ZZ[iz])*K[ik]**alpha*l**(1-alpha)+(1-delta)*K[ik]-K[ikprime]
                    if c>0:
                        v_z_kprime = 0
                        for izprime in range(N_z):
                            v_z_kprime += v_old[izprime, ikprime] * PPi[iz, izprime]
                        v_z_k = (1-beta)*u(c, l) + beta*v_z_kprime
                        if v_z_k > v_max_so_far:
                            v_max_so_far = v_z_k
                            ikprime_so_far = ikprime
                        # exploit value function concavity
                        else:
                            break
                # update value and policy function
                v_new[iz, ik] = v_max_so_far
                ikprime_new[iz, ik] = ikprime_so_far
                # now errors
                temp_err = v_old[iz, ik]-v_new[iz, ik]
                temp_err1 = v_new[iz, ik]-v_old[iz, ik]
                if temp_err1 > temp_err:
                    temp_err = temp_err1
                if temp_err > error:
                    error = temp_err
        #print(n, error)
        if error < tol:
            return n, v_new, ikprime_new
            break
        else:
            v_old = v_new.copy()
            ikprime_old = ikprime_new.copy()
            n = n+1

# (d) (iv)
@jit(nopython=True)
def VFI_monotonicity_concavity_accelerator(v):
    v_old=v.copy()
    ikprime_old = np.zeros((N_z, grid))
    v_new = np.ones((N_z, grid))
    ikprime_new = np.zeros((N_z, grid))
    n=0
    # start loop
    while n<=max_iter:
        error = 0
        if n == 10*int(n/10):
            for iz in range(N_z):
                for ik in range(grid):
                    v_max_so_far = -99999
                    ik_max_so_far = 0
                    # exploit policy function monotonicity
                    if ik ==0:
                        ikprime_start = 0
                    else:
                        ikprime_start = ikprime_old[iz, ik-1]
                    # use this to reduce cells for search
                    for ikprime in range(ikprime_start, grid):
                        l = L_dis[iz, ik, ikprime]
                        c = A*np.exp(ZZ[iz])*K[ik]**alpha*l**(1-alpha)+(1-delta)*K[ik]-K[ikprime]
                        if c>0:
                            v_z_kprime = 0
                            for izprime in range(N_z):
                                v_z_kprime += v_old[izprime, ikprime] * PPi[iz, izprime]
                            v_z_k = (1-beta)*u(c, l) + beta*v_z_kprime
                            if v_z_k > v_max_so_far:
                                v_max_so_far = v_z_k
                                ikprime_so_far = ikprime
                            # exploit value function concavity
                            else:
                                break
                    # update value and policy function
                    v_new[iz, ik] = v_max_so_far
                    ikprime_new[iz, ik] = ikprime_so_far
                    # now errors
                    temp_err = v_old[iz, ik]-v_new[iz, ik]
                    temp_err1 = v_new[iz, ik]-v_old[iz, ik]
                    if temp_err1 > temp_err:
                        temp_err = temp_err1
                    if temp_err > error:
                        error = temp_err
            #print(n, error)
            if error < tol:
                return n, v_new, ikprime_new
                break
            else:
                v_old = v_new.copy()
                ikprime_old = ikprime_new.copy()
                n = n+1
        # exploit accelerators (Howard's Improvement Algorithm)
        else:
            for iz in range(N_z):
                for ik in range(grid):
                    ikprime1 = ikprime_old[iz, ik]
                    l = L_dis[iz, ik, int(ikprime1)]
                    c = A*np.exp(ZZ[iz])*K[ik]**alpha*l**(1-alpha)+(1-delta)*K[ik]-K[int(ikprime1)]
                    v_z_kprime = 0
                    for izprime in range(N_z):
                        v_z_kprime += v_old[izprime, int(ikprime1)] * PPi[iz, izprime]
                    # update value and policy function
                    v_new[iz, ik] = (1-beta)*u(c, l) + beta*v_z_kprime
                    # now errors
                    temp_err = v_old[iz, ik]-v_new[iz, ik]
                    temp_err1 = v_new[iz, ik]-v_old[iz, ik]
                    if temp_err1 > temp_err:
                        temp_err = temp_err1
                    if temp_err > error:
                        error = temp_err
            #print(n, error)
            if error < tol:
                return n, v_new, ikprime_new
                break
            else:
                v_old = v_new.copy()
                ikprime_old = ikprime_new.copy()
                n = n+1

# (d) (v)
small_grid = 51
large_grid = 501
@jit(nopython=True)
def VFI_grid(v, grid=small_grid):
    Ktemp = np.linspace(kmin, kmax, grid)
    v_old=v.copy()
    ikprime_old = np.zeros((N_z, grid))
    v_new = np.ones((N_z, grid))
    ikprime_new = np.zeros((N_z, grid))
    n=0
    
    while n<=max_iter:
        error = 0
        if n == 10*int(n/10):
            for iz in range(N_z):
                for ik in range(grid):
                    v_max_so_far = -99999
                    ik_max_so_far = 0
                    if ik ==0:
                        ikprime_start = 0
                    else:
                        ikprime_start = ikprime_old[iz, ik-1]

                    for ikprime in range(ikprime_start, grid):
                        l = L_dis1[iz, ik, ikprime]
                        c = A*np.exp(ZZ[iz])*Ktemp[ik]**alpha*l**(1-alpha)+(1-delta)*Ktemp[ik]-Ktemp[ikprime]
                        if c>0:
                            v_z_kprime = 0
                            for izprime in range(N_z):
                                v_z_kprime += v_old[izprime, ikprime] * PPi[iz, izprime]
                            v_z_k = (1-beta)*u(c, l) + beta*v_z_kprime
                            if v_z_k > v_max_so_far:
                                v_max_so_far = v_z_k
                                ikprime_so_far = ikprime
                            else:
                                break
                    v_new[iz, ik] = v_max_so_far
                    ikprime_new[iz, ik] = ikprime_so_far

                    temp_err = v_old[iz, ik]-v_new[iz, ik]
                    temp_err1 = v_new[iz, ik]-v_old[iz, ik]
                    if temp_err1 > temp_err:
                        temp_err = temp_err1
                    if temp_err > error:
                        error = temp_err
            #print(n, error)
            if error < tol:
                return n, v_new, ikprime_new
                break
            else:
                v_old = v_new.copy()
                ikprime_old = ikprime_new.copy()
                n = n+1
        
        else:
            for iz in range(N_z):
                for ik in range(grid):
                    ikprime1 = ikprime_old[iz, ik]
                    l = L_dis1[iz, ik, int(ikprime1)]
                    c = A*np.exp(ZZ[iz])*Ktemp[ik]**alpha*l**(1-alpha)+(1-delta)*Ktemp[ik]-Ktemp[int(ikprime1)]
                    v_z_kprime = 0
                    for izprime in range(N_z):
                        v_z_kprime += v_old[izprime, int(ikprime1)] * PPi[iz, izprime]
                    v_new[iz, ik] = (1-beta)*u(c, l) + beta*v_z_kprime
                    
                    temp_err = v_old[iz, ik]-v_new[iz, ik]
                    temp_err1 = v_new[iz, ik]-v_old[iz, ik]
                    if temp_err1 > temp_err:
                        temp_err = temp_err1
                    if temp_err > error:
                        error = temp_err
            #print(n, error)
            if error < tol:
                return n, v_new, ikprime_new
                break
            else:
                v_old = v_new.copy()
                ikprime_old = ikprime_new.copy()
                n = n+1 

print('which method to choose: ')
print('0. Basic \n 1. +monotonicity \n 2. +concavity \n 3. +accelerators \n 4. +multi grid')
choice = int(input('input [0, 1, 2, 3, 4]') )
# candidate methods
functions = [VFI_basic, VFI_monotonicity, VFI_monotonicity_concavity, VFI_monotonicity_concavity_accelerator, VFI_grid]
# set up initial guess
v_0 = np.zeros((N_z, grid))
# how much time elapsed
qe.util.tic()
if choice ==4:
    # step 1 small grid
    large_grid = 501
    small_grid = 51
    iklow = np.zeros((N_z, large_grid)).astype('int')
    ikupp = np.zeros((N_z, large_grid)).astype('int')
    kweight = np.zeros((N_z, large_grid))
    K_small = np.linspace(kmin, kmax, small_grid)
    K_large = np.linspace(kmin, kmax, large_grid)
    # build another optimal labor matrix
    LL1 = np.empty((N_z, small_grid, small_grid))
    for iz in range(N_z):
        for ik in range(small_grid):
            for ikprime in range(small_grid):
                def HH(l):
                    return H(l, ZZ[iz], K_small[ik], K_small[ikprime])
                try:
                    pot_root = newton(HH, 0.5)
                # no sultion!
                except:
                    pot_root = 1
                LL1[iz, ik, ikprime] = pot_root
    # modify the matrix
    L_dis1 = np.empty((N_z, small_grid, small_grid))
    for iz in range(N_z):
        for ik in range(small_grid):
            for ikprime in range(small_grid):
                temp = LL1[iz, ik, ikprime]
                if temp<1:
                    L_dis1[iz, ik, ikprime]=temp
                else:
                    L_dis1[iz, ik, ikprime] = L_dis1[iz, ik, ikprime-1]
    v_0 = np.zeros((N_z, small_grid))
    res0 = VFI_grid(v_0)   
    v00 = res0[1]
    # step 2 kweight
    for iz in range(N_z):
        for ik_large in range(large_grid):
            for ik_small in range(small_grid):
                if K_small[ik_small]>K_large[ik_large]:
                    iklow[iz, ik_large] = max(0, ik_small-1)
                    ikupp[iz, ik_large] = ik_small
                    kweight[iz, ik_large] = (K_large[ik_large] - K_small[iklow[iz, ik_large]]) / (K_small[ikupp[iz, ik_large]] - K_small[iklow[iz, ik_large]])
                    break
            if ik_large == large_grid - 1:
                iklow[iz, ik_large] = small_grid - 2
                ikupp[iz, ik_large] = small_grid - 1
                kweight[iz, ik_large] = (K_large[ik_large] - K_small[iklow[iz, ik_large]]) / (K_small[ikupp[iz, ik_large]] - K_small[iklow[iz, ik_large]])
    # step 3 vfine
    vfine = np.ones((N_z, large_grid))*(-999)
    for iz in range(N_z):
        for ik_large in range(large_grid):
            vfine[iz, ik_large] = (1-kweight[iz, ik_large])*v00[iz, iklow[iz, ik_large]] + kweight[iz, ik_large]*v00[iz, ikupp[iz, ik_small]]
    res = VFI_monotonicity_concavity_accelerator(vfine)
else:
    res = functions[choice](v_0)
qe.util.toc()

# labor for all grid of z and k
LLL = np.empty((N_z, grid))
for iz in range(N_z):
    for ik in range(grid):
        ikprime = int(res[2][iz, ik])
        LLL[iz, ik] = L_dis[iz, ik, ikprime]

identifier = int(dt.timestamp(dt.today()))
# plot the value function, policy functions against z and k
fig, axes = plt.subplots(3, 1, figsize=(8, 15))
for i in range(3):
    for iz in range(5):
        if i==0:
            axes[i].plot(K, res[1][4-iz], label='z=z('+str(4-iz)+')')
            axes[i].set_title("Value Function")
        elif i==1:
            KK = [K[int(ikprime)] for ikprime in res[2][4-iz]]
            axes[i].plot(K, KK, label='z=z('+str(4-iz)+')')
            axes[i].set_title("Policy Function for Next Period's Capital")
        else:
            axes[i].plot(K, LLL[4-iz], label='z=z('+str(4-iz)+')')
            axes[i].set_title("Policy Function for Labor")
    axes[i].legend()
    axes[i].set_xlabel("Capital")
plt.savefig(str(identifier)+'HW101.pdf', dpi=250)
plt.show()
# save as pdf file


# (e)

# generate random sequence of shocks
mc = qe.MarkovChain(PPi)
Z_path = mc.simulate(ts_length=10000)
# a function takes initial capital and shock sequence as given, generate the dynamics
def simulate(izpath = Z_path, ik_0 = 0):
    IK = []
    IK.append(ik_0)
    IZ = izpath
    Z, KK, L, Y, C, I = [], [], [], [], [], []
    # sequence of all relevant variables
    for it in range(0, 10000):
        # shock
        shock = ZZ[IZ[it]]
        Z.append(shock)
        # next period capital index
        ikprime = int(res[2][IZ[it], IK[it]])
        IK.append(ikprime)
        # capital
        capital = K[IK[it]]
        KK.append(capital)
        # labor
        labor = LLL[IZ[it], IK[it]]
        L.append(labor)
        # output
        y = A*np.exp(shock)*capital**alpha*labor**(1-alpha)
        Y.append(y)
        # consumption
        cons = y + (1-delta)*capital-K[ikprime]
        C.append(cons)
        # investment
        invest = y - cons
        I.append(invest)
    return [[Z, KK], [L, Y], [C, I]]

# drop the first 200 obs, and then obtain 500 obs
Y, L, C, I= simulate()[1][1][200:700], simulate()[1][0][200:700], simulate()[2][0][200:700], simulate()[2][1][200:700]
# get the relative change
y_bar, l_bar, c_bar, i_bar = sum(Y)/len(Y), sum(L)/len(L), sum(C)/len(C), sum(I)/len(I)
yy = [(y-y_bar)/y_bar for y in Y]
ll = [(l-l_bar)/l_bar for l in L]
cc = [(c-c_bar)/c_bar for c in C]
ii = [(i-i_bar)/i_bar for i in I]

# plot the latter three sequences against output dynamics
fig, axes = plt.subplots(3, 1, figsize=(8, 15))
LIST = [ii, cc, ll]
names = ['investment', 'consumption', 'labor']
for i in range(3):
    axes[i].plot(yy, label='output')
    axes[i].plot(LIST[i], label=names[i])
    axes[i].set_xlim(0, 500)
    axes[i].set_ylim(-0.5, 0.5)
    axes[i].set_title(names[i])
    axes[i].legend()
plt.savefig(str(identifier)+'HW102.PDF', dpi=250)
plt.show()

# (f)
# save the data and read it in matlab to conduct business cycle analysis
Sim_data1 = np.asarray([Y, C, I, L])
Sim_data1 = Sim_data1.T
np.savetxt(str(identifier)+'.txt', Sim_data1, delimiter=',')
print('dynamics generated!')

print('now go to matlab!')