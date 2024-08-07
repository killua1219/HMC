from LinearRegression.create_line import get_line
import matplotlib.pyplot as plt
import numpy as np
import Analysis as sys
from Algorithms.metropolis import metropolis
from Algorithms.MyNUTS import MyNUTS
from Algorithms.NUTS import NUTS
from Algorithms.HMC import HMC
from Algorithms.MALA import MALA
from Hamiltonian.H import H
from Hamiltonian.PB_H import PB_H
from Compare import compare
from LinearRegression.Linear import Linear


def plot_retta(sample_x, sample_y, x, y, coef, interc, sample, label):
    plt.errorbar(sample_x, sample_y, xerr=10, yerr=15, fmt='.', capsize=5, color = 'navy', label='Data Points', alpha = 0.8)

    real_y = np.linspace(np.min(sample_x)-5, np.max(sample_x)+ 5, 1000) * coef + interc
    plt.plot(np.linspace(np.min(sample_x)-5, np.max(sample_x)+ 5, 1000), real_y, color = 'red',alpha = 0.8, label = 'True')

    sample_coef = np.mean(sample[:,-2])
    sample_interc = np.mean(sample[:,-1])
    for s in sample[:,:-2].T:
        q1,q2,q3 = np.percentile(s, [5,50,95])
        asymmetric_error = [np.array([q2-q1]),np.array([q3-q2])]
        plt.errorbar(q2, q2*sample_coef + sample_interc, yerr=asymmetric_error, fmt='o', color = 'blue', alpha=0.9)
    
    k = [np.linspace(np.min(sample_x)-5, np.max(sample_x)+ 5,1000) * q[-2] +q[-1] for q in sample]
    l1, l2, l3 = np.percentile(k, [5,50, 95], axis = 0)
    plt.plot(np.linspace(np.min(sample_x)-5, np.max(sample_x)+ 5,1000), l2, color = 'blue', alpha = 0.9, label = "Esteemed")
    plt.fill_between(np.linspace(np.min(sample_x)-5, np.max(sample_x)+ 5,1000), l1, l3, alpha = 0.3)
    plt.legend()
    sample_x = np.sort(sample_x)
    plt.title(label)
    plt.show()

    

rng = np.random.default_rng(1234)
err_x= 5
err_y = 10
coef = 30
interc = 50
num_points = 30
lim = (40,60)
sample_x,sample_y, x, y = get_line(num_points, coef, interc, err_x, err_y, lim= lim, rng = rng)


M = Linear(num_points+2, sample = np.array([sample_x, sample_y]) , parameters = np.array([err_x,err_y]))

plt.errorbar(sample_x, sample_y, xerr=10, yerr=15, fmt='.', capsize=5, label='Data Points')
plt.show()
start_q = np.zeros(num_points+2)
start_q[:-2] = sample_x.copy()
start_q[-2:] = np.array([45.,30.])

def mass_matrix(dim, rng):
    return 5*np.ones(dim)
H1 = H(5*np.eye(num_points+2))
H2 = PB_H(mass_matrix)


nuts =  NUTS(M, H1, dt = 0.05 , rng=rng)
mynuts = MyNUTS(M, H1, dt = 0.05, rng=rng)
pb = MyNUTS(M, H2, dt = 0.05, rng=rng)
metropolis = metropolis(M, .1, rng = rng)
mala = MALA(M, dt = 0.001, rng=rng)
hmc = HMC(M, H1, L = 50, dt = 0.05 , rng = rng)

a = mynuts

algs = [hmc,nuts, mynuts, pb]
iteration = 15000
all_m = compare(algs, iteration, start_q, 5000,20)
all_m = [np.load(a.alg_name()+".npy") for a in algs]
'''
for m in all_m:
    sys.plot_IAT(m, 50)
    '''

for a, m in zip(algs, all_m):
    print(a.alg_name(), np.mean(m[:,-2]), sys.var_mean_real(m, np.mean(m, axis = 0))[0][-2])
    print(a.alg_name(), np.mean(m[:,-1]), sys.var_mean_real(m, np.mean(m, axis = 0))[0][-1])
for a, m in zip(algs, all_m):
    print(np.load("info"+a.alg_name()+".npy"))
    plot_retta(sample_x, sample_y, x, y, coef, interc, m, a.alg_name())
plt.show()