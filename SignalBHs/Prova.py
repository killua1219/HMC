from SignalBHs.GenerateData import generate_data, bursts
from SignalBHs.ModelBHs import BHsSignal
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


def extract_midpoints(bounds, N, rng):
    start_q = []
    for i in range(N):
        for b in bounds:
            start_q.append(        rng.uniform(  b[0] , b[1]  )         )
    start_q[1::5] = [(i+1)*bounds[1][1]/(N+1) for i in range(N)]
    return start_q

def what_u_get(m, t, signal, noise, Nsources):
    x = [bursts(t, Nsources, q) for q in m]
    l1, l2, l3 = np.percentile(x, [10,50, 90], axis = 0)
    plt.plot(t, l2, color = 'blue', alpha = 0.8, label = "Esteemed")
    plt.plot(t, signal , color = 'red', alpha = 0.8, label = "True signal")
    #plt.plot(t, signal, color = 'green', alpha = 0.8, label = "True")
    plt.fill_between(t, l1, l3, alpha = 0.3)
    plt.legend()
    plt.show()
    
    
    

rng = np.random.default_rng(1234)
Nsources = 30
bounds = [(2. ,7.), (0., 15.), (.1, 1.), (.75, 4.), (0, 2*np.pi)]
start_q = np.array(extract_midpoints(bounds, Nsources, rng))

t, signal, real_q, noise = generate_data( bounds, Nsources, sampling_frequency=16, sigma_noise=0.8, rng = rng)

plt.plot(t, signal + noise)
plt.show()
bounds = [(2. ,7.), (0., 15.), (.1, 1.3), (.75, 4.), (0, 2*np.pi)]

M = BHsSignal( signal, Nsources, bounds = bounds*Nsources)

def mass_matrix(dim, rng):
    return rng.lognormal(.01,1.0, dim)
H2 = PB_H(mass_matrix)
H1 = H(np.eye(Nsources*5))
m, rj = MALA(M, 0.0001, rng=rng).run(5000, start_q)
#m, rj = MyNUTS(M, H1, dt = 0.001, rng=rng).run(5000, start_q)
#m, rj = metropolis(M,.01, rng=rng).run(50000, start_q)
what_u_get(m, t, signal, noise, Nsources)
sys.plot_IAT(m[4000:,:], 5)
'''
a=[]
t=[]
s=[]
w=[]
phi=[]
for i in range(Nsources):
    a.append(m[:, i*5])
    t.append(m[:, i*5+1])
    s.append(m[:, i*5+2])
    w.append(m[:, i*5+3])
    phi.append(m[:, i*5+4])
a = np.column_stack([np.ravel(a),np.ravel(t),np.ravel(s),np.ravel(w),np.ravel(phi)])
    
fig, ax = plt.subplots(5,5, figsize= (8,8))

for i in range(5):
    for j in range(i+1):
        if i == j:
            for k in range(Nsources):
                ax[i,j].axvline(real_q[k*5 + j])
        else:
            ax[i,j].scatter(real_q[j::5],real_q[i::5])
sys.corner_plot(a, figure = fig)




H2 = PB_H(1.5, 5., 1.)
m, rj = MyNUTS(M, H2, dt = 0.0005, rng=rng).run(1, start_q)
what_u_get(m, t, signal, noise, Nsources)


m, rj = NUTS(M, H1, dt = 0.005, rng=rng).run(15000, start_q)
what_u_get(m, t, signal, noise, Nsources)


m, rj = MALA(M, dt = 0.00001, rng=rng).run(15000, start_q)
what_u_get(m, t, signal, noise, Nsources)


'''

