import math
import numpy as np 
from matplotlib import pyplot as plt

np.random.seed(7)

kB = 1.380 * 10**-23
T = 300
nu = 1*10**-3
R = 1 * 10**-6
T_tot = 10000
dt = 2 * 10**-2
x0 = 0
y0 = 0 
phi0 = 0
v = 50e-6
np.random.seed(14)
KbT = kB * T
D = KbT/(6*math.pi*nu*R)
Dr = KbT/(8*math.pi*nu*R**3)

w_list = [0, (1/2) * math.pi, math.pi, (3/2)*math.pi] #Angular spins
#Generate x(t), t(t), phi(t) for each, use same noise for different 2
tau_R = 1 / Dr #Orientation relaxation time
print(f"tau r is {tau_R}")

c_noise_x = np.sqrt(2 * D * dt)
c_noise_y = np.sqrt(2 * D * dt)
c_noise_phi = np.sqrt(2 * Dr * dt)

N = math.ceil(T_tot / dt)  # Number of time steps.


n_vers = len(w_list)

x = np.zeros([n_vers, N])
y = np.zeros([n_vers, N])
phi = np.zeros([n_vers, N])

rn = np.random.normal(0, 1, size=(3, N - 1))

for j in range(n_vers):
    print(f"j is {j}")
    x[j, 0] = x0
    y[j, 0] = y0
    phi[j, 0] = phi0
    for i in range(N-1):
        print(i)
        x[j, i + 1] = x[j, i] + v * dt * np.cos(phi[j, i]) + c_noise_x * rn[0, i]
        y[j, i + 1] = y[j, i] + v * dt * np.sin(phi[j, i]) + c_noise_y * rn[1, i]
        phi[j, i + 1] = phi[j, i] + w_list[j]* dt + c_noise_phi * rn[2, i]

plt.figure(figsize=(10,10))
plt.plot(x[0][0:int(np.floor(2*tau_R/dt))], y[0][0:int(np.floor(2*tau_R/dt))], '-', linewidth=1, color='r', label = "w=0")
plt.plot(x[1][0:int(np.floor(2*tau_R/dt))], y[1][0:int(np.floor(2*tau_R/dt))], '-', linewidth=1, color='b', label = "w=(1/2)*pi")
plt.plot(x[2][0:int(np.floor(2*tau_R/dt))], y[2][0:int(np.floor(2*tau_R/dt))], '-', linewidth=1, color='g', label = "w=pi")
plt.plot(x[3][0:int(np.floor(2*tau_R/dt))], y[3][0:int(np.floor(2*tau_R/dt))], '-', linewidth=1, color='y', label = "w=(3/2)*pi")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Trajectories for different values of w")
plt.show()

time = np.arange(0, T_tot, dt)
plt.figure(figsize=(10,10))
plt.plot(time[0:int(np.floor(2*tau_R/dt))], phi[0][0:int(np.floor(2*tau_R/dt))], '-', linewidth=1, color='r', label = "w=0")
plt.plot(time[0:int(np.floor(2*tau_R/dt))], phi[1][0:int(np.floor(2*tau_R/dt))], '-', linewidth=1, color='b', label = "w=(1/2)*pi")
plt.plot(time[0:int(np.floor(2*tau_R/dt))], phi[2][0:int(np.floor(2*tau_R/dt))], '-', linewidth=1, color='g', label = "w=pi")
plt.plot(time[0:int(np.floor(2*tau_R/dt))], phi[3][0:int(np.floor(2*tau_R/dt))], '-', linewidth=1, color='y', label = "w=(3/2)*pi")
plt.xlabel("time")
plt.ylabel("phi")
plt.legend()
plt.title("phi over time for different values of w")
plt.show()


print(f"tau_R is {tau_R*1e9} nanoseconds, whcih is {tau_R} seconds") 





def MSD_2d(x, y, n_delays):
    """
    Function to calculate the MSD.
    
    Parameters
    ==========
    x : Trajectory (x component).
    y : Trajectory (y component).
    n_delays : Indicates the delays to be used to calculate the MSD.
    """
    L = np.size(n_delays)
    msd = np.zeros(L)
    
    print(f"x is  {x}")
    
    nelem = np.size(x)
    
    for i in range(L):
        n = n_delays[i]
        Nmax = nelem - n
        dx = x[n:nelem] -  x[0: Nmax]
        dy = y[n:nelem] -  y[0: Nmax]
        msd[i] += np.mean(dx ** 2 + dy ** 2)
    

    return msd

n_delays = 2 ** np.array(np.arange(np.floor(np.log(N) / np.log(2))))

msd = np.zeros([n_vers, np.size(n_delays)])
t_delay = n_delays * dt
plt.figure(figsize = (10,10))

for j in range(n_vers):
    msd[j, :] = MSD_2d(x[j, :],y[j, :],n_delays.astype(int))
    plt.plot(t_delay, msd[j, :], label=f"w is {w_list[j]}")

plt.title("Msd over time for different values of w")
plt.xlabel("time")
plt.ylabel("msd")
plt.legend()
plt.show()