import math
import numpy as np 
from matplotlib import pyplot as plt

kB = 1.380 * 10**-23
T = 300
nu = 1*10**-3
R = 1 * 10**-6
T_tot = 10000 
dt = 2 * 10**-2
x0 = 0
y0 = 0 
phi0 = 0
v = 50
np.random.seed(14)
KbT = kB * T
D = KbT/(6*math.pi*nu*R)
Dr = KbT/(8*math.pi*nu*R**3)

w_list = [0, (1/2) * math.pi, math.pi, (3/2)*math.pi] #Angular spins
#Generate x(t), t(t), phi(t) for each, use same noise for different 2
tau_R = 1 / Dr #Orientation relaxation time

def evolution_ABP(x0, y0, phi0, v, D, DR, dt, duration, w):
    """
    Function to generate the trajectory of an active Brownian particle.
    
    Parameters
    ==========
    x0, y0 : Initial position [m].
    phi0 : Initial orientation [rad].
    v : self-propulsion speed [m/s]
    D : Diffusion coefficient [m^2/s]
    DR : Rotational diffusion coefficient [1/s]
    gamma : Friction coefficient [N*s/m].    
    dt : Time step for the numerical solution [s].
    duration : Total time for which the solution is computed [s].
    """
        
    # Coefficients for the finite difference solution.
    c_noise_x = np.sqrt(2 * D * dt)
    c_noise_y = np.sqrt(2 * D * dt)
    c_noise_phi = np.sqrt(2 * DR * dt)

    N = math.ceil(duration / dt)  # Number of time steps.

    x = np.zeros(N)
    y = np.zeros(N)
    phi = np.zeros(N)

    rn = np.random.normal(0, 1, size=(3, N - 1))
    
    x[0] = x0
    y[0] = y0
    phi[0] = phi0

    for i in range(N - 1):
        x[i + 1] = x[i] + v * dt * np.cos(phi[i]) + c_noise_x * rn[0, i]
        y[i + 1] = y[i] + v * dt * np.sin(phi[i]) + c_noise_y * rn[1, i]
        phi[i + 1] = phi[i] + w* dt + c_noise_phi * rn[2, i]

    return x, y, phi


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
plt.plot(x[0][0:int(np.floor(2*tau_R/dt))], y[0][0:int(np.floor(tau_R/dt))], '-', linewidth=1, color='r', label = "w=0")
plt.plot(x[1][0:int(np.floor(2*tau_R/dt))], y[1][0:int(np.floor(tau_R/dt))], '-', linewidth=1, color='b', label = "w=(1/2)*pi")
plt.plot(x[2][0:int(np.floor(2*tau_R/dt))], y[2][0:int(np.floor(tau_R/dt))], '-', linewidth=1, color='g', label = "w=pi")
plt.plot(x[3][0:int(np.floor(2*tau_R/dt))], y[3][0:int(np.floor(tau_R/dt))], '-', linewidth=1, color='y', label = "w=(3/2)*pi")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Trajectories for different values of w")
plt.show()

time = np.arange(0, T_tot, dt)
plt.figure(figsize=(10,10))
plt.plot(time[0:int(np.floor(2*tau_R/dt))], phi[0][0:int(np.floor(tau_R/dt))], '-', linewidth=1, color='r', label = "w=0")
plt.plot(time[0:int(np.floor(2*tau_R/dt))], phi[1][0:int(np.floor(tau_R/dt))], '-', linewidth=1, color='b', label = "w=(1/2)*pi")
plt.plot(time[0:int(np.floor(2*tau_R/dt))], phi[2][0:int(np.floor(tau_R/dt))], '-', linewidth=1, color='g', label = "w=pi")
plt.plot(time[0:int(np.floor(2*tau_R/dt))], phi[3][0:int(np.floor(tau_R/dt))], '-', linewidth=1, color='y', label = "w=(3/2)*pi")
plt.xlabel("time")
plt.ylabel("phi")
plt.legend()
plt.title("phi over time for different values of w")
plt.show()


print(f"tau_R is {tau_R} nanoseconds, whcih is {tau_R/(1*10^9)} seconds") 