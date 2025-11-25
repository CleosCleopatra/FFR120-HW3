import math
import numpy as np 
import matplotlib.pyplot as plt
from functools import reduce

np.random.seed(14)



def replicas(x, y, L):
    """
    Function to generate replicas of a single particle.
    
    Parameters
    ==========
    x, y : Position.
    L : Side of the squared arena.
    """    
    xr = np.zeros(9)
    yr = np.zeros(9)

    for i in range(3):
        for j in range(3):
            xr[3 * i + j] = x + (j - 1) * L
            yr[3 * i + j] = y + (i - 1) * L
    
    return xr, yr


def pbc(x, y, L):
    """
    Function to enforce periodic boundary conditions on the positions.
    
    Parameters
    ==========
    x, y : Position.
    L : Side of the squared arena.
    """   
    
    outside_left = np.where(x < - L / 2)[0]
    x[outside_left] = x[outside_left] + L

    outside_right = np.where(x > L / 2)[0]
    x[outside_right] = x[outside_right] - L

    outside_up = np.where(y > L / 2)[0]
    y[outside_up] = y[outside_up] - L

    outside_down = np.where(y < - L / 2)[0]
    y[outside_down] = y[outside_down] + L
    
    return x, y


from functools import reduce

def calculate_intensity(x, y, I0, r0, L, r_c):
    """
    Function to calculate the intensity seen by each particle.
    
    Parameters
    ==========
    x, y : Positions.
    r0 : Standard deviation of the Gaussian light intensity zone.
    I0 : Maximum intensity of the Gaussian.
    L : Dimension of the squared arena.
    r_c : Cut-off radius. Pre-set it around 3 * r0. 
    """
    
    N = np.size(x)

    I_particle = np.zeros(N)  # Intensity seen by each particle.
    
    # Preselect what particles are closer than r_c to the boundaries.
    replicas_needed = reduce( 
        np.union1d, (
            np.where(y + r_c > L / 2)[0], 
            np.where(y - r_c < - L / 2)[0],
            np.where(x + r_c > L / 2)[0],
            np.where(x - r_c < - L / 2)[0] #?
        )
    )

    for j in range(N - 1):   
        
        # Check if replicas are needed to find the interacting neighbours.
        if np.size(np.where(replicas_needed == j)[0]):
            # Use replicas.
            xr, yr = replicas(x[j], y[j], L)
            for nr in range(9):
                dist2 = (x[j + 1:] - xr[nr]) ** 2 + (y[j + 1:] - yr[nr]) ** 2 
                nn = np.where(dist2 <= r_c ** 2)[0] + j + 1
                
                # The list of nearest neighbours is set.
                # Contains only the particles with index > j
        
                if np.size(nn) > 0:
                    nn = nn.astype(int)
        
                    # Find total intensity
                    dx = x[nn] - xr[nr]
                    dy = y[nn] - yr[nr]
                    d2 = dx ** 2 + dy ** 2
                    I = I0 * np.exp(- d2 / r0 ** 2)
                    
                    # Contribution for particle j.
                    I_particle[j] += np.sum(I)

                    # Contribution for nn of particle j nr replica.
                    I_particle[nn] += I
                
        else:
            dist2 = (x[j + 1:] - x[j]) ** 2 + (y[j + 1:] - y[j]) ** 2 
            nn = np.where(dist2 <= r_c ** 2)[0] + j + 1
        
            # The list of nearest neighbours is set.
            # Contains only the particles with index > j
        
            if np.size(nn) > 0:
                nn = nn.astype(int)
        
                # Find interaction
                dx = x[nn] - x[j]
                dy = y[nn] - y[j]
                d2 = dx ** 2 + dy ** 2
                I = I0 * np.exp(- d2 / r0 ** 2)
                
                # Contribution for particle j.
                I_particle[j] += np.sum(I)

                # Contribution for nn of particle j.
                I_particle[nn] += I
                   
    return I_particle


T_tot = 200 #Switch to 1800
dt = 0.05
n_steps = int(T_tot / dt)


tau = 1  # Timescale of the orientation diffusion.
dt = 0.05  # Time step [s].

#Ad
c_noise_phi_prefactor = np.sqrt(2 * dt / tau)

#Re
N_part = 10  # Number of light-sensitive robots. 
# Note: 5 is enough to demonstrate clustering - dispersal. 


v0 = 0.1  # Self-propulsion speed at I=0 [m/s].
v_inf = 0.01  # Self-propulsion speed at I=+infty [m/s].
Ic = 0.1  # Intensity scale where the speed decays.
I0 = 1  # Maximum intensity.
r0 = 0.3  # Standard deviation of the Gaussian light intensity zone [m].
r_c = 4 * r0  # Cut-off radius [m].
L =3  # Side of the arena[m].

N_neg = 1

delta_pos = 5 * dt  # Positive delay. More stable clustering.
delta_neg = - 5 * dt  # Negative delay. Dispersal.
#ad
delta_arr = np.full(N_part, delta_pos)
delta_arr[:N_neg] = delta_neg


# Initialization.

# Random position.
x = (np.random.rand(N_part) - 0.5) * L  # in [-L/2, L/2]
y = (np.random.rand(N_part) - 0.5) * L  # in [-L/2, L/2]

# Random orientation.
phi = 2 * (np.random.rand(N_part) - 0.5) * np.pi  # in [-pi, pi]

#ad
x_init = x.copy()
y_init = y.copy()

# Coefficients for the finite difference solution.
#c_noise_phi = np.sqrt(2 * dt / tau)

I_ref = calculate_intensity(x, y, I0, r0, L, r_c)

#ad
n_pos = N_part- N_neg
n_neg = N_neg

#ad
#For positive robot
n_delta_pos = int(delta_pos/dt)
I_memory = np.zeros([n_delta_pos, N_part])
for i in range(n_delta_pos):
    I_memory[i, :] += I_ref 

n_delta_neg = int(-delta_neg / dt)
I_fit = np.zeros((n_delta_neg, N_part))
t_fit = np.arange(n_delta_neg) * dt
#dI_dt = np.zeros(N_part)
# Initialize.
for i in range(n_delta_neg):
    I_fit[i] += I_ref  

#rp = r0 / 3
#vp = rp  # Length of the arrow indicating the velocity direction.
#line_width = 1  # Width of the arrow line.

#ad
neg_idx = 0 #Neg robot
traj_neg_x = np.zeros(n_steps + 1)
traj_neg_y = np.zeros(n_steps + 1)
traj_neg_x[0] = x[neg_idx]
traj_neg_y[0] = y[neg_idx] #Initial positions

for step in range(n_steps):
    print(step)
    
    # Calculate current I.
    I_particles = calculate_intensity(x, y, I0, r0, L, r_c)
    
    I_memory = np.roll(I_memory, -1, axis = 0)
    I_memory[-1] = I_particles
    I_delay_pos = I_memory[0]

    I_fit = np.roll(I_fit, -1, axis = 0)
    I_fit[-1] = I_particles
    
    #ad
    I_effective = np.zeros(N_part)
    for j in range(N_part):
        d = delta_arr[j]
        if d > 0:
            
            I_effective[j] = I_delay_pos[j]
        elif d < 0:
            p = np.polyfit(t_fit, I_fit[:, j], 1)
            slope = p[0]
            I_pred = I_particles[j] - d * slope
            I_effective[j] = max(I_pred, 0.0)
        else:
            I_effective[j] = I_particles[j]

    # Calculate new positions and orientations. 
    v = v_inf + (v0 - v_inf) * np.exp(- I_effective / Ic) 
    nx = x + v * dt * np.cos(phi)
    ny = y + v * dt * np.sin(phi)
    nphi = phi + c_noise_phi_prefactor * np.random.normal(0, 1, N_part)


    # Apply pbc.
    nx, ny = pbc(nx, ny, L)
                

    x[:] = nx[:]
    y[:] = ny[:]
    phi[:] = nphi[:] 
    traj_neg_x[step + 1] = x[neg_idx]
    traj_neg_y[step + 1] = y[neg_idx]


plt.figure(figsize=(6,6))
plt.scatter(x_init, y_init, c='C0', label = 'initial')
plt.scatter(x,y, c='C1', label = 'final')

plt.plot(traj_neg_x, traj_neg_y, label= 'neg-delay trajectory')
plt.scatter(traj_neg_x[0], traj_neg_y[0], marker='x', c='green', label='neg start')
plt.scatter(traj_neg_x[-1], traj_neg_y[-1], marker='x', c = 'red', label = 'neg end')

plt.xlim(-L/2, L/2)
plt.ylim(-L/2, L/2)
plt.legend()
plt.title('Initial in green and final in red')
plt.show()