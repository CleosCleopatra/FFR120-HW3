import numpy as np 
import matplotlib.pyplot as plt
from functools import reduce

np.random.seed(14)



def replicas(x, y, L):   
    xr = np.zeros(9)
    yr = np.zeros(9)

    for i in range(3):
        for j in range(3):
            xr[3 * i + j] = x + (j - 1) * L
            yr[3 * i + j] = y + (i - 1) * L
    
    return xr, yr


def pbc(x, y, L): 
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


T_tot = 1800 # change to 1800 ? 
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
# Initialize.
for i in range(n_delta_neg):
    I_fit[i] += I_ref  
#ad
neg_idx = 0 #Neg robot
traj_neg_x = np.zeros(n_steps + 1)
traj_neg_y = np.zeros(n_steps + 1)
traj_neg_x[0] = x[neg_idx]
traj_neg_y[0] = y[neg_idx] #Initial positions 

big_x = np.zeros((n_steps, N_part))
big_y = np.zeros((n_steps, N_part))
big_x[0,:] = x
big_y[0, :] = y

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
    
    big_x[step,:] = x
    big_y[step, :] = y
    
    
print(f"x is {x}")
    

traj_neg_x_plot = traj_neg_x.copy()
traj_neg_y_plot = traj_neg_y.copy()

# Detect large jumps (crossing boundary)
jump_idx = np.where(np.abs(np.diff(traj_neg_x)) > L/2)[0]
mask = np.zeros_like(traj_neg_x_plot, dtype=bool)
mask[jump_idx + 1] = True

traj_neg_x_masked = np.ma.masked_where(mask, traj_neg_x_plot)
traj_neg_y_masked = np.ma.masked_where(mask, traj_neg_y_plot)


plt.figure(figsize=(6,6))
plt.scatter(x_init, y_init, c='C0', label = 'initial')
plt.scatter(x,y, c='C1', label = 'final')

plt.plot(traj_neg_x_masked, traj_neg_y_masked, label= 'neg-delay trajectory')
plt.scatter(traj_neg_x[0], traj_neg_y[0], marker='x', c='green', label='neg start')
plt.scatter(traj_neg_x[-1], traj_neg_y[-1], marker='x', c = 'red', label = 'neg end')

plt.xlim(-L/2, L/2)
plt.ylim(-L/2, L/2)
plt.legend()
plt.title('Initial in green and final in red')
plt.show()










segments_x = []
segments_y = []
current_x = [x[0]]
current_y = [y[0]]

for i in range(1, len(x)):
    dx = traj_neg_x_masked[i] - traj_neg_x_masked[i-1]
    dy = traj_neg_y_masked[i] - traj_neg_y_masked[i-1]
    if abs(dx) > L/2 or abs(dy) > L/2:
        # finish current segment
        segments_x.append(current_x)
        segments_y.append(current_y)
        # start new one
        current_x = [x[i]]
        current_y = [y[i]]
    else:
        current_x.append(x[i])
        current_y.append(y[i])

# add last segment
segments_x.append(current_x)
segments_y.append(current_y)

for sx, sy in zip(segments_x, segments_y):
    plt.plot(sx, sy, '-', linewidth=1)





def convert2RBG(I_profile, RGB0, RGB1):
    """
    Function to convert the 2 dimensional numpy array into a RGB image.
    
    Parameters
    ==========
    I_profile : Intensity profile.
    RGB0 : Components R, G, B of the chosen color shade for minimum I_profile.
    RGB1 : Components R, G, B of the chosen color shade for maximum I_profile.
    """
    
    [n_rows, n_cols] = I_profile.shape
    
    I_RGB = np.zeros([n_rows, n_cols, 3])
    
    # Set I_profile between 0 and 1
    I_profile -= np.amin(I_profile)    
    I_profile /= np.amax(I_profile)  
    
    for c in range(3):
        I_RGB[:, :, c] = I_profile * RGB1[c] + (1 - I_profile) * RGB0[c]

    return I_RGB

# Define grid over the square arena
dx = 0.05   # pixel size in meters
dy = 0.05
x_lin = np.arange(-L/2, L/2+dx, dx)
y_lin = np.arange(-L/2, L/2+dy, dy)
x_coo, y_coo = np.meshgrid(x_lin, y_lin)
Lx, Ly = x_coo.shape

# Initialize exploration map
exploration = np.zeros((Lx, Ly))

# Accumulate visits: loop over all robots and all timesteps
for t in range(n_steps):
    for n in range(N_part):
        # Convert robot position to pixel index
        ix = int((big_x[t,n] +L/2) / dx)
        iy = int((big_y[t,n] +L/2)/ dy)
        if 0 <= ix < Lx and 0 <= iy < Ly:
            exploration[iy, ix] += 1

# Convert to RGB image using your function
RGB0 = [1.0, 1.0, 1.0]   # white for min
RGB1 = [0.3, 0.3, 1.0]   # bluish for max
exploration_img = convert2RBG(exploration, RGB0, RGB1)

from matplotlib.colors import LogNorm
# Plot exploration profile
plt.figure(figsize=(6,6))
plt.imshow(exploration, origin='lower', extent=[-L/2,L/2,-L/2,L/2], norm=LogNorm())
plt.title("Exploration Histogram (All Robots)")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.axis('equal')
plt.colorbar(label="Visits")
plt.show()