import math
import numpy as np 

def evolution_GI(x0, y0, phi0, v_inf, v0, Ic, I0, r0, tau, dt, duration):
    """
    Function to generate the trajectory of a light-sensitive robot in a Gaussian
    light intensity zone.
    
    Parameters
    ==========
    x0, y0 : Initial position [m].
    phi0 : Initial orientation [rad].
    v_inf : Self-propulsion speed at I=0 [m/s]
    v0 : Self-propulsion speed at I=I0 [m/s]
    Ic : Intensity scale over which the speed decays.
    I0 : Maximum intensity.
    r0 : Standard deviation of the Gaussian intensity.
    tau : Time scale of the rotational diffusion coefficient [s]
    dt : Time step for the numerical solution [s].
    duration : Total time for which the solution is computed [s].
    """
        
    # Coefficients for the finite difference solution.
    c_noise_phi = np.sqrt(2 / tau * dt)

    N = math.ceil(duration / dt)  # Number of time steps.

    x = np.zeros(N)
    y = np.zeros(N)
    phi = np.zeros(N)

    rn = np.random.normal(0, 1, N - 1)
    
    x[0] = x0
    y[0] = y0
    phi[0] = phi0

    for i in range(N - 1):
        I =  I0 * np.exp(- (x[i] ** 2 + y[i] ** 2) / r0 ** 2)
        v = v_inf + (v0 - v_inf) * np.exp(- I / Ic) 
        x[i + 1] = x[i] + v * dt * np.cos(phi[i])
        y[i + 1] = y[i] + v * dt * np.sin(phi[i])
        phi[i + 1] = phi[i] + c_noise_phi * rn[i]

    return x, y, phi


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
            np.where(x - r_c > - L / 2)[0]
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


T_tot = 1800.0
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

delta = 0  # No delay. Tends to cluster.
delta_pos = 5 * dt  # Positive delay. More stable clustering.
delta_neg = - 5 * dt  # Negative delay. Dispersal.
#ad
delta_arr = np.full(N_part, delta_pos)



# Initialization.

# Random position.
x = (np.random.rand(N_part) - 0.5) * L  # in [-L/2, L/2]
y = (np.random.rand(N_part) - 0.5) * L  # in [-L/2, L/2]

# Random orientation.
phi = 2 * (np.random.rand(N_part) - 0.5) * np.pi  # in [-pi, pi]

# Coefficients for the finite difference solution.
c_noise_phi = np.sqrt(2 * dt / tau)

I_ref = calculate_intensity(x, y, I0, r0, L, r_c)

if delta < 0:
    # Negative delay.
    n_fit = int(- delta / dt)  # Delay in units of time steps.
    I_fit = np.zeros([n_fit, N_part])
    t_fit = np.arange(n_fit) * dt
    dI_dt = np.zeros(N_part)
    # Initialize.
    for i in range(n_fit):
        I_fit[i, :] += I_ref   
        
if delta > 0:
    # Positive delay.
    n_delay = int(delta / dt)  # Delay in units of time steps.
    I_memory = np.zeros([n_delay, N_part])
    # Initialize.
    for i in range(n_delay):
        I_memory[i, :] += I_ref   
    


rp = r0 / 3
vp = rp  # Length of the arrow indicating the velocity direction.
line_width = 1  # Width of the arrow line.



    


step = 0


for step in range(T_tot):
    
    # Calculate current I.
    I_particles = calculate_intensity(x, y, I0, r0, L, r_c)
    
    if delta < 0:
        # Estimate the derivative of I linear using the last n_fit values.
        # Update I_fit.
        I_fit = np.roll(I_fit, -1, axis=0)
        I_fit[-1, :] = I_particles
        # Fit to determine the slope.
        for j in range(N_part):
            p = np.polyfit(t_fit, I_fit[:, j], 1)
            dI_dt[j] = p[0]
        # Determine forecast. Remember that here delta is negative.
        I = I_particles - delta * dI_dt  
        I[np.where(I < 0)[0]] = 0
    elif delta > 0:
        # Update I_memory.
        I_memory = np.roll(I_memory, -1, axis=0)
        I_memory[-1, :] = I_particles    
        I = I_memory[0, :]
    else:
        I = I_particles
       
    # Calculate new positions and orientations. 
    v = v_inf + (v0 - v_inf) * np.exp(- I / Ic) 
    nx = x + v * dt * np.cos(phi)
    ny = y + v * dt * np.sin(phi)
    nphi = phi + c_noise_phi * np.random.normal(0, 1, N_part)


    # Apply pbc.
    nx, ny = pbc(nx, ny, L)
                
                    
    step += 1
    x[:] = nx[:]
    y[:] = ny[:]
    phi[:] = nphi[:]  