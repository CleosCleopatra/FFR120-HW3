import numpy as np 
    
def diffuse_spread_recover(x, y, status, d, beta, gamma, L):
    """
    Function performing the diffusion step, the infection step, and the 
    recovery step happening in one turn for a population of agents.
    
    Parameters
    ==========
    x, y : Agents' positions.
    status : Agents' status.
    d : Diffusion probability.
    beta : Infection probability.
    gamma : Recovery probability.
    L : Side of the square lattice.
    """
    
    N = np.size(x)
    
    # Diffusion step.
    diffuse = np.random.rand(N)
    move = np.random.randint(4, size=N)
    for i in range(N):
        if diffuse[i] < d:
            if move[i] == 0:
                x[i] = x[i] - 1
            elif move[i] == 1:
                y[i] = y[i] - 1
            elif move[i] == 2:
                x[i] = x[i] + 1
            else: 
                # move[i] == 3
                y[i] = y[i] + 1
                
    # Enforce pbc.
    x = x % L
    y = y % L

    # Spreading disease step.
    infected = np.where(status == 1)[0]
    
    for i in infected:
        # Check whether other particles share the same position.
        same_x = np.where(x == x[i])
        same_y = np.where(y == y[i])
        same_cell = np.intersect1d(same_x, same_y)
        for j in same_cell:
            if status[j] == 0:
                if np.random.rand() < beta:
                    status[j] = 1
        
    # Recover step.
    for i in infected:
        # Check whether the infected recovers.
        if np.random.rand() < gamma:
            status[i] = 2
    
    return x, y, status


N_part = 1000  # Total agent population.
d = 0.8  # Diffusion probability.
beta = 0.6  # Infection spreading probability.
gamma = 0.01  # Recovery probability.
L = 100  # Side of the lattice.

I0 = 10  # Initial number of infected agents.

# Initialize agents position.
x = np.random.randint(L, size=N_part)
y = np.random.randint(L, size=N_part)

# Initialize agents status.
status = np.zeros(N_part)
status[0:I0] = 1


N_part = 1000  # Total agent population.
d = 0.8  # Diffusion probability.


betas = [0.8, 0.1]  # Infection spreading probability.
gamma = [0.02, 0.01]  # Recovery probability.
L = 100  # Side of the lattice.

I0 = 10  # Initial number of infected agents.

# Initialize agents position.
x = np.random.randint(L, size=N_part)
y = np.random.randint(L, size=N_part)

# Initialize agents status.
status = np.zeros(N_part)
status[0:I0] = 1

step = 0

S = [[],[]]  # Keeps track of the susceptible agents.
I = [[],[]]  # Keeps track of the infectious agents.
R = [[],[]]  # Keeps track of the recovered agents.
S[0].append(N_part - I0)
S[1].append(N_part - I0)
I[0].append(I0)
I[1].append(I0)
R[0].append(0)
R[0].append(1)

running = True  # Flag to control the loop.
while running:
    for i in range(len(betas)):
        x, y, status = diffuse_spread_recover(x, y, status, d, beta, gamma, L)  
    
        S[i].append(np.size(np.where(status == 0)[0]))
        I[i].append(np.size(np.where(status == 1)[0]))
        R[i].append(np.size(np.where(status == 2)[0]))
    
    step += 1
    if I[-1] == 0: 
        running = False
        
print('Done.')



t = np.array(np.arange(len(S)))
S1_agents = np.array(S[0]) 
I1_agents = np.array(I[0]) 
R1_agents = np.array(R[0]) 

S2_agents = np.array(S[1]) 
I2_agents = np.array(I[1]) 
R2_agents = np.array(R[1]) 

import matplotlib.pyplot as plt

plt.plot(t, S1_agents, '-', label='S1')
plt.plot(t, I1_agents, '-', label='I1')
plt.plot(t, R1_agents, '-', label='R1')

plt.plot(t, S2_agents, '-', label='S2')
plt.plot(t, I2_agents, '-', label='I2')
plt.plot(t, R2_agents, '-', label='R2')
plt.legend()
plt.title('Course of the disease')
plt.xlabel('step')
plt.ylabel('S, I, R ')
plt.show()