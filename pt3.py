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


N_part = 2000  # Total agent population.
d = 0.9  # Diffusion probability.
L = 100  # Side of the lattice.

I0 = 10  # Initial number of infected agents.

betas = [0.8, 0.1]  # Infection spreading probability.
gammas = [0.02, 0.01]  # Recovery probability.


import matplotlib.pyplot as plt

step = 0
n_reps = 6

plt.figure(figsize=(15, 6))
for rep in range(n_reps):
    print(f"rep is {rep}")
    x = np.random.randint(L, size=N_part)
    y = np.random.randint(L, size=N_part)
    status = np.zeros(N_part)
    status[0:I0] = 1
    S, I, R = [N_part-I0], [I0], [0]
    running = True  # Flag to control the loop.
    while running:
        x, y, status = diffuse_spread_recover(x, y, status, d, betas[0], gammas[0], L)  
    
        S.append(np.size(np.where(status == 0)[0]))
        I.append(np.size(np.where(status == 1)[0]))
        R.append(np.size(np.where(status == 2)[0]))
        if I[-1] == 0: 
            running = False

    t = np.array(np.arange(len(S)))
    plt.plot(t, S, '-', label=f'S for rep{rep+1}')
    plt.plot(t, I, '-', label=f"i for rep{rep + 1}")
    plt.plot(t, R, '-', label=f"R fpr rep {rep +1}")


plt.legend(loc='upper left', bbox_to_anchor=(0.95, 1), borderaxespad=0)
plt.title('Course of the disease, condition 1')
plt.xlabel('step')
plt.ylabel('S, I, R ')
plt.show()





#Second condition
plt.figure(figsize=(15, 6))
for rep in range(n_reps):
    print(f"rep is {rep}")
    x = np.random.randint(L, size=N_part)
    y = np.random.randint(L, size=N_part)
    status = np.zeros(N_part)
    status[0:I0] = 1
    S, I, R = [N_part-I0], [I0], [0]
    running = True  # Flag to control the loop.
    while running:
        x, y, status = diffuse_spread_recover(x, y, status, d, betas[1], gammas[1], L)  
    
        S.append(np.size(np.where(status == 0)[0]))
        I.append(np.size(np.where(status == 1)[0]))
        R.append(np.size(np.where(status == 2)[0]))
        if I[-1] == 0: 
            running = False

    t = np.array(np.arange(len(S)))
    plt.plot(t, S, '-', label=f'S for rep{rep+1}')
    plt.plot(t, I, '-', label=f"i for rep{rep + 1}")
    plt.plot(t, R, '-', label=f"R fpr rep {rep +1}")


plt.legend(loc='upper left', bbox_to_anchor=(0.95, 1), borderaxespad=0)
plt.title('Course of the disease, condition 2')
plt.xlabel('step')
plt.ylabel('S, I, R ')
plt.show()






#pt 2d 
def diffuse_spread_recover_split(x, y, status, d, beta, gamma, L, c1, c2):
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
    #print(f"c1 is {c1}")
    #print(f"x size is {len(x)}")
    
    # Diffusion step.
    diffuse_1 = np.random.rand(c1)
    diffuse_2 = np.random.rand(c2)
    move_1 = np.random.randint(4, size=c1)
    move_2 = np.random.randint(4, size=c2)
    for i in range(c1):
        if diffuse_1[i] < d:
            if move_1[i] == 0:
                x[i] = x[i] - 1
            elif move_1[i] == 1:
                y[i] = y[i] - 1
            elif move_1[i] == 2:
                x[i] = x[i] + 1
            else: 
                # move[i] == 3
                y[i] = y[i] + 1
    for i in range(c2):
        #print(f"c is {c1-1+i}")
        if diffuse_2[i] < d:
            if move_2[i] == 0:
                x[c1+i] = x[c1+i] - 1
            elif move_2[i] == 1:
                y[c1+i] = y[c1+i] - 1
            elif move_2[i] == 2:
                x[c1+i] = x[c1+i] + 1
            else: 
                # move[i] == 3
                y[c1+i] = y[c1+i] + 1
                
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
                ran = np.random.rand()
                #print(f"beta 0 is {beta[0]}")
                if (ran < beta[0] and j < c1) or (ran<beta[1] and j >= c1) :
                    status[j] = 1

        
    # Recover step.
    for i in infected:
        # Check whether the infected recovers.
        ran = np.random.rand()
        if (ran < gamma[0] and i < c1) or (ran< gamma[1] and i >= c1):
            status[i] = 2
    
    return x, y, status

#third condition
pop_1=1000
pop_2=N_part-pop_1
plt.figure(figsize=(15, 6))
for rep in range(n_reps):
    print(f"rep is {rep}")
    x = np.random.randint(L, size=N_part)
    y = np.random.randint(L, size=N_part)
    status = np.zeros(N_part)
    random_infected = np.random.choice(N_part, size = I0, replace=False)
    status[random_infected] = 1
    S, I, R = [N_part-I0], [I0], [0]
    running = True  # Flag to control the loop.
    while running:
        x, y, status = diffuse_spread_recover_split(x, y, status, d, betas, gammas, L, pop_1, pop_2)  
    
        S.append(np.size(np.where(status == 0)[0]))
        I.append(np.size(np.where(status == 1)[0]))
        R.append(np.size(np.where(status == 2)[0]))
        if I[-1] == 0: 
            running = False

    t = np.array(np.arange(len(S)))
    plt.plot(t, S, '-', label=f'S for rep{rep+1}')
    plt.plot(t, I, '-', label=f"i for rep{rep + 1}")
    plt.plot(t, R, '-', label=f"R fpr rep {rep +1}")


plt.legend(loc='upper left', bbox_to_anchor=(0.95, 1), borderaxespad=0)
plt.title('Course of the disease, condition 3')
plt.xlabel('step')
plt.ylabel('S, I, R ')
plt.show()