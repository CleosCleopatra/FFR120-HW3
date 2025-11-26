import matplotlib.pyplot as plt
import numpy as np
np.random.seed(14)


def watts_strogatz_sw(n, c, p, P, rng):
    """
    Function generating a Watts-Strogatz small-world model
    
    Parameters
    ==========
    n : Number of nodes.
    c : Number of connected nearest neigbours. Must be even.
    p : Probability that each existing edge is randomly rewired.
    """
    
    A = np.zeros([n, n])        
    A_rewired = np.zeros([n, n])    
    
    c_half = int(c / 2)
    
    for i in range(n):
        for j in range(i + 1, i + 1 + c_half):
            A[i, j % n] = 1
            A[j % n, i] = 1

            
    for i in range(n):
        for j in range(i + 1, i + 1 + c_half):
            if P[i, j % n] < p:
                # Select a random node.
                k = (rng.integers(n-1)+1+i) % n
                A_rewired[i, k % n] = 1
                A_rewired[k % n, i] = 1
            else:
                A_rewired[i, j % n] = 1
                A_rewired[j % n, i] = 1

    # This below is for plotting in a circular arrangement.
    x = np.cos(np.arange(n) / n * 2 * np.pi)
    y = np.sin(np.arange(n) / n * 2 * np.pi) 
    
    return A_rewired, x, y


from matplotlib import pyplot as plt

n = 100
c = 6  # c must be even.
ps = [10**-4, 10**-3, 10**-2, 10**-1, 1]
num_runs = 3
num_runs_total = num_runs * len(ps)

P = np.random.rand(num_runs_total, n, n)
rngs = [np.random.default_rng(1000+r) for r in range(num_runs_total)]

rand_indx = 0
colours = ["b", "y"]
"""
for s, p in enumerate(ps):
    plt.figure(figsize=(8, 8))
    for idx in range(3):
        A_WS, x_WS, y_WS = watts_strogatz_sw(n, c, p, P[rand_indx], rngs[rand_indx])
        rand_indx+=1
        
        for i in range(n):
            for j in range(i + 1, n):
                if A_WS[i, j] > 0:
                    plt.plot([x_WS[i], x_WS[j]], [y_WS[i], y_WS[j]], '-', 
                            color=colours[idx])
        plt.plot(x_WS, y_WS, '.', markersize=12, color= colours[idx], label=f'j = {idx}')
      
    plt.legend()
    plt.title(f'Watts-Strogatz small-world for p= {p}')
    plt.axis('equal')
    plt.show()

"""
def path_length(A, i, j):
    """
    Function returning the minimum path length between thwo nodes.
    
    Parameters
    ==========
    A : Adjacency matrix (assumed symmetric).
    i, j : Nodes indices.
    """
    
    Lij = - 1
    
    if A[i, j] > 0:
        Lij = 1
    else:
        N = np.size(A[0, :])
        P = np.zeros([N, N]) + A
        n = 1
        running = True
        while running:
            P = np.matmul(P, A)
            n += 1
            running
            if P[i, j] > 0:
                Lij = n           
            if (n > N) or (Lij > 0):
                running = False   
    
    return Lij

    
def matrix_path_length(A):
    """
    Function returning a matrix L of minimum path length between nodes.
    
    Parameters
    ==========
    A : Adjacency matrix (assumed symmetric).
    """
    
    N = np.size(A[0, :])
    L = np.zeros([N, N]) - 1 
    
    for i in range(N):
        for j in range(i + 1, N):
            L[i, j] = path_length(A, i, j)
            L[j, i] = L[i, j]
    
    return L


import numpy as np 
    
def nodes_degree(A):
    """
    Function returning the degree of a node.
    
    Parameters
    ==========
    A : Adjacency matrix (assumed symmetric).
    """
    
    degree = np.sum(A, axis=0)
    
    return degree


def clustering_coefficient(A):
    """
    Function returning the clustering coefficient of a graph.
    
    Parameters
    ==========
    A : Adjacency matrix (assumed symmetric).
    """
            
    K = nodes_degree(A)
    N = np.size(K)

    C_n = np.sum(np.diagonal(np.linalg.matrix_power(A, 3)))
    C_d = np.sum(K * (K - 1))
    
    C = C_n / C_d
    
    return C



L_av = np.zeros((np.size(ps), num_runs))
D = np.zeros((np.size(ps), num_runs))
rand_indx = 0

for s, p in enumerate(ps):
    for idx in range(num_runs):
        A_WS, x_WS, y_WS = watts_strogatz_sw(n, c, p, P[rand_indx], rngs[rand_indx])
        rand_indx +=1
        L = matrix_path_length(A_WS)
        upper = L[np.triu_indices(n, k=1)]
        valid = upper[upper >= 0]
        for j in range(n):
            L_av[s][idx] += np.sum(L[j, j + 1:n])
        L_av[s][idx] /= (n * (n - 1) / 2)

import statistics
means = []
errors =[]
for s in range(len(ps)):
    means.append(np.mean(L_av[s]))

    errors.append(statistics.stdev(L_av[s]))

plt.figure()
x = np.arange(len(ps))
labels = [f"{p:g}" for p in ps]
plt.bar(x, means, yerr= errors)
plt.xticks(x, labels)
plt.xlabel('Rewiring probability p')
plt.ylabel('Average shortest path length')
plt.title('Watts–Strogatz: average shortest path vs p')
plt.grid(True, axis='y', ls='--', alpha=0.4)
plt.tight_layout()
plt.show()

