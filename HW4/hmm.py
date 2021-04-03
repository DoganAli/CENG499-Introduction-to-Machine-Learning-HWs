import numpy as np

def forward(A, B, pi, O):
    """
    Calculates the probability of an observation sequence O given the model(A, B, pi).
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities (N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The probability of the observation sequence and the calculated alphas in the Trellis diagram with shape
             (N, T) which should be a numpy array.
    """
    N = len(pi)
    T = len(O)
    
    alpha = np.zeros((N,T))
    
    for j in range(N):
        alpha[j][0] = pi[j] * B[j][O[0]] #first step, at state = O[0] 
    
    for t in range(1, T): # t = 1 den başlayıp,deavm T iterate
        for j in range(N): # N iterate , N*T up to here 
            alpha[j][t] = B[j][O[t]] * sum(A[i][j] * alpha[i][t-1] for i in range(N)) # N iterate, O(N^2 x T) in total

    prob = sum(alpha[i][T - 1] for i in range(N))
    return  prob,alpha

def viterbi(A, B, pi, O):
    """
    Calculates the most likely state sequence given model(A, B, pi) and observation sequence.
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities(N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The most likely state sequence with shape (T,) and the calculated deltas in the Trellis diagram with shape
             (N, T). They should be numpy arrays.
    """
    N = len(pi)
    T = len(O)
    
    delta = np.zeros((N,T))
   
    path = {} # keep track of the winning sequences, dictionary with N keys, each T length of states

    for i in range(N):
        delta[i][0] = pi[i] * B[i][O[0]] # ilk observation initialization
        path[i] = [i] 

    for t in range(1, T):
        new_path = {}
        for j in range(N):
            max_prob = A[0][j]*delta[0][t-1] 
            state = 0 # the most probable t-1 state which will come to j at t
            for i in range(N):
                prob = A[i][j] * delta[i][t - 1]
                if(prob > max_prob):
                    max_prob = prob
                    state = i
                    
            delta[j][t] = B[j][O[t]] * max_prob
            new_path[j] = path[state] + [j]
            
        path.update( new_path)


    state = 0 # the most probable state at T 
    max_prob = delta[0][T-1]
    for i in range(N):
        prob = delta[i][T-1]
        if(prob > max_prob):
            state = i 
            
    return path[state],delta,path