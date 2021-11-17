# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
from scipy.sparse import coo_matrix
import pickle
import linecache
import sys


def _load_network(filename, mtrx='adj'):
    print("### Loading [%s]..." % (filename))
    if mtrx == 'adj':
        i, j, val = np.loadtxt(filename).T
        #deepDTnet_data drug 732,protein 1915; DTINet_data drug 708,protein 1512
        A = coo_matrix((val, (i,j)),shape=(732,732))
        #A = coo_matrix((val, (i,j)),shape=(1915,1915))
        A = A.todense()
        A = np.squeeze(np.asarray(A))
        if A.min() < 0:
            print("### Negative entries in the matrix are not allowed!")
            A[A < 0] = 0
            print("### Matrix converted to nonnegative matrix.")
            print
        if (A.T == A).all():
            pass
        else:
            print("### Matrix not symmetric!")
            A = A + A.T
            print("### Matrix converted to symmetric.")
    else:
        print("### Wrong mtrx type. Possible: {'adj', 'inc'}")
    A = A - np.diag(np.diag(A))
    A = A + np.diag(A.sum(axis=1) == 0)

    return A


def load_networks(filenames, mtrx='adj'):
    Nets = []
    for filename in filenames:
        Nets.append(_load_network(filename, mtrx))

    return Nets


def _net_normalize(X):
    """
    Normalizing networks according to node degrees.
    """
    if X.min() < 0:
        print("### Negative entries in the matrix are not allowed!")
        X[X < 0] = 0
        print("### Matrix converted to nonnegative matrix.")
        print
    if (X.T == X).all():
        pass
    else:
        print("### Matrix not symmetric.")
        X = X + X.T - np.diag(np.diag(X))
        print("### Matrix converted to symmetric.")

    # normalizing the matrix
    deg = X.sum(axis=1).flatten()
    deg = np.divide(1., np.sqrt(deg))
    deg[np.isinf(deg)] = 0
    D = np.diag(deg)
    X = D.dot(X.dot(D))

    return X


def net_normalize(Net):
    """
    Normalize Nets or list of Nets.
    """
    if isinstance(Net, list):
        for i in range(len(Net)):
            Net[i] = _net_normalize(Net[i])
    else:
        Net = _net_normalize(Net)

    return Net


def _scaleSimMat(A):
    """Scale rows of similarity matrix"""
    A = A - np.diag(np.diag(A))
    A = A + np.diag(A.sum(axis=0) == 0)
    col = A.sum(axis=0)
    A = A.astype(np.float64)/col[:, None]

    return A


def RWR(A, K=3, alpha=0.98):
    """Random Walk on graph"""
    A = _scaleSimMat(A)
    # Random surfing
    n = A.shape[0]
    P0 = np.eye(n, dtype=float)
    P = P0.copy()
    M = np.zeros((n, n), dtype=float)
    for i in range(0, K):
        P = alpha*np.dot(P, A) + (1. - alpha)*P0
        M = M + P

    return M


def PPMI_matrix(M):
    """ Compute Positive Pointwise Mutual Information Matrix"""
    M = _scaleSimMat(M)
    n = M.shape[0]
    col = np.asarray(M.sum(axis=0), dtype=float)
    col = col.reshape((1, n))
    row = np.asarray(M.sum(axis=1), dtype=float)
    row = row.reshape((n, 1))
    D = np.sum(col)

    np.seterr(all='ignore')
    PPMI = np.log(np.divide(D*M, np.dot(row, col)))
    PPMI[np.isnan(PPMI)] = 0
    PPMI[PPMI < 0] = 0

    return PPMI


if __name__ == "__main__":

    #DTINet_data
    #path_to_string_nets = './DTINet_data/drug/'
    #path_to_string_nets = './DTINet_data/protein/'
    #string_nets = ['Drugs','drug_se','drug_drug','drug_disease']
    #string_nets = ['Proteins','protein_protein','protein_disease']

    #deepDTnet_data
    path_to_string_nets = './deepDTnet_data/drug/'
    #path_to_string_nets = './deepDTnet_data/protein/'
    string_nets = ['drugdrug','Sim_drugDisease','Sim_drugsideEffect','drugsim1network','drugsim2network','drugsim3network','drugsim4network','drugsim5network','drugsim6network']
    #string_nets = ['proteinprotein','Sim_proteinDisease','proteinsim1network','proteinsim2network','proteinsim3network','proteinsim4network']

    filenames = []
    for net in string_nets:
        filenames.append(path_to_string_nets + net +'.txt')

    # Load STRING networks
    Nets = load_networks(filenames)
    nodesCount = Nets[0].shape[0]
    # Compute RWR + PPMI
    for i in range(0, len(Nets)):
        print("### Computing PPMI for network: %s" % (string_nets[i]))
        Nets[i] = RWR(Nets[i])
        Nets[i] = PPMI_matrix(Nets[i])
        #Nets[i]=net_normalize(Nets[i])

        file = open("./deepDTnet_data/drug/"+string_nets[i]+"_G.txt", "w")
        #file = open("./deepDTnet_data/protein/"+string_nets[i]+"_G.txt", "w")

        #file = open("./DTINet_data/drug/"+string_nets[i]+"_G.txt", "w")
        #file = open("./DTINet_data/protein/"+string_nets[i]+"_G.txt", "w")

        for p in range(0, nodesCount-1):
            for q in range(p+1, nodesCount):
                value=Nets[i][p,q]
                if value!=0:
                    file.write(str(p)+' '+str(q)+' '+str(value)+'\n')
        file.close()
        print(string_nets[i]+' saved')
