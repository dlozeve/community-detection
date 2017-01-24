import numpy as np
import scipy.sparse.linalg as linalg
import sklearn.cluster as cluster
import networkx as nx
import itertools



## Random graph models

## Stochastic Block Model

def findsubset(e, partition):
    """Finds the subset which contains e
    """
    for i in range(len(partition)):
        if e in partition[i]:
            return i
    return -1

def stochasticblockmodel(n, partition, edge_prob):
    """Returns a stochastic block model graph

    n is the number of vertices
    partition is a partition of {1, ..., n} into disjoint subsets {C_1, ..., C_r}
    edge_prob is a symmetric r*r matrix of edge probabilities
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i):
            r_i = findsubset(i, partition)
            r_j = findsubset(j, partition)
            u = np.random.random() # random number in [0,1]
            if u < edge_prob[r_i][r_j]:
                G.add_edge(i, j)
    return G

## In/Out Block Model

def regularPartition(r, n, verbose = False):
    """Returns the regular partition that will be used"""
    partition = np.arange(r*n).reshape((r,n))
    if verbose:
        print("Regular partition:")
        print(partition)
    return partition

def inOutEdgeMatrix(r, n, cIn, cOut, verbose = False):
    """Creates the edge matrix required to generate the stochastic block model"""
    #print("Edge matrix creation starting !")
    matrix = np.zeros((r,r))
    for i in range(r):
        for j in range(r):
            if i==j:
                matrix[i,j] = float(cIn)/(r*n)
            else:
                matrix[i,j] = float(cOut)/(r*n)
    if verbose:
        print("Edge matrix:")
        print(matrix)
    #print("Edge matrix created !")
    return matrix

def inOutBlockModel(r, n, d, epsilon, verbose = False):
    Matrix = inOutEdgeMatrix(r, n, float(d)/(1+epsilon), float(d*epsilon)/(1+epsilon), verbose)
    return stochasticblockmodel(r*n, regularPartition(r, n, verbose), Matrix)



## Spectral clustering algorithm

def spectralclustering(g, k, l = None):
    """Computes the spectral clustering of graph g in k clusters, 
    using the l first eigenvectors of the unnormalized Laplacian matrix.
    """
    # We compute the Laplacian matrix of g. It is necessary to convert
    # it to floating-point type.
    L = nx.laplacian_matrix(g).asfptype()
    # We compute the first k eigenvectors (the k smallest in magnitude)
    if l is None:
        l = k
    eigs, u = linalg.eigsh(L, l, which='SM')
    kmeans = cluster.KMeans(n_clusters=k).fit(u)
    return kmeans.labels_



## Performance measure

def overlap(r, n, node2cluster, cluster2node):
    maximum = 0
    for pi in itertools.permutations(range(len(cluster2node)), r):
        total = 0
        for i in range(r*n):
            if node2cluster[i] == pi[int(i/n)]:
                total += 1
        result = (float(total)/(r*n) - float(1)/r)/(1 - float(1)/r)
        if result > maximum:
            maximum = result
    return maximum

def performance(graph, labels):
    # number of vertices
    n = len(graph)
    # list of edges
    edges = graph.edges()
    count = 0
    for i in range(n):
        for j in range(n):
            if (i, j) in edges and labels[i] == labels[j]:
                count += 1
            elif (i, j) not in edges and labels[i] != labels[j]:
                count +=1
    return count / (n*(n-1))

def modularity(graph, edgeprob, labels):
    n = len(graph)
    m = len(graph.edges())
    A = nx.adjacency_matrix(graph).todense()
    res = 0
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                res += A[i,j] - edgeprob[i][j]
    return res / (2*m)



