#!/usr/bin/env python

import networkx as nx


""" This script finds the largest clique in an undirected graph."""

def findMaximalClique(graph):
    """ Returns the largest cliques in a graph. 
        graph: list of lists of size n x n"""
    # Graph be n x n
    assert(len(graph) == len(graph[0]))

    max_cliques = []
    max_size = 0
    G = nx.from_numpy_matrix(graph)
    groups = nx.find_cliques(G)
    for i in groups:
        if len(i) == max_size:
            max_cliques.append(i)
        elif len(i) > max_size:
            max_cliques = []
            max_cliques.append(i)
            max_size = len(i)

    return max_cliques

if __name__ == '__main__':
    # dealing with a graph as list of lists 
    import numpy as np
    graph = np.array([[0,1,0,0,1,0,0],[1,0,1,0,1,0,0],[0,1,0,1,0,0,0],
             [0,0,1,0,1,1,0],[1,1,0,1,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,0,0,0]])
    print findMaximalClique(graph)