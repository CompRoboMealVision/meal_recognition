#!/usr/bin/env python

""" This script finds the largest clique in an undirected graph using the
    Bron-kerbosch algorithm. Code is adapted from:
    http://stackoverflow.com/questions/13904636/implementing-bron-kerbosch-algorithm-in-python
    ."""

def findMaximalClique(graph):
    """ Returns the largest cliques in a graph. 
        graph: list of lists of size n x n"""
    # Graph be n x n
    assert(len(graph) == len(graph[0]))

    max_cliques = []
    max_size = 0
    for i in findCliques([], range(len(graph)), [], graph):
        if len(i) == max_size:
            max_cliques.append(i)
        elif len(i) > max_size:
            max_cliques = []
            max_cliques.append(i)
            max_size = len(i)



    return max_cliques

def findCliques(clique, candidates, excluded, g):
    """ Implementation of the Bron-Kerbosch Alogorithm."""
    if not any((candidates, excluded)):
        yield clique
    for node in candidates[:]:
        new_clique = clique + [node]
        new_candidates = [v1 for v1 in candidates if v1 in numNeighbors(node, g)]
        new_excluded = [v1 for v1 in excluded if v1 in numNeighbors(node, g)]
        for r in findCliques(new_clique, new_candidates, new_excluded, g):
            yield r
        candidates.remove(node)
        excluded.append(node)

def numNeighbors(node, g):
    """ Determine number of neighbors of a given node in graph g."""
    return [i for i, n_v in enumerate(g[node]) if n_v]

if __name__ == '__main__':
    # dealing with a graph as list of lists 
    graph = [[0,1,0,0,1,0,0],[1,0,1,0,1,0,0],[0,1,0,1,0,0,0],
             [0,0,1,0,1,1,0],[1,1,0,1,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,0,0,0]]
    print findMaximalClique(graph)