from queue import Queue
import numpy as np


def special_neighbours(v, s, t, h, w):
    neighbours = []
    n = h * w
    if v == s or v == t:
        for u in range(n):
            neighbours.append(u)
    else:
        x = v // w
        y = v % w
        for (i, j) in [(x + 1, y), (x, y + 1), (x + 1, y + 1), (x + 1, y - 1), (x - 1, y), (x, y - 1), (x - 1, y - 1), (x - 1, y + 1)]:
            if 0 <= i < h and 0 <= j < w and (i != x or j != y):
                neighbours.append(i * w + j)
        neighbours.append(s)
        neighbours.append(t)
    return neighbours


def bfs(graph_copy, s, t, h, w, parent):
    q = Queue()
    global visited
    visited = set([s])
    q.put(s)

    while not q.empty():
        u = q.get()
        neighbours = special_neighbours(u, s, t, h, w)
        for v in neighbours:
            if v not in visited and (u, v) in graph_copy.keys() and graph_copy[(u, v)] > 0:
                q.put(v)
                parent[v] = u
                visited.add(v)
                if v == t:
                    return True
    return False


def FordFulkerson(graph, s, t, h, w):
    print("Running Ford-Fulkerson algorithm")
    graph_copy = graph.copy()
    n = len(graph)
    parent = np.zeros(n, dtype='int32')

    while bfs(graph_copy, s, t, h, w, parent):
        flow = float("inf")
        v = t
        while v != s:
            u = parent[v]
            # pathFlow = min(min(pathFlow, rGraph[u][v]), rGraph[v][u] - graph[v][u])
            flow = min(flow, graph_copy[(u, v)])
            v = parent[v] # v = u

        v = t
        while v != s:
            u = parent[v]
            graph_copy[(u, v)] -= flow
            if (v, u) not in graph_copy.keys():
                graph_copy[(v, u)] = 0
            graph_copy[(v, u)] += flow
            v = parent[v] # v = u
        parent = np.zeros(n, dtype='int32')
        
    cuts = []

    for i in range(n):
        for j in range(n):
            if i in visited and j not in visited and (i, j) in graph.keys():
                cuts.append((i, j))
    return cuts