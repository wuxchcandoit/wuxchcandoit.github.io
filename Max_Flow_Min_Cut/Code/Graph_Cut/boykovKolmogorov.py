from collections import deque
from copy import deepcopy
from dataStructure import queue
from queue import Queue
import numpy as np

class BoykovKolmogorov(object):
    def __init__(self, o_graph, s, t, h, w, store_parent_info=True, perfect_info=True, store_child_info=True):
        self.flow = 0
        self.n = h * w + 2
        self.parent = {s: None, t: None}
        self.store_parent_info = store_parent_info
        if self.store_parent_info:
            self.parent_info = {s: 0, t: 0}
        self.store_child_info = store_child_info
        if self.store_child_info:
            self.child_info = dict()
            for i in range(self.n):
                self.child_info[i] = set()
        self.perfect_info = perfect_info

        # S and T are the sets of nodes belonging to the trees rooted by the source and target nodes respectively
        self.S = set([s])
        self.T = set([t])
        self.A = queue([s, t]) # think
        self.O = queue()       # think

        self.s = s
        self.t = t
        self.h = h
        self.w = w
        self.graph = deepcopy(o_graph)
        # print(o_graph)
        self.o_graph = o_graph

    # def special_neighbours(self, v):
    #     neighbours = []
    #     if v == self.s or v == self.t:
    #         for u in range(len(self.graph) - 2):
    #             neighbours.append(u)
    #     else:
    #        for u in range(len(self.graph)):
    #            neighbours.append(u)
    #     return neighbours
    
    def special_neighbours(self, v):
        neighbours = []
        if v == self.s or v == self.t:
            for u in range(self.n - 2):
                neighbours.append(u)
        else:
            x = v // self.w
            y = v % self.w
            for (i, j) in [(x + 1, y), (x, y + 1), (x + 1, y + 1), (x + 1, y - 1), (x - 1, y), (x, y - 1), (x - 1, y - 1), (x - 1, y + 1)]:
                if 0 <= i < self.h and 0 <= j < self.w and (i != x or j != y):
                    neighbours.append(i * self.w + j)
            neighbours.append(self.s)
            neighbours.append(self.t)
        return neighbours

    def set_distance_to_origin(self, n, distance):
        q = deque([n])
        self.parent_info[n] = distance
        visited = set([n])
        # if n in self.S:
        #     current_tree = self.S
        # else:
        #     current_tree = self.T
        while len(q) != 0:
            current_node = q.pop()
            if self.store_child_info:
                children = self.child_info[current_node]
            else:
                children = [k for k, v in self.parent.items() if v == current_node]
            for child in children:
                if child not in visited:
                    q.appendleft(child)
                    visited.add(child)
                    if distance == -1:
                        self.parent_info[child] = -1
                    else:
                        self.parent_info[child] = self.parent_info[current_node] + 1
    
    def path(self, s_node, t_node):
        s_path = [s_node]
        s_start = self.s
        p = s_path[0]
        while p != s_start:
            p = self.parent[p]
            s_path.append(p)

        t_path = [t_node]
        t_end = self.t
        p = t_path[0]
        while p != t_end:
            p = self.parent[p]
            t_path.append(p)
        
        s_path.reverse()
        return s_path + t_path

    def grow(self):
        while self.A.length():
            p = self.A.get()
            if p in self.S:
                current_tree = self.S
                other_tree = self.T
                in_flag = False
                out_flag = True
            else:
                current_tree = self.T
                other_tree = self.S
                in_flag = True
                out_flag = False
            for q in self.special_neighbours(p):
                if (in_flag and (q, p) in self.graph.keys() and self.graph[(q, p)] > 0) or (out_flag and (p, q) in self.graph.keys() and self.graph[(p, q)] > 0):
                    # Found a free node with sparse capacity and add it to current tree
                    if q not in self.S and q not in self.T:
                        current_tree.add(q)
                        self.parent[q] = p
                        if self.store_child_info:
                            self.child_info[p].add(q)
                        if self.store_parent_info:
                            if self.perfect_info:
                                self.set_distance_to_origin(q, self.parent_info[p] + 1)
                            else:
                                self.parent_info[q] = self.parent_info[p] + 1
                        self.A.add(q)
                    # Encounter the other tree, so return an augmenting path
                    if q in other_tree:
                        self.A.add(p)
                        if current_tree == self.S:
                            return self.path(p, q)
                        else:
                            return self.path(q, p)
        return None
        
    def augment(self, P):
        pairs = [(P[i], P[i+1]) for i in range(len(P) - 1)]
        delta = min([self.graph[(p, q)] for p, q in pairs])
        self.flow += delta
        # print("P: ", P, "delta: ", delta, "S = ", self.S, " T = ", self.T, " O = ", self.O, " A = ", self.A)
        for p, q in pairs:
            self.graph[(p, q)] -= delta
            if (q, p) not in self.graph:
                self.graph[(q, p)] = 0
            self.graph[q, p] += delta
            # print("p = ", p, " q = ", q, "graph[p][q] = ", self.graph[p][q])
            # In source tree, the target of an edge is the orphan
            if p in self.S and q in self.S and self.graph[(p, q)] == 0:
                if self.store_child_info:
                    self.child_info[self.parent[q]].remove(q)
                self.parent[q] = None
                if self.store_parent_info:
                    self.set_distance_to_origin(q, -1)
                self.O.add(q)
                # print("S add in O: ", q)
            # In target tree, the source side of an edge is the orphan
            if p in self.T and q in self.T and self.graph[(p, q)] == 0:
                if self.store_child_info:
                    self.child_info[self.parent[p]].remove(p)
                self.parent[p] = None
                if self.store_parent_info:
                    self.set_distance_to_origin(p, -1)
                self.O.add(p)
                # print("T add in O: ", p)
    
    # check whether a node n has a connection back to the root of a tree, either source or target
    def rooted(self, n):
        if self.store_parent_info:
            try:
                distance_to_parent = self.parent_info[n]
                return distance_to_parent != -1
            except:
                return False
        else:
            while n != None and n != self.s and n != self.t:
                n = self.parent[n]
            return n != None

    def adopt(self):
        # print("adopt O: ", self.O)
        while self.O.length():
            p = self.O.get()
            if p in self.S:
                current_tree = self.S
                other_tree = self.T
                in_flag = True
                out_flag = False
            else:
                current_tree = self.T
                other_tree = self.S
                in_flag = False
                out_flag = True
            new_parent = None
            neighbours = self.special_neighbours(p)
            for q in neighbours:
                if q in current_tree and self.rooted(q) and ((in_flag and (q, p) in self.graph.keys() and self.graph[(q, p)] > 0) or (out_flag and (p, q)in self.graph.keys() and self.graph[(p, q)] > 0)):
                    # print("orpahn p = ", p, "possible q = ", q)
                    if self.store_parent_info:
                        if new_parent == None:
                            new_parent = q
                        if self.parent_info[q] < self.parent_info[new_parent] and self.parent_info[q] != -1:
                            new_parent = q
                    else:
                        new_parent = q
                        break
            self.parent[p] = new_parent
            # print("new parent of p = ", new_parent)
            # perfect_info had better be True
            # if self.store_parent_info and new_parent != None:
            #     if self.perfect_info:
            #         self.set_distance_to_origin(p, self.parent_info[new_parent] + 1)
            #     else:
            #         self.parent_info[p] = self.parent_info[new_parent] + 1
            if self.store_parent_info and new_parent != None: # and self.perfect_info:
                self.set_distance_to_origin(p, self.parent_info[new_parent] + 1)
            if self.store_child_info and new_parent != None:
                self.child_info[new_parent].add(p)
            if self.parent[p] == None:
                for q in neighbours:
                    # if (in_flag and (q, p) in self.graph.keys() and self.graph[(q, p)] > 0) or (out_flag and (p, q) in self.graph.keys() and self.graph[(p, q)] > 0):
                    #     if q not in self.A:
                    #         self.A.add(q)
                    if q in current_tree:
                        if (in_flag and (q, p) in self.graph.keys() and self.graph[(q, p)] > 0) or (out_flag and (p, q) in self.graph.keys() and self.graph[(p, q)] > 0):
                            if q not in self.A:
                                self.A.add(q)
                    elif q in other_tree:
                        if (not in_flag and (p, q) in self.graph.keys() and self.graph[(p, q)] > 0) or (not out_flag and (q, p) in self.graph.keys() and self.graph[(q, p)] > 0):
                            if q not in self.A:
                                self.A.add(q)
                    if q in current_tree:
                        if self.parent[q] == p:
                            if self.store_child_info:
                                self.child_info[p].remove(q)
                            self.parent[q] = None
                            if self.store_parent_info: # and self.perfect_info:
                                self.set_distance_to_origin(q, -1)
                            # print("Add to O: ", q)
                            self.O.add(q)
                current_tree.remove(p)
                if p in self.A:
                    self.A.remove(p)

    def max_flow(self):
        cnt = 1
        while True:
            P = self.grow()
            print(cnt)
            cnt+=1
            if P is None:
                break
            self.augment(P)
            self.adopt()

        # print(self.S)
        # print(self.T)

        # result, visited = self.bfs() # self.dfs()
        # cuts = []
        # if result == False:
        #     for i in range(self.n):
        #         for j in range(self.n):
        #             if i in visited and j not in visited and (i, j) in self.o_graph.keys() and self.o_graph[(i, j)] > 0:
        #                 cuts.append((i, j))

        belong, boundary = self.min_cut()
        return belong, boundary

    def min_cut(self):
        S = self.S.copy()
        T = self.T.copy()
        S.remove(self.s)
        T.remove(self.t)
        S = list(S)
        T = list(T)
        belong = []
        belong.extend(list(zip([self.s] * len(S), S)))
        belong.extend(list(zip(T, [self.t] * len(T))))
        # return belong
        boundary = []
        # for v in S:
        #     tag = False
        #     neighbours = self.special_neighbours(v)
        #     for u in neighbours:
        #         if u in T:
        #             tag = True
        #             boundary.append(u)
        #     if tag:
        #         boundary.append(v)
        return belong, boundary


    def bfs(self):
        q = Queue()
        q.put(self.s)
        visited = set([self.s])
        # parent = dict()
        # parent[self.s] = -1
        while not q.empty():
            u = q.get()
            neighbours = self.special_neighbours(u)
            for v in neighbours:
                if v not in visited and (u, v) in self.graph.keys() and self.graph[(u, v)] > 0:
                    q.put(v)
                    # parent[v] = u
                    visited.add(v)
                    if v == self.t:
                        return True, visited
        return False, visited

    def dfs(self):
        stack = [self.s]
        visited = set([])
        while(stack):
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                if v == self.t:
                    return True, visited
                neighbours = self.special_neighbours(v)
                for u in neighbours:
                    if (v, u) in self.graph.keys() and self.graph[(v, u)] > 0:
                        stack.append(u)
        return False, visited

'''
def bfs(rGraph, V, s, t, graph, h, w, parent):
    q = Queue()
    global visited
    visited = np.zeros(V, dtype=bool)
    q.put(s)
    visited[s] = True
    parent[s]  = -1

    while not q.empty():
        u = q.get()
        if u == s or u == t:
            for v in range(V):
                if (not visited[v]) and rGraph[u][v] > 0:
                    q.put(v)
                    parent[v] = u
                    visited[v] = True
                    if(v == t):
                        return True
        else:
            for p in speical_neighbours(u, h, w):
                v = p[0] * w + p[1]
                # if (not visited[v]) and (rGraph[u][v] > 0 or rGraph[v][u] - graph[v][u] > 0):
                if (not visited[v]) and (rGraph[u][v] > 0):
                    q.put(v)
                    parent[v] = u
                    visited[v] = True
                    if(v == t):
                        return True
            for v in list([s, t]):
                if (not visited[v]) and rGraph[u][v] > 0:
                    q.put(v)
                    parent[v] = u
                    visited[v] = True
                    if(v == t):
                        return True
    return False
'''
# graph = np.zeros((8, 8), dtype='int32')
# s = 6
# t = 7
# h = 2
# w = 3
# graph[s, 0] = 10
# graph[s, 1] = 5
# graph[s, 2] = 15
# graph[0, 1] = 4
# graph[0, 3] = 9
# graph[0, 4] = 15
# graph[1, 2] = 4
# graph[1, 4] = 8
# graph[2, 5] = 16
# graph[3, 4] = 15
# graph[4, 5] = 15
# graph[5, 1] = 6
# graph[3, t] = 10
# graph[4, t] = 10
# graph[5, t] = 10
# bk = BoykovKolmogorov(graph, s, t, h, w)
# flow = bk.max_flow()