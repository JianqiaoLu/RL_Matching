class edge():
    def __init__(self, v, cap, flow, rev):
        self.v = v
        self.cap = cap
        self.flow = flow
        self.rev = rev


class flow_graph():
    def __init__(self, S, T):
        self.s = S
        self.t = T
        self.adj = []
        for i in range(self.t + 1):
            self.adj.append([])

    def add_edge(self, x, y, cap):
        x_e = len(self.adj[x])
        y_e = len(self.adj[y])
        self.adj[x].append(edge(y, cap, 0, y_e));
        self.adj[y].append(edge(x, 0, 0, x_e));


    def max_flow(self):
        self.dep = [0] * (self.t + 1)
        self.cur = [0] * (self.t + 1)
        while self.bfs():
            self.cur[:] = 0
            while (self.dfs(self.s, 1 )):
                pass


    def bfs(self):
        q = []
        self.dep[:] = 0
        self.dep[self.s] = 1
        q.push(self.s)
        while len(q) :
            u = q.pop(0)
            for u_e in len(self.adj[u]):
                v =self.adj[u][u_e].v
                if(not self.dep[v] and self.adj[u][u_e].cap):
                    self.dep[v] = self.dep[u] + 1
                    q.append(v)
        return self.dep[self.t]

    def dfs(self, u, num):
        if (u == self.t or not num) :
            return num
        u_e = self.cur[u]
        while (u_e < len(self.adj[u])):
            v =  self.adj[u][u_e].v
            v_e = self.adj[u][u_e].rev
            if (self.dep[v] == self.dep[u] + 1 and self.adj[u][u_e].cap):
                d = self.dfs(v, min(num, self.adj[u][u_e].cap))
                if (d):
                    self.adj[u][u_e].cap -= d
                    self.adj[u][u_e].flow += d
                    self.adj[v][v_e].cap += d
                    self.adj[v][v_e].flow -= d
                    return d

            u_e += 1
        return 0








