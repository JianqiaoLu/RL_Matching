// Flow graph stored in adjacency list representation


struct flow_graph{

    // Edge u -> v with capacity and flow
    // rev helps find the reversed edge v -> u
    struct edge
    {
        int v, cap, flow, rev;
        edge(int v, int cap, int flow, int rev) : v(v), cap(cap), flow(flow), rev(rev){}
    };

    // Graph stored by adjacency lists
    vector<vector<edge>> adj;

    vector<int> dep, cur;

    // Source and Sink
    int s, t;

    const int inf = 1e9;


    // Initialize flow graph with source S and sink T
    // NOTE: Assume T is the vertex with largest label
    flow_graph(int S, int T)
    {
        s = S, t = T;
        adj.resize(t + 1);
        for (int i = 0; i <= t; i++)
            adj[i] = {};
    }

    // Add an edge x -> y with capacity
    void add_edge(int x, int y, int cap)
    {
        int x_e = adj[x].size(), y_e = adj[y].size();
        adj[x].push_back(edge(y, cap, 0, y_e));
        adj[y].push_back(edge(x, 0, 0, x_e));
    }

    // Assign levels to vertices by BFS
    bool bfs()
    {
        queue<int> q;
        fill(dep.begin(), dep.end(), 0);
        dep[s] = 1;
        q.push(s);
        do
        {
            int u = q.front();
            q.pop();
            for (int u_e = 0; u_e < (int)adj[u].size(); u_e++){
                int v = adj[u][u_e].v;
                if (!dep[v] && adj[u][u_e].cap){
                    dep[v] = dep[u] + 1;
                    q.push(v);
                }
            }
        }
        while(!q.empty());

        return dep[t];
    }

    // Send flows in G by DFS in level graph
    int dfs(int u, int num)
    {
        if ( (u == t) || !num ) return num;
        for (int &u_e = cur[u]; u_e < (int)adj[u].size(); u_e++)
        {
            int v = adj[u][u_e].v, v_e = adj[u][u_e].rev;
            if ((dep[v] == dep[u] + 1) && adj[u][u_e].cap)
            {
                int d = dfs(v, min(num, adj[u][u_e].cap));
                if (d) {
                    adj[u][u_e].cap -= d;
                    adj[u][u_e].flow += d;

                    adj[v][v_e].cap += d;
                    adj[v][v_e].flow -= d;
                    return d;
                }
            }
        }
        return 0;
    }

    // Compute maximum flow by Dinic's algorithm
    void max_flow()
    {
        dep.resize(t + 1);
        cur.resize(t + 1);
        while (bfs()) {
            fill(cur.begin(), cur.end(), 0);
            while (dfs(s, inf));
        }
    }


    // Compute the (canonical) reachability min-cut from residual graph
    // Must call maxflow() before calling this
    vector<int> min_cut()
    {
        vector<int> inS(t + 1, 0);
        queue<int> Q;
        Q.push(s);
        inS[s] = 1;

        while(!Q.empty())
        {
            int u = Q.front();
            Q.pop();

            for (int u_e = 0; u_e < (int)adj[u].size(); u_e++){
                int v = adj[u][u_e].v;
                if (!inS[v] && adj[u][u_e].cap){
                    inS[v] = 1;
                    Q.push(v);
                }
            }
        }
        return inS;
    }
};
