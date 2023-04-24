from algorithms.flow_graph import flow_graph

def Max_matching(env):
    s = env.realsize + env.offline
    t = s + 1;
    g = flow_graph(s, t)

    for i in range(env.realsize):
        for j in env.edges[env.online_type_list[i]]:
            g.add_edge(i, j - env.online + env.realsize, 1)

    for i in range(env.realsize):
        g.add_edge(s, i, 1)

    for j in range(env.realsize, env.realsize + env.offline):
        g.add_edge(j, t, 1)

    g.max_flow()

    res = [-1] * env.realsize
    for i in range(env.realsize) :
        for e in g.adj[i] :
            if e.flow > 0 :
                res[i] = e.v - env.realsize + env.online

    del g
    return res

