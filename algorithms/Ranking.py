import random

def Ranking(env):
    rank = list(range(0, env.online + env.offline))
    res = [-1] * env.realsize
    matched = [-1] * (env.online + env.offline)

    random.shuffle(rank)

    for i in range(env.realsize):
        match = -1
        for j in env.edges[env.online_type_list[i]]:
            if matched[j] == -1:
                if match == -1 or rank[j] > rank[match]:
                    match = j

        res[i] = match
        if (match != -1):
            matched[match] = i
    return res





