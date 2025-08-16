"""
Install the choix library to run this code. Some functions in the code borrow implementation from the library.
https://github.com/lucasmaystre/choix
"""
import numpy as np
import pandas as pd
import choix
import random
import os
import os.path
import tqdm
from utils import init, get_args, print_metric
np.set_printoptions(precision=3, suppress=True)

def halve(m, a, scores, data, comp_matrix):
    """
    Compares each item in the row in pairs of 2 for a times and prmotes the winner to the next row.
    """
    size = np.size(a)
    if(size % 2 == 0):
        new_a = np.arange(int(size/2))
    else:
        new_a = np.arange(int(size/2)+1)
    num_votes = 0
    random.shuffle(a)
    for i in range(int(size/2)):
        p = a[2*i]
        q = a[2*i+1]
        win_p = 0
        win_q = 0
        for j in range(m):
            if(Vote(scores[p], scores[q])):
                win_p += 1
                data.append((p+1, q+1))
                comp_matrix[p+1][q+1] += 1
            else:
                win_q += 1
                data.append((q+1, p+1))
                comp_matrix[q+1][p+1] += 1
        num_votes += m
        if(win_p >= win_q):
            new_a[i] = p
        else:
            new_a[i] = q
    if(size % 2 == 1):
        new_a[int(size/2)] = a[size-1]
    return new_a, num_votes, data, comp_matrix

def SELECT(n, m, scores, data, comp_matrix):
    """
    Selects the topper among the given set of items using the SELECT algorithm routine.
    """
    a = np.arange(n)
    total_votes = 0
    while(np.size(a) > 1):
        a, num_votes, data, comp_matrix = halve(m, a, scores, data, comp_matrix)
        total_votes += num_votes
    top = a[0]
    return top, total_votes, data, comp_matrix

def reset(n, scores, initialization="heuristic"):
    """
    To intialize the algorithm using the first shoe-string of n-1 votes.

    initialization options:
    heuristic -> The default method as discussed in the paper.
    cyclic -> Compare (1,2), (2,3), ..., (n-1, n), (n, 1) to form a complete cycle of votes.
    SELECT -> Use the select algorithm routine to intialize the algorithm.
    """
    # Initializing the comparison matrix
    data = []
    comp_matrix = np.zeros((n+1, n+1))
    comp_matrix[0][0] = float("inf")
    for i in range(1, n+1):
        comp_matrix[i][i] = float("inf")
        comp_matrix[0][i] = float("inf")
        comp_matrix[i][0] = float("inf")
        data.append((i, 0))
        data.append((0, i))
    # Initializing the Shoe-String and getting the Ranking.
    if(initialization == "heuristic"):
        initial = n-1
        p = 1
        q = 2
        for i in range(initial):
            if(Vote(scores[p-1], scores[q-1])):
                data.append((p,q))
                comp_matrix[p][q] += 1
            else:
                data.append((q,p))
                comp_matrix[q][p] += 1
                p = q
            q = (q % n)+1
    elif(initialization == "cyclic"):
        initial = n
        p = 1
        q = 2
        for i in range(initial):
            if(Vote(scores[p-1], scores[q-1])):
                data.append((p,q))
                comp_matrix[p][q] += 1
            else:
                data.append((q,p))
                comp_matrix[q][p] += 1
            p = (p % n)+1
            q = (p % n)+1
    elif(initialization == "SELECT"):
        top, initial, data, comp_matrix = SELECT(n, 1, scores, data, comp_matrix)
    estimates = Rank_Centrality(n, data)
    estimates = normalize(estimates)
    # Initializing Markov Chain.
    chain = create_chain(n, data)
    A = np.identity(n+1, dtype = float) - chain
    A_hash = np.linalg.pinv(A)
    return data, initial, comp_matrix, chain, A_hash, estimates

def Vote(i, j):
    """
    Simulate voting on pair (a, b) when theit BTL scores are (i, j) respectively.
    """
    if(random.uniform(0, i+j) < i):
        return True
    else:
        return False

def normalize(array):
    """
    Normalize the BTL score vector such that sum of all scores is 1.
    """
    asum = np.sum(array)
    for i in range(0, len(array)):
        array[i] = array[i]/asum
    return array

def create_chain(n, data):
    """
    Create the underlying Markov Chain using the pairwise comparison data.
    """
    chain = np.zeros(((n+1), (n+1)))
    for winner, loser in data:
        chain[loser, winner] += 1.0
    # Transform the counts into ratios.
    idx = chain > 0  # Indices (i,j) of non-zero entries.
    chain[idx] = chain[idx] / (chain + chain.T)[idx]
    # Finalize the Markov chain by adding the self-transition rate.
    chain -= np.diag(chain.sum(axis=1))
    chain = chain / n
    for i in range(n+1):
        chain[i][i] += 1.0
    return chain

def get_ranks(scores):
    """
    Caluculate Ranks of items by assigning the average rank to the items with the same scores.
    """
    ranks = np.zeros(len(scores))
    tmp = [(scores[i], i) for i in range(len(scores))]
    tmp.sort(key=lambda x: x[0], reverse = True)
    (rank,n,i) = (1,1,0)
    while(i < len(scores)):
        j = i
        while((j < len(scores)-1) and (tmp[j][0] == tmp[j+1][0])):
            j += 1
        n = j-i+1
        for j in range(n):
            idx = tmp[i+j][1]
            ranks[idx] = rank + (n-1)/2
        rank += n
        i += n
    return ranks

def get_ranking(n, parameters):
    """
    Get the current ranking of items, their ranks as well as the current topper.
    """
    params = np.delete(parameters, 0)
    ranks = get_ranks(params)
    ranking = sorted(range(len(parameters)), key = lambda k: parameters[k])
    ranking.remove(0)
    toppers = np.where(ranks == ranks.min())[0]
    top = np.random.choice(toppers) + 1
    return ranking, ranks, top

def power_iter(n, comp_matrix, pair, phi):
    """
    L2-norm after one Power Iteration of the Markov Chain for a pair (i,j).
    """
    i = pair[0]
    j = pair[1]
    x = comp_matrix[j][i]+comp_matrix[i][j] 
    e = 1
    # m = (comp_matrix[j][i]+e)*np.sqrt(phi[i]**2 + phi[j]**2)/((x+2*e)*(x+1+2*e))
    m = (comp_matrix[j][i]+e)*(phi[i]+phi[j])/((x+2*e)*(x+1+2*e))
    return m

def least_squares(n, comp_matrix, pair):
    """
    Least-squares objective one-step minimization. 
    """
    i = pair[0]
    j = pair[1]
    x = comp_matrix[i][j]
    e = 1
    return np.log((x+1+e)/(x+e))

def IMC_prune(n, comp_matrix, pair, phi, MC, MC_sq, metric="L2"):
    """
    Incremental Markov Chain pruning for one pair to compute the metric for pick_a_pair function.
    In the pair (i,j), increment chain such that i wins over j.
    """
    i = pair[0]
    j = pair[1]
    
    # Add small epsilon to avoid division by zero
    eps = 1e-10
    
    if((comp_matrix[j][i]+comp_matrix[i][j]) == 0):
        delta_i = - ((comp_matrix[j][i])/(comp_matrix[j][i]+comp_matrix[i][j]+1))/n
    else:
        delta_i = - ((comp_matrix[j][i])/(comp_matrix[j][i]+comp_matrix[i][j]+1))/n + ((comp_matrix[j][i])/(comp_matrix[j][i]+comp_matrix[i][j]))/n
    
    denominator_k = 1+(-delta_i*MC[i,i])+(delta_i*MC[j,i])
    if abs(denominator_k) < eps:
        denominator_k = eps if denominator_k >= 0 else -eps
    k = phi[i]/denominator_k
    
    error = k * ((-delta_i*MC[i]) + (delta_i*MC[j]))
    pi = phi - error
    pi = normalize(pi)
    
    if abs(phi[i]) < eps:
        gamma = 0
    else:
        gamma = np.dot(error, MC[:, i])/phi[i]
    
    mid_term = k*((-delta_i*MC_sq[i]) + (delta_i*MC_sq[j])) - gamma*error
    
    if abs(phi[i]) < eps:
        new_MC_i = MC[i] + mid_term
        new_MC_j = MC[j] + mid_term
    else:
        new_MC_i = MC[i] + mid_term - MC[i,i]*error/phi[i]
        new_MC_j = MC[j] + mid_term - MC[j,i]*error/phi[i]
    
    if((comp_matrix[j][i]+comp_matrix[i][j]) == 0):
        delta_j = - ((comp_matrix[i][j]+1)/(comp_matrix[j][i]+comp_matrix[i][j]+1))/n
    else:
        delta_j = - ((comp_matrix[i][j]+1)/(comp_matrix[j][i]+comp_matrix[i][j]+1))/n + ((comp_matrix[i][j])/(comp_matrix[j][i]+comp_matrix[i][j]))/n
    
    denominator_new = 1+(-delta_j*new_MC_j[j])+(delta_j*new_MC_i[j])
    if abs(denominator_new) < eps:
        denominator_new = eps if denominator_new >= 0 else -eps
    
    new_error = (pi[j]/denominator_new) * ((-delta_j*new_MC_j) + (delta_j*new_MC_i))
    new_pi = pi - new_error
    new_pi = normalize(new_pi)
    
    result = np.linalg.norm(new_pi-phi)
    
    # Handle NaN or inf results
    if np.isnan(result) or np.isinf(result):
        return 0.0
    
    return result

def IMC_update(n, comp_matrix, pair, inc, phi, chain, MC, est, metric="L2"):
    """
    Incremental Markov Chain update when one row of the chain has changed.
    There are various ways possible to caluculate the disruption score but we stick to the default L2-norm as discussed in the paper.
    """
    i = pair[0]
    j = pair[1]
    q = np.copy(chain[i])
    p = np.copy(q)
    if(inc):
        p[j] = ((comp_matrix[j][i])/(comp_matrix[j][i]+comp_matrix[i][j]+1))/n
    else:
        p[j] = ((comp_matrix[j][i]+1)/(comp_matrix[j][i]+comp_matrix[i][j]+1))/n
    p[i] = q[i] + q[j] - p[j]
    delta = q - p
    error = (phi[i]/(1 + np.dot(delta, MC[:, i]))) * np.dot(delta, MC)
    pi = phi - error
    gamma = np.dot(error, MC[:, i])/phi[i]
    temp = MC - (gamma * np.identity(n+1, dtype = float))
    IP_hash = MC + np.ones(n+1)[:, np.newaxis]*np.dot(error, temp) - (MC[:, i][:, np.newaxis]*error)/phi[i]
    pi = normalize(pi)
    new_chain = np.copy(chain)
    new_chain[i] = p
    ranking1, ranks1, top1 = get_ranking(n, pi)
    ranking2, ranks2, top2 = get_ranking(n, est)
    if(metric == "top_drop"):
        # Top Rank Drop
        return ranks1[top2-1]-ranks2[top2-1], pi, new_chain, IP_hash
    elif(metric == "L2"):
        # L2-norm of Difference in score vectors
        return np.linalg.norm(pi-est), pi, new_chain, IP_hash
    elif(metric == "L1"):
        # L1-norm of Difference in rank vectors
        return np.linalg.norm(ranks1 - ranks2, 1), pi, new_chain, IP_hash
    elif(metric == "abs_rank_diff"):
        # Absolute Rank Difference for the pair
        return (abs(ranks1[i-1]-ranks2[i-1])+abs(ranks1[j-1]-ranks2[j-1]))/2, pi, new_chain, IP_hash
    elif(metric == "drop_gain"):
        # Rank drop gain for the pair
        if(ranks2[i-1] > ranks2[j-1]):
            return ((ranks2[i-1]-ranks1[i-1])+(ranks1[j-1]-ranks2[j-1]))/2, pi, new_chain, IP_hash
        else:
            return ((ranks1[i-1]-ranks2[i-1])+(ranks2[j-1]-ranks1[j-1]))/2, pi, new_chain, IP_hash

def pick_a_pair_old(n, comp_matrix, estimates, top, chain, A_hash, ranking, k=1, method="weighted"):
    """
    Pick the next pair to be compared by evaluating the disruption score for the set of possible pairs.
    We only choose to compare the current topper with one of the rest to form a pair and reduce computation cost.
    """
    pair = np.random.choice(a = np.arange(1,n+1), size=2, replace=False)
    cand_pairs = []
    max_array = []
    for p in range(1,k+1):
        for q in range(p+1,n+1):
            i = ranking[n-p]
            j = ranking[n-q]
            # IMC for (i, j)
            m1, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (i, j), True, estimates, chain, A_hash, estimates)
            m1, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (j, i), False, pi, new_chain, IP_hash, estimates)
            # IMC for (j, i)
            m2, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (j, i), True, estimates, chain, A_hash, estimates)
            m2, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (i, j), False, pi, new_chain, IP_hash, estimates)
            # Update Max
            if(method == "average"):
                # Average
                m = m1+m2
            elif(method == "weighted"):
                # Weighted
                m = (estimates[i]*m1 + estimates[j]*m2)/(estimates[i]+estimates[j])
            cand_pairs.append((i,j))
            max_array.append(m)
    candidates = np.where(max_array == np.max(max_array))[0]
    idx = np.random.choice(candidates)
    pair = cand_pairs[idx]
    return pair

def pick_a_pair(n, comp_matrix, estimates, chain, A_hash, ranking, ranks, top, compute="exact", method="weighted"):
    """
    Pick the next pair to be compared by evaluating the disruption score for the set of possible pairs.
    We only choose to compare the current topper with one of the rest to form a pair and reduce computation cost.
    """
    pair = np.random.choice(a = np.arange(1,n+1), size=2, replace=False)
    # Randomly select a pair to compare.
    if(compute == "random"):
        # print(compute, top, pair, ranks)
        return pair
    # Randomly select an item to compare with the top.
    if(compute == "random-top"):
        items = np.arange(1, n+1)
        possibles = np.delete(items, top-1)
        candidate = np.random.choice(possibles)
        pair = (top, candidate)
        # print(compute, top, pair, ranks)
        return pair
    # One iteration towards optimizing the log likelihood score - pick the top two items and compare.
    if(compute == "loglike"):
        # Exclude dummy item (index 0) from consideration
        valid_ranks = ranks[1:]  # Only consider items 1 to n
        toppers = np.where(valid_ranks == valid_ranks.min())[0] + 1
        toppers = toppers[toppers != top]
        if(len(toppers) == 0):
            min2 = np.min(valid_ranks[valid_ranks > valid_ranks.min()])
            toppers2 = np.where(valid_ranks == min2)[0] + 1
            # Filter out any invalid indices
            toppers2 = toppers2[toppers2 <= n]
            if len(toppers2) == 0:
                # Fallback to random selection
                items = np.arange(1, n+1)
                possibles = np.delete(items, top-1)
                top2 = np.random.choice(possibles)
            else:
                top2 = np.random.choice(toppers2)
            pair = (top, top2)
            # print(compute, top, pair, ranks)
            return pair
        else:
            top2 = np.random.choice(toppers)
            pair = (top, top2)
            # print(compute, top, pair, ranks)
            return pair
    cand_pairs = []
    max_array = []
    A_hash_sq = np.dot(A_hash, A_hash)
    for p in range(1, n+1):
        if(not (p == top)):
        # for q in range(p+1, n+1):
        # 	i = q
            i = top
            j = p
            # Exact update of the Markov Chain.
            if(compute == "exact"):
                m1 = IMC_prune(n, comp_matrix, (i, j), estimates, A_hash, A_hash_sq)
                m2 = IMC_prune(n, comp_matrix, (j, i), estimates, A_hash, A_hash_sq)
            # One power iteration of the Markov Chain.
            elif(compute == "power"):
                m1 = power_iter(n, comp_matrix, (i, j), estimates)
                m2 = power_iter(n, comp_matrix, (j, i), estimates)
            # One iteration towards optimizing the least squares objective.
            elif(compute == "squares"):
                m1 = least_squares(n, comp_matrix, (i, j))
                m2 = least_squares(n, comp_matrix, (j, i))
            # Update Max
            if(method == "average"):
                # Average
                m = m1+m2
            elif(method == "weighted"):
                # Weighted
                denominator = estimates[i]+estimates[j]
                if abs(denominator) < 1e-10:
                    m = (m1 + m2) / 2  # Fallback to average if weights sum to zero
                else:
                    m = (estimates[i]*m1 + estimates[j]*m2)/denominator
            
            # Handle NaN or inf values
            if np.isnan(m) or np.isinf(m):
                m = 0.0
                
            cand_pairs.append((i,j))
            max_array.append(m)
    
    # Handle empty max_array or all NaN/inf values
    if len(max_array) == 0:
        # Fallback to random selection
        items = np.arange(1, n+1)
        possibles = np.delete(items, top-1)
        if len(possibles) > 0:
            candidate = np.random.choice(possibles)
            pair = (top, candidate)
        else:
            # This should not happen, but as a last resort
            pair = (1, 2) if top != 1 else (1, 3) if n > 2 else (1, 2)
        return pair
    
    # Filter out NaN and inf values
    valid_indices = []
    valid_values = []
    for idx, val in enumerate(max_array):
        if not (np.isnan(val) or np.isinf(val)):
            valid_indices.append(idx)
            valid_values.append(val)
    
    if len(valid_values) == 0:
        # All values are invalid, fallback to random selection
        items = np.arange(1, n+1)
        possibles = np.delete(items, top-1)
        if len(possibles) > 0:
            candidate = np.random.choice(possibles)
            pair = (top, candidate)
        else:
            pair = (1, 2) if top != 1 else (1, 3) if n > 2 else (1, 2)
        return pair
    
    # Find candidates with maximum valid value
    max_val = np.max(valid_values)
    candidates = [valid_indices[i] for i, val in enumerate(valid_values) if val == max_val]
    
    if len(candidates) == 0:
        # This should not happen, but as a fallback
        idx = valid_indices[0]
    else:
        idx = candidates[np.random.choice(len(candidates))]
    
    pair = cand_pairs[idx]
    # print(compute, top, pair, ranks)
    return pair

def random_pick(n, top):
    """
    To random pick one of the items apart from the topper.
    """
    item = random.randint(1,n-1)
    if(item >= top):
        item += 1
    return (top, item)

def Rank_Centrality(n, data):
    """
    Run the Rank Centrality algorithm to get a score vector based on the pairwise data seen so far.
    """
    params = choix.rank_centrality(n+1, data, alpha=0.01)
    # estimates = np.exp(params)
    estimates = params
    return estimates

def run_simulation_custom(n, toppers, experiments, iterations, budget, recovery_count, performance_factor, 
						  compute="exact"):
    """
    Simulation for the case when the topper has a score x and rest have 100-x and we have an array of candidate x's.
    """
    aux_bgt = int(budget/int(budget/(n)))
    for tp, topper in tqdm.tqdm(enumerate(toppers), desc="toppers"):
        scores, true_top = init(n, topper=topper)
        for exp in tqdm.tqdm(range(experiments), desc="experiments"):
            for itr in tqdm.tqdm(range(iterations), desc="iterations"):
                data, initial, comp_matrix, chain, A_hash, estimates = reset(n, scores)
                ranking, ranks, top = get_ranking(n, estimates)
                for batch in range(int(budget/(n))):
                    for i in range(initial, aux_bgt):
                        # (p,q) = pick_a_pair(n, comp_matrix, estimates, top, chain, A_hash, ranking)
                        (p,q) = pick_a_pair(n, comp_matrix, estimates, chain, A_hash, ranking, ranks, top, compute=compute)
                        if(Vote(scores[p-1], scores[q-1])):
                            data.append((p,q))
                            # comp_matrix[p][q] += 1
                            m, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (p,q), True, estimates, chain, A_hash, estimates)
                            m, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (q,p), False, pi, new_chain, IP_hash, estimates)
                            comp_matrix[p][q] += 1
                        else:
                            data.append((q,p))
                            # comp_matrix[q][p] += 1
                            m, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (q,p), True, estimates, chain, A_hash, estimates)
                            m, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (p,q), False, pi, new_chain, IP_hash, estimates)
                            comp_matrix[q][p] += 1
                        estimates = np.copy(pi)
                        ranking, ranks, top = get_ranking(n, estimates)
                        A_hash = np.copy(IP_hash)
                        chain = np.copy(new_chain)
                    ranking, ranks, top = get_ranking(n, estimates)
                    initial = 0
                    # Update the metrics
                    if(true_top == (top-1)):
                        recovery_count[tp][batch][exp] += 1
                    performance_factor[tp][batch][exp] += ranks[true_top]

    performance_factor /= iterations

    return ranking, ranks, data, scores, true_top, estimates, recovery_count, performance_factor

def run_simulation(n, experiments, iterations, budget, recovery_count, performance_factor, current_top,
                   precomputed=True, dataset=None, compute="exact"):
    """
    Simulation of the algorithm for the synthetic and real-world dataset cases.
    """
    aux_bgt = int(budget/int(budget/(n)))
    scores, true_top = init(n, precomputed=precomputed, dataset=dataset)
    true_ranks = get_ranks(scores)
    for exp in tqdm.tqdm(range(experiments), desc="experiments"):
        for itr in tqdm.tqdm(range(iterations), desc="iterations"):
            data, initial, comp_matrix, chain, A_hash, estimates = reset(n, scores)
            ranking, ranks, top = get_ranking(n, estimates)
            for batch in range(int(budget/(n))):
                for i in range(initial, aux_bgt):
                    # (p,q) = pick_a_pair(n, comp_matrix, estimates, top, chain, A_hash, ranking)
                    (p,q) = pick_a_pair(n, comp_matrix, estimates, chain, A_hash, ranking, ranks, top, compute=compute)
                    if(Vote(scores[p-1], scores[q-1])):
                        data.append((p,q))
                        # comp_matrix[p][q] += 1
                        m, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (p,q), True, estimates, chain, A_hash, estimates)
                        m, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (q,p), False, pi, new_chain, IP_hash, estimates)
                        comp_matrix[p][q] += 1
                    else:
                        data.append((q,p))
                        # comp_matrix[q][p] += 1
                        m, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (q,p), True, estimates, chain, A_hash, estimates)
                        m, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (p,q), False, pi, new_chain, IP_hash, estimates)
                        comp_matrix[q][p] += 1
                    estimates = np.copy(pi)
                    ranking, ranks, top = get_ranking(n, estimates)
                    A_hash = np.copy(IP_hash)
                    chain = np.copy(new_chain)
                ranking, ranks, top = get_ranking(n, estimates)
                initial = 0
                # Update the metrics
                if(true_top == (top-1)):
                    recovery_count[batch][exp] += 1
                performance_factor[batch][exp] += ranks[true_top]
                current_top[batch][exp] += true_ranks[top-1]

    performance_factor /= iterations
    current_top /= iterations

    return ranking, ranks, data, scores, true_top, estimates, recovery_count, performance_factor, current_top

"""
Runner Code.
"""
if __name__ == "__main__":
    args = get_args()

    if args.dataset is not None:
        if args.dataset == "sushi-A":
            N = 10
        elif args.dataset == "sushi-B" or args.dataset == "jester" or args.dataset == "netflix" or args.dataset == "movielens":
            N = 100
        else:
            print("Invalid Dataset")
            exit()
        Experiments = args.experiments
        Iterations = args.iterations
        Budget = N*args.budget

        print(f"Running {Experiments} experiments of {Iterations} iterations each\n on the \"{args.dataset}\" dataset where N = {N}.\n")

        RC = np.zeros((int(Budget/(N)), Experiments))
        PF = np.zeros((int(Budget/(N)), Experiments))
        CT = np.zeros((int(Budget/(N)), Experiments))

        Ranking, Ranks, Data, Scores, True_top, Estimates, RC, PF, CT = run_simulation(N, Experiments, Iterations, Budget, RC, PF, CT,
                                                                                       precomputed=args.precomputed, dataset=args.dataset,
                                                                                       compute=args.compute)
        
        exp = f"{args.dataset}_N_{N}" + "_" + args.name

        log_data = {'Budget' : [N*b for b in range(1, args.budget+1)],
        			'Recovery_Counts' : np.mean(RC, axis=1),
        			'RC_std' : np.sqrt(np.var(RC, axis=1)),
        			'Reported_Rank_of_True_Winner(PF)' : np.mean(PF, axis=1),
        			'PF_std' : np.sqrt(np.var(PF, axis=1)),
        			'True_Rank_of_Reported_Winner(CT)' : np.mean(CT, axis=1),
        			'CT_std' : np.sqrt(np.var(CT, axis=1)),
        			'Experiment' : [exp for b in range(args.budget)]
        			}

        df = pd.DataFrame(log_data, columns=['Budget', 'Recovery_Counts', 'RC_std', 'Reported_Rank_of_True_Winner(PF)', 
        									 'PF_std', 'True_Rank_of_Reported_Winner(CT)', 'CT_std', 'Experiment'])

        if(not args.no_save):
            df.to_csv(os.path.join(args.save_dir, exp+".csv"), index=False)

        print_metric("Recovery_Counts", RC)
        print_metric("Reported_Rank_of_True_Winner", PF)
        print_metric("True_Rank_of_Reported_Winner", CT)

    elif args.toppers is not None:
        Toppers = args.toppers

        N = args.n
        Experiments = args.experiments
        Iterations = args.iterations
        Budget = N*args.budget

        print(f"Running {Experiments} experiments of {Iterations} iterations each with N = {N}\n such that topper has score x and rest have score 100-x\n where x belongs to the array {args.toppers}.\n")

        RC = np.zeros((len(Toppers), int(Budget/(N)), Experiments))
        PF = np.zeros((len(Toppers), int(Budget/(N)), Experiments))

        Ranking, Ranks, Data, Scores, True_top, Estimates, RC, PF = run_simulation_custom(N, Toppers, Experiments, Iterations, Budget, RC, PF, compute=args.compute)

        exp = f"toppers_N_{N}" + "_" + args.name

        log_data = {'Budget' : [N*b for b in range(1, args.budget+1)]*len(Toppers),
        			'Recovery_Counts' : np.mean(RC, axis=2).reshape(args.budget*len(Toppers)),
        			'RC_std' : np.sqrt(np.var(RC, axis=2)).reshape(args.budget*len(Toppers)),
        			'Reported_Rank_of_True_Winner(PF)' : np.mean(PF, axis=2).reshape(args.budget*len(Toppers)),
        			'PF_std' : np.sqrt(np.var(PF, axis=2)).reshape(args.budget*len(Toppers)),
        			'Experiment' : [exp+f"_{Toppers[int(t/args.budget)]}" for t in range(len(Toppers)*args.budget)]
        			}

        df = pd.DataFrame(log_data, columns=['Budget', 'Recovery_Counts', 'RC_std', 'Reported_Rank_of_True_Winner(PF)', 
        									 'PF_std', 'Experiment'])

        if(not args.no_save):
            df.to_csv(os.path.join(args.save_dir, exp+".csv"), index=False)

        print_metric("Recovery_Counts", RC, Toppers)
        print_metric("Reported_Rank_of_True_Winner", PF, Toppers)

    else:
        N = args.n
        Experiments = args.experiments
        Iterations = args.iterations
        Budget = N*args.budget

        print(f"Running {Experiments} experiments of {Iterations} iterations each with N = {N}\n topper has score 100 and the rest have scores\n uniformly at random in 0 to 75.\n")
        if args.precomputed:
        	print("This score vector has been precomputed and stored for reproducibility.\n")

        RC = np.zeros((int(Budget/(N)), Experiments))
        PF = np.zeros((int(Budget/(N)), Experiments))
        CT = np.zeros((int(Budget/(N)), Experiments))

        Ranking, Ranks, Data, Scores, True_top, Estimates, RC, PF, CT = run_simulation(N, Experiments, Iterations, Budget, RC, PF, CT,
                                                                                       precomputed=args.precomputed, dataset=args.dataset,
                                                                                       compute=args.compute)

        exp = f"synthetic_N_{N}" + "_" + args.name

        log_data = {'Budget' : [N*b for b in range(1, args.budget+1)],
        			'Recovery_Counts' : np.mean(RC, axis=1),
        			'RC_std' : np.sqrt(np.var(RC, axis=1)),
        			'Reported_Rank_of_True_Winner(PF)' : np.mean(PF, axis=1),
        			'PF_std' : np.sqrt(np.var(PF, axis=1)),
        			'True_Rank_of_Reported_Winner(CT)' : np.mean(CT, axis=1),
        			'CT_std' : np.sqrt(np.var(CT, axis=1)),
        			'Experiment' : [exp for b in range(args.budget)]
        			}

        df = pd.DataFrame(log_data, columns=['Budget', 'Recovery_Counts', 'RC_std', 'Reported_Rank_of_True_Winner(PF)', 
        									 'PF_std', 'True_Rank_of_Reported_Winner(CT)', 'CT_std', 'Experiment'])

        if(not args.no_save):
            df.to_csv(os.path.join(args.save_dir, exp+".csv"), index=False)

        print_metric("Recovery_Counts", RC)
        print_metric("Reported_Rank_of_True_Winner", PF)
        print_metric("True_Rank_of_Reported_Winner", CT)
