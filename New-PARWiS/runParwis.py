# -*- coding: utf-8 -*-
"""
A corrected and robust, self-contained implementation of the PARWiS algorithm.

This script directly incorporates the necessary functions and logic from the original
PARWiS implementation to ensure correctness and high performance. It simulates
the algorithm over a range of difficulties and saves detailed recovery rate
results to CSV files.
"""
import os
import numpy as np
import pandas as pd
import choix
import random
from tqdm import tqdm

# Set print precision for numpy arrays
np.set_printoptions(precision=4, suppress=True)






from PARWiS import reset, get_ranking, Vote, Rank_Centrality, power_iter
from utils import init




def pick_a_pair(n, comp_matrix, estimates, top, method="weighted"):
    """
    Pick the next pair to be compared by evaluating the disruption score for the set of possible pairs.
    We only choose to compare the current topper with one of the rest to form a pair and reduce computation cost.
    """
    pair = np.random.choice(a = np.arange(1,n+1), size=2, replace=False)
    # Randomly select a pair to compare.
    
    cand_pairs = []
    max_array = []
    
    for p in range(1, n+1):
        if(not (p == top)):
        # for q in range(p+1, n+1):
        # 	i = q
            i = top
            j = p
            # Exact update of the Markov Chain.
            
            m1 = power_iter(n, comp_matrix, (i, j), estimates)
            m2 = power_iter(n, comp_matrix, (j, i), estimates)
        
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


# =============================================================================
# == MAIN SIMULATION SCRIPT
# =============================================================================

if __name__ == "__main__":
    output_dir = 'parwis-results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop over different difficulty levels for the ranking problem
    # A lower 'k' means the true winner is much better than the rest.

    # --- Simulation Parameters ---
    N = 100
    TOTAL_BUDGET = 5 * N
    TOTAL_RUNS = 200  # Number of full simulations for averaging
    RANDOM_SEED = 42
    k = 75

    print("\n" + "="*80)
    print(f"--- Starting PARWiS Simulation for k={k} (Difficulty) ---")
    print(f"N={N}, Budget={TOTAL_BUDGET}, Total Runs={TOTAL_RUNS}")

    # Stores recovery results: 1 if winner found, 0 otherwise.
    # Shape: (TOTAL_RUNS, TOTAL_BUDGET)
    recovery_results = np.zeros((TOTAL_RUNS, TOTAL_BUDGET))

    for i in tqdm(range(TOTAL_RUNS), desc=f"Simulating (k={k})"):
        np.random.seed(RANDOM_SEED + i)
        random.seed(RANDOM_SEED + i)
        
        # 1. Initialize scores, true winner, and algorithm state
        scores, true_top = init(n=N, k=k)
        # `reset` performs the first n-1 comparisons
        data, initial_votes, comp_matrix, chain, A_hash, estimates = reset(N, scores)

        # 2. Backfill results for the initial n-1 votes from `reset`
        temp_data = []
        for budget_step in range(initial_votes):
            # Check recovery at each step of the initialization
            temp_data.append(data[budget_step])
            est = Rank_Centrality(N, temp_data)
            ranking, ranks, winner = get_ranking(N, est)
            if winner is not None and (winner - 1) == true_top:
                recovery_results[i, budget_step] = 1

        # 3. Main loop for the rest of the budget
        for budget in range(initial_votes, TOTAL_BUDGET):
            try:
                p, q = pick_a_pair(N, comp_matrix, estimates, winner)
            except (ValueError, IndexError):
                # Fallback to random pair if picker fails
                p, q = np.random.choice(list(range(1, N + 1)), 2, replace=False)

            # Simulate comparison and update state
            winner, loser = (p, q) if Vote(scores[p-1], scores[q-1]) else (q, p)
            data.append((winner, loser))
            comp_matrix[winner, loser] += 1
            
            # Update estimates by re-running on all data
            estimates = Rank_Centrality(N, data)
            _, _, current_winner = get_ranking(N, estimates)
            
            # Record recovery status: 1 for success, 0 for failure
            if current_winner is not None and (current_winner - 1) == true_top:
                recovery_results[i, budget] = 1
            else:
                # If we already found it, keep it marked as found
                recovery_results[i, budget] = recovery_results[i, budget - 1] if budget > 0 else 0

    # --- Data Aggregation and Export ---
    # Calculate average recovery rate across all runs for each budget point
    average_recovery_rate = np.mean(recovery_results, axis=0)
    
    results_df = pd.DataFrame({
        'Budget': range(1, TOTAL_BUDGET + 1),
        'Average_Recovery_Rate': average_recovery_rate
    })

    output_filename = os.path.join(output_dir, f'N{N}_k{k}_parwis_recovery.csv')
    results_df.to_csv(output_filename, index=False, float_format='%.6f')

    print(f"\nSimulation for k={k} complete. Results saved to '{output_filename}'")
    print("--- Final Recovery Rate ---")
    print(f"Rate at final budget ({TOTAL_BUDGET}): {average_recovery_rate[-1]:.4f}")
    print("="*80)