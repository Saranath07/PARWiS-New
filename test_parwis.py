import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from collections import deque
import random
import copy

# Ensure the PARWiS and utils modules are accessible.
# These would typically be in files named PARWiS.py and utils.py in the same directory.
from PARWiS import reset, Vote, Rank_Centrality, power_iter, get_ranking
from utils import init

def _get_parwis_pair(n, items_to_consider, estimates, comp_matrix):
    """
    Selects the next pair of items to compare using the PARWiS strategy.
    This function is copied directly from your provided example.
    """
    if len(items_to_consider) < 2:
        return np.random.choice(list(range(1, n + 1)), 2, replace=False)

    sub_estimates = {i: estimates[i] for i in items_to_consider if i in estimates}
    if not sub_estimates:
        return np.random.choice(list(range(1, n + 1)), 2, replace=False)

    top_item = max(sub_estimates, key=sub_estimates.get)
    cand_pairs, max_array = [], []

    for p_idx in items_to_consider:
        if p_idx != top_item:
            m1 = power_iter(n, comp_matrix, (top_item, p_idx), estimates)
            m2 = power_iter(n, comp_matrix, (p_idx, top_item), estimates)
            if np.isnan(m1) or np.isnan(m2):
                continue
            
            # Add a small epsilon to the denominator to prevent division by zero
            denominator = estimates.get(top_item, 0) + estimates.get(p_idx, 0) + 1e-9
            m = (estimates.get(top_item, 0) * m1 + estimates.get(p_idx, 0) * m2) / denominator
            cand_pairs.append((top_item, p_idx))
            max_array.append(m)

    if not cand_pairs:
        # Fallback if no valid pairs can be evaluated
        return random.sample(items_to_consider, 2) if len(items_to_consider) >= 2 else np.random.choice(list(range(1, n + 1)), 2, replace=False)

    max_val = np.nanmax(max_array)
    if np.isnan(max_val):
        # Fallback if all calculations result in NaN
        return random.sample(items_to_consider, 2) if len(items_to_consider) >= 2 else np.random.choice(list(range(1, n + 1)), 2, replace=False)

    best_indices = np.where(np.nan_to_num(max_array, nan=-np.inf) == max_val)[0]
    if len(best_indices) == 0:
        # Fallback if no best indices are found
        return random.sample(items_to_consider, 2) if len(items_to_consider) >= 2 else np.random.choice(list(range(1, n + 1)), 2, replace=False)

    return cand_pairs[np.random.choice(best_indices)]


if __name__ == "__main__":
    # --- Create Output Directory ---
    output_dir = 'parwis-results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Loop over different values for the true winner's rank 'k'
    for k in [10, 20, 30, 40, 50, 60, 70, 80, 90, 93, 95, 97]:
        # --- Simulation Parameters ---
        N = 100                 # Number of items
        TOTAL_BUDGET = 5 * N      # The maximum number of comparisons for each run
        TOTAL_RUNS = 200        # Number of independent simulations to run for averaging
        RANDOM_SEED = 42        # Seed for reproducibility
        DATASET_NAME = None     # Use None for synthetic data as per the `init` function

        print("\n" + "="*80)
        print(f"--- Starting Pure PARWiS Simulation for k={k} ---")
        print(f"N={N}, Total Budget={TOTAL_BUDGET}, Total Runs={TOTAL_RUNS}")
        print("="*80)

        # 2D array to store recovery results: 1 if winner is found, 0 otherwise.
        # Rows correspond to runs, columns correspond to budget.
        recovery_results = np.zeros((TOTAL_RUNS, TOTAL_BUDGET))
        
        # --- Main Simulation Loop ---
        for i in tqdm(range(TOTAL_RUNS), desc=f"Running Simulations (k={k})"):
            np.random.seed(RANDOM_SEED + i)
            random.seed(RANDOM_SEED + i)
            
            # 1. Initialization for the current run
            scores, true_top = init(n=N, dataset=DATASET_NAME, k=k)
            data, initial_votes, comp_matrix, _, _, estimates = reset(N, scores)
            all_items = list(range(1, N + 1))

            # 2. Iterate for each budget point from 1 to TOTAL_BUDGET
            for budget in range(TOTAL_BUDGET):
                try:
                    p, q = _get_parwis_pair(N, all_items, estimates, comp_matrix)
                except (ValueError, IndexError):
                    p, q = np.random.choice(all_items, 2, replace=False)
                
                # Simulate comparison and update state
                winner_node, loser_node = (p, q) if Vote(scores[p - 1], scores[q - 1]) else (q, p)
                data.append((winner_node, loser_node))
                comp_matrix[winner_node][loser_node] += 1
                
                # Update estimates and get current ranking
                estimates = Rank_Centrality(N, data)
                _, _, current_winner = get_ranking(N, estimates)
                
                # 3. Record recovery status at this budget checkpoint
                if current_winner is not None and (current_winner - 1) == true_top:
                    recovery_results[i, budget] = 1
                else:
                    recovery_results[i, budget] = 0

        print(f"\n--- Aggregating Results for k={k} ---")

        # --- Data Aggregation and CSV Export ---
        
        # Calculate the average recovery rate across all runs for each budget
        average_recovery_rate = np.mean(recovery_results, axis=0)
        
        # Create a DataFrame for easy export
        results_df = pd.DataFrame({
            'Budget': range(1, TOTAL_BUDGET + 1),
            'Average Recovery Rate': average_recovery_rate
        })

        # Set float format to display all decimal places without scientific notation
        pd.options.display.float_format = '{:.15f}'.format

        # Save the DataFrame to a CSV file, ensuring full precision
        output_filename = os.path.join(output_dir, f'N_{N}_k_{k}_parwis_recovery_by_budget.csv')
        results_df.to_csv(output_filename, index=False, float_format='%.15f')

        print("="*50)
        print(f"Simulation for k={k} Complete.")
        print(f"Results have been saved to '{output_filename}'")
        print("="*50)
        print("\nFinal 5 rows of the output data:")
        print(results_df.tail())
        print("="*50)