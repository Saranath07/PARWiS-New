
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from collections import deque
import random
import copy
import re

# --- DEPENDENCIES ---
# Make sure the PARWiS.py and utils.py files from your original code
# are in the same directory as this script.
try:
    from PARWiS import reset, Vote, Rank_Centrality, power_iter, get_ranking
    from utils import init
except ImportError:
    print("Error: Make sure 'PARWiS.py' and 'utils.py' are in the same directory.")
    sys.exit(1)


def _get_parwis_pair(n, items_to_consider, estimates, comp_matrix):
    """
    Selects the next pair of items to compare using the PARWiS strategy.
    This function is copied from your provided example.
    """
    # Ensure items_to_consider is a list and has at least 2 elements
    items_list = list(items_to_consider)
    if len(items_list) < 2:
        return np.random.choice(list(range(1, n + 1)), 2, replace=False)

    sub_estimates = {i: estimates[i] for i in items_list if i in estimates}
    if not sub_estimates:
        return np.random.choice(list(range(1, n + 1)), 2, replace=False)

    top_item = max(sub_estimates, key=sub_estimates.get)
    cand_pairs, max_array = [], []

    for p_idx in items_list:
        if p_idx != top_item:
            m1 = power_iter(n, comp_matrix, (top_item, p_idx), estimates)
            m2 = power_iter(n, comp_matrix, (p_idx, top_item), estimates)
            if np.isnan(m1) or np.isnan(m2):
                continue
            
            denominator = estimates.get(top_item, 0) + estimates.get(p_idx, 0) + 1e-9
            m = (estimates.get(top_item, 0) * m1 + estimates.get(p_idx, 0) * m2) / denominator
            cand_pairs.append((top_item, p_idx))
            max_array.append(m)

    if not cand_pairs:
        return random.sample(items_list, 2)

    max_val = np.nanmax(max_array)
    if np.isnan(max_val):
        return random.sample(items_list, 2)

    best_indices = np.where(np.nan_to_num(max_array, nan=-np.inf) == max_val)[0]
    if len(best_indices) == 0:
        return random.sample(items_list, 2)

    return cand_pairs[np.random.choice(best_indices)]


if __name__ == "__main__":
    # --- Create Output Directory ---
    output_dir = 'all_algos_recovery_by_budget'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # --- Simulation Parameters ---
    N = 100                 # Number of items
    TOTAL_BUDGET = 5 * N    # The maximum number of comparisons for each run
    TOTAL_RUNS = 200        # Number of independent simulations to run for averaging
    RANDOM_SEED = 42        # Seed for reproducibility
    DATASET_NAME = None     # Use None for synthetic data via the `init` function

    # --- Algorithm Configurations ---
    # Define all algorithm variants to be tested, same as in your first script.
    baseline_config = {'name': "Pure PARWiS", 'trigger_strategy': 'full_budget'}
    
    parwis_p2_configs = []
    egreedy_focused_configs = []
    egreedy_general_configs = []
    
    for m in [N // 2, int(N * 0.75)]:
        parwis_p2_configs.append({'name': f"PARWiS-P2 (m={m})", 'trigger_strategy': 'winner', 'stability_window_m': m, 'phase2_strategy': 'parwis_phase2', 'phase2_focus_on_contenders': False})
        for e in [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
            egreedy_focused_configs.append({'name': f"E-Greedy (m={m}, e={e:.1f}, Top-k)", 'trigger_strategy': 'winner', 'stability_window_m': m, 'phase2_strategy': 'epsilon_greedy', 'epsilon': e, 'candidate_selection_method': 'top_k', 'phase2_focus_on_contenders': True})
            egreedy_focused_configs.append({'name': f"E-Greedy (m={m}, e={e:.1f}, Prob-k)", 'trigger_strategy': 'winner', 'stability_window_m': m, 'phase2_strategy': 'epsilon_greedy', 'epsilon': e, 'candidate_selection_method': 'probabilistic', 'phase2_focus_on_contenders': True})
            egreedy_general_configs.append({'name': f"Gen-Prob E-Greedy (m={m}, e={e:.1f})", 'trigger_strategy': 'winner', 'stability_window_m': m, 'phase2_strategy': 'epsilon_greedy', 'epsilon': e, 'is_general_probabilistic': True})

    all_configs = [baseline_config] + parwis_p2_configs + egreedy_focused_configs + egreedy_general_configs

    # --- Main Experiment Loop ---
    # Loop over different values for the true winner's initial rank 'k'
    for k in [10, 20, 30, 40, 50, 60, 70, 80, 90, 93, 95, 97]:
        print("\n" + "="*80)
        print(f"--- Starting All Algorithm Simulations for k={k} ---")
        print("="*80)

        # Loop over each algorithm configuration
        for config in all_configs:
            config_name = config['name']
            print(f"  -> Processing Algorithm: {config_name}")

            # 2D array to store recovery results for this specific algorithm config
            recovery_results = np.zeros((TOTAL_RUNS, TOTAL_BUDGET))

            # Main simulation loop for the current algorithm
            for i in tqdm(range(TOTAL_RUNS), desc=f"    Runs (k={k})", leave=False):
                # Set seeds for reproducibility
                np.random.seed(RANDOM_SEED + i)
                random.seed(RANDOM_SEED + i)

                # 1. Initialization for the current run
                scores, true_top = init(n=N, dataset=DATASET_NAME, k=k)
                data, _, comp_matrix, _, _, estimates = reset(N, scores)
                all_items = list(range(1, N + 1))

                # 2. State variables for handling two-phase logic
                phase2_started = False
                items_to_consider_p2 = all_items.copy()
                history = deque(maxlen=config.get('stability_window_m')) if config.get('trigger_strategy') == 'winner' else None

                # 3. Iterate for each budget point from 1 to TOTAL_BUDGET
                for budget in range(TOTAL_BUDGET):
                    p, q = None, None

                    # --- PHASE TRANSITION LOGIC ---
                    # Check if we should transition from Phase 1 to Phase 2
                    if not phase2_started and history is not None:
                        _, _, winner_for_history = get_ranking(N, estimates)
                        history.append(winner_for_history)
                        
                        # Trigger condition: winner has been stable for 'm' comparisons
                        if len(history) == history.maxlen and len(set(history)) == 1:
                            phase2_started = True
                            
                            # Determine the item set for Phase 2 based on the config
                            if config.get('phase2_focus_on_contenders', False):
                                remaining_budget = TOTAL_BUDGET - (budget + 1)
                                k_p2 = int(np.sqrt(2 * remaining_budget)) if remaining_budget > 0 else 0
                                k_p2 = min(k_p2, N) # Cannot be more than N

                                if config.get('candidate_selection_method') == 'probabilistic':
                                    scores_samp = estimates[1:].copy(); scores_samp[scores_samp < 0] = 0
                                    total_score = np.sum(scores_samp)
                                    non_zero_count = np.count_nonzero(scores_samp)
                                    if total_score > 0 and non_zero_count >= k_p2:
                                        items_to_consider_p2 = np.random.choice(np.arange(1, N + 1), size=k_p2, replace=False, p=scores_samp / total_score).tolist()
                                    else:
                                        items_to_consider_p2 = random.sample(range(1, N + 1), k_p2) if k_p2 > 0 else []
                                else: # 'top_k'
                                    ranking, _, _ = get_ranking(N, estimates)
                                    items_to_consider_p2 = ranking[:k_p2]
                            else: # PARWiS-P2 or other non-focusing strategies
                                items_to_consider_p2 = all_items.copy()
                    
                    # --- PAIR SELECTION LOGIC ---
                    current_items = items_to_consider_p2 if phase2_started and config.get('phase2_focus_on_contenders') else all_items
                    if len(current_items) < 2: current_items = all_items
                    
                    use_epsilon_logic = phase2_started and config.get('phase2_strategy') == 'epsilon_greedy' and np.random.rand() < config.get('epsilon', 0.0)

                    if use_epsilon_logic:
                        # Epsilon-greedy (exploration) step
                        if config.get('is_general_probabilistic', False):
                            # Special case for Gen-Prob: sample from all items based on score
                            scores_samp = estimates[1:].copy(); scores_samp[scores_samp < 0] = 0
                            total_score = np.sum(scores_samp)
                            if total_score > 0 and np.count_nonzero(scores_samp) >= 2:
                                p, q = np.random.choice(np.arange(1, N + 1), size=2, replace=False, p=scores_samp / total_score)
                            else:
                                p, q = random.sample(all_items, 2)
                        else:
                            # Standard epsilon-greedy: random pair from the current item set
                            p, q = random.sample(current_items, 2)
                    else:
                        # PARWiS / Exploitation step
                        p, q = _get_parwis_pair(N, current_items, estimates, comp_matrix)

                    # --- SIMULATE AND UPDATE ---
                    winner, loser = (p, q) if Vote(scores[p - 1], scores[q - 1]) else (q, p)
                    data.append((winner, loser))
                    comp_matrix[winner][loser] += 1
                    estimates = Rank_Centrality(N, data)
                    
                    # 4. Record recovery status at this budget checkpoint
                    _, _, current_winner = get_ranking(N, estimates)
                    if current_winner is not None and (current_winner - 1) == true_top:
                        recovery_results[i, budget] = 1
                    else:
                        recovery_results[i, budget] = 0

            # --- AGGREGATE AND SAVE RESULTS FOR THE CURRENT ALGORITHM ---
            average_recovery_rate = np.mean(recovery_results, axis=0)
            results_df = pd.DataFrame({
                'Budget': range(1, TOTAL_BUDGET + 1),
                'Average Recovery Rate': average_recovery_rate
            })

            # Sanitize the config name to create a valid filename
            safe_config_name = re.sub(r'[^\w\-.=() ]', '_', config_name).replace(' ', '_')
            output_filename = os.path.join(output_dir, f'N{N}_k{k}_budget{TOTAL_BUDGET}_{safe_config_name}.csv')
            
            # Save the DataFrame to a CSV file with full precision
            results_df.to_csv(output_filename, index=False, float_format='%.15f')
            
            print(f"    -> Results for '{config_name}' saved to '{output_filename}'")
            
    print("\n" + "="*80)
    print("All simulations complete.")
    print(f"All CSV files have been saved in the '{output_dir}' directory.")
    print("="*80)

