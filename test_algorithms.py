import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from collections import deque
from itertools import combinations, product
import copy
import random
from joblib import Parallel, delayed
import json
import re



from PARWiS import reset, Vote, Rank_Centrality, power_iter, get_ranking
from utils import init 



# --- CORE ALGORITHM FUNCTIONS (Unchanged) ---
def _get_parwis_pair(n, items_to_consider, estimates, comp_matrix):
    if len(items_to_consider) < 2: return np.random.choice(list(range(1, n + 1)), 2, replace=False)
    sub_estimates = {i: estimates[i] for i in items_to_consider}
    if not sub_estimates: return np.random.choice(list(range(1, n + 1)), 2, replace=False)
    top_item = max(sub_estimates, key=sub_estimates.get)
    cand_pairs, max_array = [], []
    for p_idx in items_to_consider:
        if p_idx != top_item:
            m1 = power_iter(n, comp_matrix, (top_item, p_idx), estimates)
            m2 = power_iter(n, comp_matrix, (p_idx, top_item), estimates)
            if np.isnan(m1) or np.isnan(m2): continue
            m = (estimates[top_item] * m1 + estimates[p_idx] * m2) / (estimates[top_item] + estimates[p_idx] + 1e-9)
            cand_pairs.append((top_item, p_idx)); max_array.append(m)
    if not cand_pairs: return random.sample(items_to_consider, 2)
    max_val = np.nanmax(max_array)
    if np.isnan(max_val): return random.sample(items_to_consider, 2)
    best_indices = np.where(np.nan_to_num(max_array, nan=-np.inf) == max_val)[0]
    if len(best_indices) == 0: return random.sample(items_to_consider, 2)
    return cand_pairs[np.random.choice(best_indices)]

# --- WORKER FUNCTIONS FOR EXPERIMENT STAGES (Unchanged) ---
def _run_parwis_with_checkpoints(n, scores, total_budget, trigger_configs):
    data, initial_votes, comp_matrix, _, _, estimates = reset(n, scores)
    num_comp, checkpoints, triggered_configs = initial_votes, {}, set()
    history_trackers = {c['name']: deque(maxlen=c.get('stability_window_m')) for c in trigger_configs}
    all_items = list(range(1, n + 1))
    while num_comp < total_budget:
        try: p, q = _get_parwis_pair(n, all_items, estimates, comp_matrix)
        except (ValueError, IndexError): p, q = np.random.choice(all_items, 2, replace=False)
        winner, loser = (p, q) if Vote(scores[p - 1], scores[q - 1]) else (q, p)
        data.append((winner, loser)); comp_matrix[winner][loser] += 1; num_comp += 1
        estimates = Rank_Centrality(n, data)
        _, _, current_winner = get_ranking(n, estimates)
        for config in trigger_configs:
            name = config['name']
            if name in triggered_configs: continue
            history = history_trackers[name]
            history.append(current_winner)
            if len(history) == history.maxlen and len(set(history)) == 1:
                checkpoints[name] = {"data": copy.deepcopy(data), "estimates": estimates.copy(), "comp_matrix": copy.deepcopy(comp_matrix), "budget_used": num_comp}
                triggered_configs.add(name)
    final_state = {"data": data, "estimates": estimates, "comp_matrix": comp_matrix, "budget_used": total_budget}
    for config in trigger_configs:
        if config['name'] not in checkpoints: checkpoints[config['name']] = final_state
    return checkpoints, final_state

def _run_phase2(n, scores, total_budget, phase1_state, config):
    data, estimates, comp_matrix = list(phase1_state["data"]), phase1_state["estimates"].copy(), phase1_state["comp_matrix"].copy()
    remaining_budget = total_budget - phase1_state["budget_used"]
    phase2_strategy = config.get('phase2_strategy')
    if remaining_budget <= 0 or not phase2_strategy: return phase1_state['data'], 0
    k_used, items_to_consider = 0, list(range(1, n + 1))
    is_general_probabilistic = config.get('is_general_probabilistic', False)
    if is_general_probabilistic:
        for _ in range(remaining_budget):
            if np.random.rand() < config.get('epsilon', 0.0):
                scores_samp = estimates[1:].copy(); scores_samp[scores_samp < 0] = 0; total_score = np.sum(scores_samp)
                # Check if we have enough non-zero entries for sampling 2 items without replacement
                non_zero_count = np.count_nonzero(scores_samp)
                if total_score > 0 and non_zero_count >= 2:
                    p, q = np.random.choice(np.arange(1, n + 1), size=2, replace=False, p=scores_samp / total_score)
                else:
                    p, q = random.sample(items_to_consider, 2)
            else: p, q = _get_parwis_pair(n, items_to_consider, estimates, comp_matrix)
            winner, loser = (p, q) if Vote(scores[p-1], scores[q-1]) else (q, p)
            data.append((winner, loser)); comp_matrix[winner][loser] += 1; estimates = Rank_Centrality(n, data)
        return data, k_used
    if config.get('phase2_focus_on_contenders', False):
        k = int(np.sqrt(2 * remaining_budget)) if remaining_budget > 0 else 0; k_used = k
        if config.get('candidate_selection_method') == 'probabilistic':
            scores_samp = estimates[1:].copy(); scores_samp[scores_samp < 0] = 0; total_score = np.sum(scores_samp)
            # Check if we have enough non-zero entries for sampling without replacement
            non_zero_count = np.count_nonzero(scores_samp)
            if total_score > 0 and non_zero_count >= min(k, n):
                items_to_consider = np.random.choice(np.arange(1, n + 1), size=min(k, n), replace=False, p=scores_samp / total_score).tolist()
            else:
                items_to_consider = random.sample(range(1, n + 1), min(k, n))
        else: ranking, _, _ = get_ranking(n, estimates); items_to_consider = ranking[:k]
    if len(items_to_consider) < 2: return data, k_used
    for _ in range(remaining_budget):
        if phase2_strategy == 'epsilon_greedy' and np.random.rand() < config.get('epsilon', 0.0): p, q = random.sample(items_to_consider, 2)
        else: p, q = _get_parwis_pair(n, items_to_consider, estimates, comp_matrix)
        winner, loser = (p, q) if Vote(scores[p-1], scores[q-1]) else (q, p)
        data.append((winner, loser)); comp_matrix[winner][loser] += 1; estimates = Rank_Centrality(n, data)
    return data, k_used

def run_phase2_worker(args):
    config, trial_checkpoints, n, scores, true_top, total_budget = args
    name = config['name']
    p1_state = trial_checkpoints.get(name)
    if p1_state is None: return name, None

    # --- Phase 1 calculations ---
    p1_estimates = p1_state['estimates']
    p1_ranking_list, _, p1_winner = get_ranking(n, p1_estimates)
    is_p1_winner_correct = (p1_winner is not None) and ((p1_winner - 1) == true_top)
    p1_winner_score, p1_second_score = 0.0, 0.0
    if p1_ranking_list and len(p1_ranking_list) > 1:
        p1_winner_score, p1_second_score = p1_estimates[p1_ranking_list[-1]], p1_estimates[p1_ranking_list[-2]]
    # NEW: Get the estimated score of the true winner in Phase 1
    p1_true_winner_score = p1_estimates[true_top + 1]

    # --- Phase 2 execution ---
    final_data, k_used = _run_phase2(n, scores, total_budget, p1_state, config)
    final_estimates = Rank_Centrality(n, final_data)
    final_ranking_list, _, final_winner = get_ranking(n, final_estimates)
    is_final_winner_correct = (final_winner is not None) and ((final_winner - 1) == true_top)
    try: rank_of_true = final_ranking_list.index(true_top + 1) + 1
    except (ValueError, TypeError): rank_of_true = n
    p2_winner_score, p2_second_score = 0.0, 0.0
    if final_ranking_list and len(final_ranking_list) > 1:
        p2_winner_score, p2_second_score = final_estimates[final_ranking_list[-1]], final_estimates[final_ranking_list[-2]]
    # NEW: Get the estimated score of the true winner in Phase 2
    p2_true_winner_score = final_estimates[true_top + 1]
    
    # MODIFIED: Return dictionary includes new scores
    return name, {
        'p1_correct_winner_count': 1 if is_p1_winner_correct else 0, 'avg_k_sum': k_used,
        'p2_winner_change_count': 1 if p1_winner != final_winner else 0,
        'p2_fix_count': 1 if not is_p1_winner_correct and is_final_winner_correct else 0,
        'p2_error_count': 1 if is_p1_winner_correct and not is_final_winner_correct else 0,
        'recovery_count': 1 if is_final_winner_correct else 0,
        'phase1_budget_sum': p1_state['budget_used'], 'reported_rank_sum': rank_of_true,
        'p1_winner_score_sum': p1_winner_score, 'p1_second_score_sum': p1_second_score,
        'p2_winner_score_sum': p2_winner_score, 'p2_second_score_sum': p2_second_score,
        'p1_true_winner_score_sum': p1_true_winner_score, # NEW
        'p2_true_winner_score_sum': p2_true_winner_score, # NEW
    }

def parse_config_to_json_parts(config):
    name, family = config['name'], "Unknown"
    if "Pure PARWiS" in name: family = "PARWiS Baselines"
    elif "PARWiS-P2" in name: family = "PARWiS Baselines"
    elif "Gen-Prob E-Greedy" in name: family = "Gen-Prob E-Greedy"
    elif "Top-k" in name: family = "E-Greedy (Top-k)"
    elif "Prob-k" in name: family = "E-Greedy (Prob-k)"
    m_match, e_match, k_type = re.search(r'm=(\d+)', name), re.search(r'e=([\d.]+)', name), None
    if "Top-k" in name: k_type = "Top-k"
    elif "Prob-k" in name: k_type = "Prob-k"
    return family, {"m": int(m_match.group(1)) if m_match else None, "e": float(e_match.group(1)) if e_match else None, "k_type": k_type}

if __name__ == "__main__":

    for N in [100]:
        for k in [97]:
       
            DATASET_NAME, EXPERIMENTS, ITERATIONS, TOTAL_BUDGET_C, RANDOM_SEED = None, 20, 10, 5, 42
            total_budget, total_runs = TOTAL_BUDGET_C * N, EXPERIMENTS * ITERATIONS
            scores, true_top = init(n=N, dataset=DATASET_NAME, k=k)

            baseline_config = {'name': "Pure PARWiS (Baseline)", 'trigger_strategy': 'full_budget'}
            parwis_p2_configs, egreedy_focused_configs, egreedy_general_configs = [], [], []
            for m in [N // 2, int(N * 0.75)]:
                parwis_p2_configs.append({'name': f"PARWiS-P2 (m={m})", 'trigger_strategy': 'winner', 'stability_window_m': m, 'phase2_strategy': 'parwis_phase2', 'phase2_focus_on_contenders': False})
                for e in [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
                    egreedy_focused_configs.append({'name': f"E-Greedy (m={m}, e={e:.1f}, Top-k)", 'trigger_strategy': 'winner', 'stability_window_m': m, 'phase2_strategy': 'epsilon_greedy', 'epsilon': e, 'candidate_selection_method': 'top_k', 'phase2_focus_on_contenders': True})
                    egreedy_focused_configs.append({'name': f"E-Greedy (m={m}, e={e:.1f}, Prob-k)", 'trigger_strategy': 'winner', 'stability_window_m': m, 'phase2_strategy': 'epsilon_greedy', 'epsilon': e, 'candidate_selection_method': 'probabilistic', 'phase2_focus_on_contenders': True})
                    egreedy_general_configs.append({'name': f"Gen-Prob E-Greedy (m={m}, e={e:.1f})", 'trigger_strategy': 'winner', 'stability_window_m': m, 'phase2_strategy': 'epsilon_greedy', 'epsilon': e, 'is_general_probabilistic': True})
            all_trigger_configs, all_parallel_configs = parwis_p2_configs + egreedy_focused_configs + egreedy_general_configs, egreedy_focused_configs + egreedy_general_configs
            
            print("--- Stage 1: Generating PARWiS Checkpoints ---")
            all_trials_checkpoints, baseline_final_states = [], []
            for i in tqdm(range(total_runs), desc="Generating Checkpoints"):
                np.random.seed(RANDOM_SEED + i); random.seed(RANDOM_SEED + i)
                checkpoints, final_state = _run_parwis_with_checkpoints(N, scores, total_budget, all_trigger_configs)
                all_trials_checkpoints.append(checkpoints); baseline_final_states.append(final_state)
            
            print(f"\n--- Stage 2: Processing {len(all_parallel_configs)} Two-Phase Configurations in Parallel ---")
            args_list = [(config, all_trials_checkpoints[i], N, scores, true_top, total_budget) for i in range(total_runs) for config in all_parallel_configs]
            results_list = Parallel(n_jobs=-1)(delayed(run_phase2_worker)(args) for args in tqdm(args_list, desc="Processing Phase 2"))

            all_display_configs = [baseline_config] + parwis_p2_configs + all_parallel_configs
            # MODIFIED: Added new stat keys
            stat_keys = ['recovery_count', 'reported_rank_sum', 'phase1_budget_sum', 'avg_k_sum',
                        'p1_correct_winner_count', 'p2_winner_change_count', 'p2_fix_count', 'p2_error_count',
                        'p1_winner_score_sum', 'p1_second_score_sum', 'p2_winner_score_sum', 'p2_second_score_sum',
                        'p1_true_winner_score_sum', 'p2_true_winner_score_sum']
            agg_results = {c['name']: {k: 0 for k in stat_keys} for c in all_display_configs}
            for name, res in results_list:
                if res:
                    for key, value in res.items(): agg_results[name][key] += value

            for i in range(total_runs):
                final_state, checkpoints = baseline_final_states[i], all_trials_checkpoints[i]
                final_estimates = final_state['estimates']
                final_ranking, _, final_winner = get_ranking(N, final_estimates)
                is_final_winner_correct = (final_winner is not None) and ((final_winner - 1) == true_top)
                try: final_rank_of_true = final_ranking.index(true_top + 1) + 1
                except (ValueError, TypeError): final_rank_of_true = N
                p2_winner_score, p2_second_score = (final_estimates[final_ranking[-1]], final_estimates[final_ranking[-2]]) if final_ranking and len(final_ranking) > 1 else (0.0, 0.0)
                # NEW: Calculate true winner's score in Phase 2 once per trial
                p2_true_winner_score = final_estimates[true_top + 1]

                for config in [baseline_config] + parwis_p2_configs:
                    name = config['name']
                    agg_results[name]['recovery_count'] += 1 if is_final_winner_correct else 0
                    agg_results[name]['reported_rank_sum'] += final_rank_of_true
                    agg_results[name]['p2_winner_score_sum'] += p2_winner_score
                    agg_results[name]['p2_second_score_sum'] += p2_second_score
                    agg_results[name]['p2_true_winner_score_sum'] += p2_true_winner_score # NEW: Add to P2 sum
                    
                    if config['name'] == baseline_config['name']:
                        agg_results[name]['phase1_budget_sum'] += final_state['budget_used']
                        agg_results[name]['p1_correct_winner_count'] += 1 if is_final_winner_correct else 0
                        agg_results[name]['p1_winner_score_sum'] += p2_winner_score
                        agg_results[name]['p1_second_score_sum'] += p2_second_score
                        agg_results[name]['p1_true_winner_score_sum'] += p2_true_winner_score # NEW: Add to P1 sum for baseline
                    else:
                        p1_state = checkpoints.get(name)
                        if not p1_state: continue
                        p1_estimates = p1_state['estimates']
                        p1_ranking, _, p1_winner = get_ranking(N, p1_estimates)
                        is_p1_winner_correct = (p1_winner is not None) and ((p1_winner - 1) == true_top)
                        p1_winner_score, p1_second_score = (p1_estimates[p1_ranking[-1]], p1_estimates[p1_ranking[-2]]) if p1_ranking and len(p1_ranking) > 1 else (0.0, 0.0)
                        # NEW: Calculate and add P1 true winner score
                        p1_true_winner_score = p1_estimates[true_top + 1]
                        agg_results[name]['p1_true_winner_score_sum'] += p1_true_winner_score
                        
                        agg_results[name]['p1_winner_score_sum'] += p1_winner_score
                        agg_results[name]['p1_second_score_sum'] += p1_second_score
                        agg_results[name]['phase1_budget_sum'] += p1_state['budget_used']
                        agg_results[name]['p1_correct_winner_count'] += 1 if is_p1_winner_correct else 0
                        agg_results[name]['p2_winner_change_count'] += 1 if p1_winner != final_winner else 0
                        agg_results[name]['p2_fix_count'] += 1 if not is_p1_winner_correct and is_final_winner_correct else 0
                        agg_results[name]['p2_error_count'] += 1 if is_p1_winner_correct and not is_final_winner_correct else 0

            final_df_data, json_output_data = [], []
            for config in all_display_configs:
                name, res = config['name'], agg_results[name]
                metrics = {
                    "recovery_pct": (res['recovery_count'] / total_runs) * 100, "avg_rank": res['reported_rank_sum'] / total_runs,
                    "p1_budget": res['phase1_budget_sum'] / total_runs, "p1_correct_pct": (res['p1_correct_winner_count'] / total_runs) * 100,
                    "p1_winner_score": res['p1_winner_score_sum'] / total_runs, "p1_2nd_score": res['p1_second_score_sum'] / total_runs,
                    "p2_winner_score": res['p2_winner_score_sum'] / total_runs, "p2_2nd_score": res['p2_second_score_sum'] / total_runs,
                    "avg_k": res['avg_k_sum'] / total_runs,
                    "p1_true_winner_score": res['p1_true_winner_score_sum'] / total_runs, # NEW
                    "p2_true_winner_score": res['p2_true_winner_score_sum'] / total_runs, # NEW
                }
                family, params = parse_config_to_json_parts(config)
                json_output_data.append({"algorithm_name": name, "algorithm_family": family, "parameters": params, "metrics": metrics})
                df_row = {"Algorithm": name}; df_row.update(metrics)
                df_row.update({"P1 Avg difference": metrics["p1_winner_score"] - metrics["p1_2nd_score"], "P2 Avg difference": metrics["p2_winner_score"] - metrics["p2_2nd_score"],
                            "P2 Changes %": (res['p2_winner_change_count'] / total_runs) * 100, "P2 Fixes %": (res['p2_fix_count'] / total_runs) * 100,
                            "P2 Errors %": (res['p2_error_count'] / total_runs) * 100})
                final_df_data.append(df_row)

        
            
            results_df = pd.DataFrame(final_df_data)
            pd.set_option('display.max_columns', None); pd.set_option('display.width', 220)
            for col in ['recovery_pct', 'p1_correct_pct']: results_df[col] = results_df[col].map('{:.2f}%'.format)
            results_df['p1_budget'] = results_df['p1_budget'].map('{:,.0f}'.format)
            results_df['avg_rank'] = results_df['avg_rank'].map('{:.2f}'.format)
            # MODIFIED: Added new columns to the formatting list
            score_cols = ['p1_winner_score', 'p1_2nd_score', 'p2_winner_score', 'p2_2nd_score', 'p1_true_winner_score', 'p2_true_winner_score']
            for col in score_cols: results_df[col] = results_df[col].map('{:.4f}'.format)
            results_df['avg_k'] = results_df['avg_k'].map('{:.1f}'.format)

            print("\n" + "="*180); print(f"FINAL SIMULATION RESULTS (N={N}, Budget={total_budget}, Dataset: {DATASET_NAME})"); print("="*180)
            results_df['Recovery %_sort'] = results_df['recovery_pct'].str.rstrip('%').astype(float)
            results_df = results_df.sort_values(by="Recovery %_sort", ascending=False).drop(columns=['Recovery %_sort']).reset_index(drop=True)

            results_df.to_csv(f'results/N_{N}_k_{k}_simulation_results.csv', index=False)
            
            print(results_df.to_string(index=False)); print("="*180)