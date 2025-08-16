"""
Two-Phase PARWiS Algorithm Comparison

Phase 1: Original PARWiS using power_iter with stability window m
Phase 2: Top-k disruption score calculation for KC2 pairs (k = sqrt(N))
"""

import numpy as np
import pandas as pd
import random
import copy
from tqdm import tqdm
from itertools import combinations
import math
from collections import deque

from PARWiS import (reset, Vote, Rank_Centrality, power_iter, get_ranking, 
                    pick_a_pair, IMC_update)
from utils import init

def pick_topk_disruption_pair(n, comp_matrix, estimates, chain, A_hash, ranking, ranks, top, k=None):
    """
    Phase 2: Calculate disruption scores for top-k items and choose most disruptive pair
    from all KC2 combinations (k = sqrt(N))
    """
    if k is None:
        k = int(math.sqrt(n))
    
    # Get top k items from current ranking
    top_k_items = ranking[-k:] if len(ranking) >= k else ranking
    
    if len(top_k_items) < 2:
        # Fallback to random pair
        return np.random.choice(list(range(1, n + 1)), 2, replace=False)
    
    cand_pairs = []
    max_array = []
    
    # Calculate disruption scores for all KC2 combinations
    for pair in combinations(top_k_items, 2):
        i, j = pair
        
        # Calculate disruption scores using power iteration (same as original PARWiS)
        m1 = power_iter(n, comp_matrix, (i, j), estimates)
        m2 = power_iter(n, comp_matrix, (j, i), estimates)
        
        if np.isnan(m1) or np.isnan(m2):
            continue
            
        # Weighted average (same method as original PARWiS)
        m = (estimates[i] * m1 + estimates[j] * m2) / (estimates[i] + estimates[j] + 1e-9)
        cand_pairs.append((i, j))
        max_array.append(m)

    
    # Choose pair with maximum disruption score
    candidates = np.where(max_array == np.max(max_array))[0]
    idx = np.random.choice(candidates)
    return cand_pairs[idx]

def run_parwis_phase1(n, scores, total_budget, stability_window_m):
    """
    Phase 1: Run original PARWiS until winner is stable for m rounds
    Returns checkpoint when stability is achieved
    """
    data, initial, comp_matrix, chain, A_hash, estimates = reset(n, scores)
    num_comparisons = initial
    
    # Track winner history for stability window using deque like test_algorithms.py
    winner_history = deque(maxlen=stability_window_m)
    phase1_checkpoint = None
    
    while num_comparisons < total_budget:
        ranking, ranks, top = get_ranking(n, estimates)
        
        # Track winner for stability - same logic as test_algorithms.py
        winner_history.append(top)
        
        # Check if we have stable winner for m rounds - same condition as test_algorithms.py
        if (len(winner_history) == winner_history.maxlen and
            len(set(winner_history)) == 1 and
            phase1_checkpoint is None):
            
            phase1_checkpoint = {
                'data': copy.deepcopy(data),
                'estimates': estimates.copy(),
                'comp_matrix': copy.deepcopy(comp_matrix),
                'chain': copy.deepcopy(chain),
                'A_hash': copy.deepcopy(A_hash),
                'budget_used': num_comparisons
            }
            print(f"Phase 1 stability achieved at budget {num_comparisons}!")
            break  # Exit Phase 1 once stability is achieved
        
        # Use original PARWiS pair selection with fallback
        try:
            p, q = pick_a_pair(n, comp_matrix, estimates, chain, A_hash, ranking, ranks, top, compute="power")
        except (ValueError, IndexError):
            # Fallback to random pair selection
            p, q = np.random.choice(list(range(1, n + 1)), 2, replace=False)
        
        # Perform comparison
        if Vote(scores[p-1], scores[q-1]):
            winner, loser = p, q
        else:
            winner, loser = q, p
            
        data.append((winner, loser))
        comp_matrix[winner][loser] += 1
        num_comparisons += 1
        
        # Update estimates using IMC
        m, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (winner, loser), True, estimates, chain, A_hash, estimates)
        m, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (loser, winner), False, pi, new_chain, IP_hash, estimates)
        estimates = pi
        A_hash = IP_hash
        chain = new_chain
    
    # If no checkpoint was created, use final state
    if phase1_checkpoint is None:
        phase1_checkpoint = {
            'data': data,
            'estimates': estimates,
            'comp_matrix': comp_matrix,
            'chain': chain,
            'A_hash': A_hash,
            'budget_used': total_budget
        }
    
    return phase1_checkpoint, {
        'data': data,
        'estimates': estimates,
        'budget_used': total_budget
    }

def run_two_phase_algorithm(n, scores, total_budget, phase1_checkpoint):
    """
    Phase 2: Continue from checkpoint using top-k KC2 disruption pairs
    """
    data = list(phase1_checkpoint['data'])
    estimates = phase1_checkpoint['estimates'].copy()
    comp_matrix = phase1_checkpoint['comp_matrix'].copy()
    chain = phase1_checkpoint['chain'].copy()
    A_hash = phase1_checkpoint['A_hash'].copy()
    
    remaining_budget = total_budget - phase1_checkpoint['budget_used']
    num_comparisons = phase1_checkpoint['budget_used']
    
    # k = int(math.sqrt(n))  # k = sqrt(N)
    k = 40
    
    # Phase 2: Use top-k KC2 disruption pairs
    while num_comparisons < total_budget and remaining_budget > 0:
        ranking, ranks, top = get_ranking(n, estimates)
        
        # Use top-k KC2 pair selection with fallback
        try:
            p, q = pick_topk_disruption_pair(n, comp_matrix, estimates, chain, A_hash, ranking, ranks, top, k)
        except (ValueError, IndexError):
            # Fallback to random pair selection
            p, q = np.random.choice(list(range(1, n + 1)), 2, replace=False)
        
        # Perform comparison
        if Vote(scores[p-1], scores[q-1]):
            winner, loser = p, q
        else:
            winner, loser = q, p
            
        data.append((winner, loser))
        comp_matrix[winner][loser] += 1
        num_comparisons += 1
        remaining_budget -= 1
        
        # Update estimates using IMC
        m, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (winner, loser), True, estimates, chain, A_hash, estimates)
        m, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (loser, winner), False, pi, new_chain, IP_hash, estimates)
        estimates = pi
        A_hash = IP_hash
        chain = new_chain
    
    return {
        'data': data,
        'estimates': estimates,
        'budget_used': num_comparisons
    }
def run_original_parwis_full(n, scores, total_budget):
    """
    Run original PARWiS for full budget using the same approach as two-phase phase1
    """
    data, initial, comp_matrix, chain, A_hash, estimates = reset(n, scores)
    num_comparisons = initial
    
    while num_comparisons < total_budget:
        ranking, ranks, top = get_ranking(n, estimates)
        
        # Use original PARWiS pair selection with fallback
        try:
            p, q = pick_a_pair(n, comp_matrix, estimates, chain, A_hash, ranking, ranks, top, compute="power")
        except (ValueError, IndexError):
            # Fallback to random pair selection
            p, q = np.random.choice(list(range(1, n + 1)), 2, replace=False)
        
        # Perform comparison
        if Vote(scores[p-1], scores[q-1]):
            winner, loser = p, q
        else:
            winner, loser = q, p
            
        data.append((winner, loser))
        comp_matrix[winner][loser] += 1
        num_comparisons += 1
        
        # Update estimates using IMC
        m, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (winner, loser), True, estimates, chain, A_hash, estimates)
        m, pi, new_chain, IP_hash = IMC_update(n, comp_matrix, (loser, winner), False, pi, new_chain, IP_hash, estimates)
        estimates = pi
        A_hash = IP_hash
        chain = new_chain
    
    return {
        'data': data,
        'estimates': estimates,
        'budget_used': total_budget
    }
    
    

def run_comparison_trial(n, scores, true_top, total_budget, stability_window_m, seed):
    """Run single trial comparing original PARWiS vs two-phase algorithm"""
    
    # Set seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    # Run original PARWiS for full budget - use the same approach as main PARWiS.py
    original_result = run_original_parwis_full(n, scores, total_budget)
    
    # Reset seed
    np.random.seed(seed)
    random.seed(seed)
    
    # Run Phase 1 with checkpoint
    phase1_checkpoint, _ = run_parwis_phase1(n, scores, total_budget, stability_window_m)
    
    # Reset seed again for Phase 2
    np.random.seed(seed)
    random.seed(seed)
    
    # Continue with Phase 2
    two_phase_result = run_two_phase_algorithm(n, scores, total_budget, phase1_checkpoint)
    
    # Calculate metrics using the same approach as main PARWiS.py
    # Original PARWiS
    orig_ranking, orig_ranks, orig_winner = get_ranking(n, original_result['estimates'])
    orig_recovery = (orig_winner - 1) == true_top
    orig_rank_of_true = orig_ranks[true_top]
    
    # Two-phase algorithm
    two_phase_ranking, two_phase_ranks, two_phase_winner = get_ranking(n, two_phase_result['estimates'])
    two_phase_recovery = (two_phase_winner - 1) == true_top
    two_phase_rank_of_true = two_phase_ranks[true_top]
    
    # Phase 1 metrics
    phase1_ranking, phase1_ranks, phase1_winner = get_ranking(n, phase1_checkpoint['estimates'])
    phase1_recovery = (phase1_winner - 1) == true_top
    
    return {
        'original_recovery': orig_recovery,
        'original_rank_of_true': orig_rank_of_true,
        'two_phase_recovery': two_phase_recovery,
        'two_phase_rank_of_true': two_phase_rank_of_true,
        'phase1_recovery': phase1_recovery,
        'phase1_budget_used': phase1_checkpoint['budget_used'],
        'k_used': 40
    }
def run_full_comparison(n=100, experiments=10, iterations=10, budget_multiplier=5, stability_window_m=50,
                       precomputed=True, dataset=None, k=75):
    """Run full comparison experiment using the same approach as main PARWiS.py"""
    
    total_budget = n * budget_multiplier
    
    print(f"PARWiS Two-Phase Algorithm Comparison")
    print(f"N = {n}, Total Budget = {total_budget}")
    print(f"Stability Window m = {stability_window_m}")
    print(f"Phase 2 k = sqrt(N) = {40}")
    print(f"Experiments = {experiments}, Iterations = {iterations}")
    print()
    
    # Initialize scores using the same approach as main PARWiS.py
    scores, true_top = init(n, precomputed=precomputed, dataset=dataset, k=k)
    
    # Run both original PARWiS and two-phase algorithm
    print("Running Original PARWiS and Two-Phase Algorithm...")
    
    # Results storage
    results = []
    total_runs = experiments * iterations
    
    for exp in tqdm(range(experiments), desc="Experiments"):
        for itr in tqdm(range(iterations), desc="Iterations", leave=False):
            seed = exp * iterations + itr + 42  # Add offset for reproducibility
            
            trial_result = run_comparison_trial(n, scores, true_top, total_budget, stability_window_m, seed)
            results.append(trial_result)
    
    # Calculate summary statistics
    orig_recovery_rate = np.mean([r['original_recovery'] for r in results]) * 100
    two_phase_recovery_rate = np.mean([r['two_phase_recovery'] for r in results]) * 100
    phase1_recovery_rate = np.mean([r['phase1_recovery'] for r in results]) * 100
    
    orig_avg_rank = np.mean([r['original_rank_of_true'] for r in results])
    two_phase_avg_rank = np.mean([r['two_phase_rank_of_true'] for r in results])
    
    avg_phase1_budget = np.mean([r['phase1_budget_used'] for r in results])
    
    # Create summary
    summary = {
        'Algorithm': ['Original PARWiS', 'Two-Phase Algorithm'],
        'Recovery Rate (%)': [f"{orig_recovery_rate:.2f}", f"{two_phase_recovery_rate:.2f}"],
        'Avg Rank of True Winner': [f"{orig_avg_rank:.2f}", f"{two_phase_avg_rank:.2f}"],
        'Phase 1 Recovery Rate (%)': [f"{orig_recovery_rate:.2f}", f"{phase1_recovery_rate:.2f}"],
        'Avg Phase 1 Budget': [f"{total_budget:.0f}", f"{avg_phase1_budget:.0f}"],
        'Phase 2 k': ['N/A', f"{40}"]
    }
    
    summary_df = pd.DataFrame(summary)
    
    print("="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    
    # Save detailed results
    detailed_results = []
    for i, result in enumerate(results):
        detailed_results.append({
            'Trial': i+1,
            'Original_Recovery': result['original_recovery'],
            'Original_Rank_of_True': result['original_rank_of_true'],
            'TwoPhase_Recovery': result['two_phase_recovery'],
            'TwoPhase_Rank_of_True': result['two_phase_rank_of_true'],
            'Phase1_Recovery': result['phase1_recovery'],
            'Phase1_Budget_Used': result['phase1_budget_used'],
            'K_Used': result['k_used']
        })
    
    detailed_df = pd.DataFrame(detailed_results)
    
    return summary_df, detailed_df
    return summary_df, detailed_df

if __name__ == "__main__":
    # Run comparison
    summary_df, detailed_df = run_full_comparison(
        n=100,
        experiments=10,
        iterations=20,
        budget_multiplier=5,
        stability_window_m=50,
        precomputed=True,
        k=75  # Match the main PARWiS.py score generation parameter
    )
    
    # Save results
    summary_df.to_csv('parwis_two_phase_summary.csv', index=False)
    detailed_df.to_csv('parwis_two_phase_detailed.csv', index=False)
    
    print(f"\nResults saved to:")
    print(f"- parwis_two_phase_summary.csv")
    print(f"- parwis_two_phase_detailed.csv")