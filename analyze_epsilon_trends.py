import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from glob import glob

def load_and_analyze_results():
    """Load all CSV files and analyze trends for epsilon selection."""
    
    results_dir = "results/"
    csv_files = glob(os.path.join(results_dir, "*.csv"))
    
    all_data = []
    
    print("Loading CSV files...")
    for file in csv_files:
        # Extract N and k values from filename
        match = re.search(r'N_(\d+)_k_(\d+)_simulation_results\.csv', file)
        if match:
            n_val = int(match.group(1))
            k_val = int(match.group(2))
            
            df = pd.read_csv(file)
            df['N'] = n_val
            df['k'] = k_val
            df['scenario'] = f"N{n_val}_k{k_val}"
            
            all_data.append(df)
    
    if not all_data:
        print("No CSV files found!")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Parse algorithm details
    combined_df['algorithm_type'] = combined_df['Algorithm'].apply(parse_algorithm_type)
    combined_df['epsilon'] = combined_df['Algorithm'].apply(extract_epsilon)
    combined_df['m_value'] = combined_df['Algorithm'].apply(extract_m_value)
    combined_df['strategy'] = combined_df['Algorithm'].apply(extract_strategy)
    
    # Convert percentage strings to float
    for col in ['recovery_pct', 'p1_correct_pct']:
        if col in combined_df.columns:
            combined_df[col + '_float'] = combined_df[col].str.rstrip('%').astype(float)
    
    return combined_df

def parse_algorithm_type(algorithm_name):
    """Extract algorithm family from name."""
    if "Pure PARWiS" in algorithm_name:
        return "Pure PARWiS"
    elif "PARWiS-P2" in algorithm_name:
        return "PARWiS-P2"
    elif "Gen-Prob E-Greedy" in algorithm_name:
        return "Gen-Prob E-Greedy"
    elif "E-Greedy" in algorithm_name and "Top-k" in algorithm_name:
        return "E-Greedy Top-k"
    elif "E-Greedy" in algorithm_name and "Prob-k" in algorithm_name:
        return "E-Greedy Prob-k"
    else:
        return "Other"

def extract_epsilon(algorithm_name):
    """Extract epsilon value from algorithm name."""
    match = re.search(r'e=([\d.]+)', algorithm_name)
    return float(match.group(1)) if match else None

def extract_m_value(algorithm_name):
    """Extract m value from algorithm name."""
    match = re.search(r'm=(\d+)', algorithm_name)
    return int(match.group(1)) if match else None

def extract_strategy(algorithm_name):
    """Extract strategy type."""
    if "Top-k" in algorithm_name:
        return "Top-k"
    elif "Prob-k" in algorithm_name:
        return "Prob-k"
    elif "Gen-Prob" in algorithm_name:
        return "Gen-Prob"
    else:
        return "None"

def analyze_epsilon_performance(df):
    """Analyze performance patterns for different epsilon values."""
    
    print("=== EPSILON PERFORMANCE ANALYSIS ===")
    
    # Focus on epsilon-greedy algorithms
    epsilon_algos = df[df['epsilon'].notna()].copy()
    
    if epsilon_algos.empty:
        print("No epsilon-greedy algorithms found!")
        return
    
    # Group by scenario and find best performing algorithms
    best_performers = []
    
    for scenario in epsilon_algos['scenario'].unique():
        scenario_data = epsilon_algos[epsilon_algos['scenario'] == scenario].copy()
        
        # Find best recovery rate in this scenario
        best_recovery = scenario_data['recovery_pct_float'].max()
        best_algos = scenario_data[scenario_data['recovery_pct_float'] == best_recovery]
        
        # Also get PARWiS baseline for comparison
        parwis_data = df[(df['scenario'] == scenario) & (df['algorithm_type'] == 'Pure PARWiS')]
        parwis_recovery = parwis_data['recovery_pct_float'].iloc[0] if not parwis_data.empty else 0
        
        for _, row in best_algos.iterrows():
            best_performers.append({
                'scenario': scenario,
                'N': row['N'],
                'k': row['k'],
                'algorithm': row['Algorithm'],
                'algorithm_type': row['algorithm_type'],
                'epsilon': row['epsilon'],
                'm_value': row['m_value'],
                'strategy': row['strategy'],
                'recovery_pct': row['recovery_pct_float'],
                'p1_winner_score': row['p1_winner_score'],
                'p1_2nd_score': row['p1_2nd_score'],
                'p1_true_winner_score': row['p1_true_winner_score'],
                'p1_diff': row['P1 Avg difference'],
                'p2_diff': row['P2 Avg difference'],
                'parwis_recovery': parwis_recovery,
                'beats_parwis': row['recovery_pct_float'] > parwis_recovery
            })
    
    best_df = pd.DataFrame(best_performers)
    
    if best_df.empty:
        print("No best performers found!")
        return best_df
    
    print(f"\nFound {len(best_df)} best performing configurations")
    print(f"Scenarios where epsilon-greedy beats PARWiS: {best_df['beats_parwis'].sum()}")
    
    return best_df

def find_epsilon_patterns(best_df):
    """Find patterns in optimal epsilon selection."""
    
    print("\n=== EPSILON SELECTION PATTERNS ===")
    
    # Group by scenario characteristics
    patterns = []
    
    for _, row in best_df.iterrows():
        if row['beats_parwis']:
            patterns.append({
                'scenario': row['scenario'],
                'N': row['N'],
                'k': row['k'],
                'k_ratio': row['k'] / row['N'],  # Important: ratio of k to N
                'epsilon': row['epsilon'],
                'm_value': row['m_value'],
                'strategy': row['strategy'],
                'p1_diff': row['p1_diff'],
                'p1_true_winner_score': row['p1_true_winner_score'],
                'recovery_improvement': row['recovery_pct'] - row['parwis_recovery']
            })
    
    if not patterns:
        print("No scenarios where epsilon-greedy beats PARWiS!")
        return pd.DataFrame()
    
    patterns_df = pd.DataFrame(patterns)
    
    print(f"\nScenarios where epsilon-greedy beats PARWiS: {len(patterns_df)}")
    
    # Analyze patterns by k_ratio
    print("\n--- Pattern Analysis by k/N ratio ---")
    patterns_df['k_ratio_bin'] = pd.cut(patterns_df['k_ratio'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    ratio_analysis = patterns_df.groupby('k_ratio_bin').agg({
        'epsilon': ['mean', 'std', 'count'],
        'p1_diff': ['mean', 'std'],
        'p1_true_winner_score': ['mean', 'std'],
        'recovery_improvement': 'mean'
    }).round(4)
    
    print(ratio_analysis)
    
    # Analyze patterns by strategy
    print("\n--- Pattern Analysis by Strategy ---")
    strategy_analysis = patterns_df.groupby('strategy').agg({
        'epsilon': ['mean', 'std', 'count'],
        'p1_diff': ['mean', 'std'],
        'recovery_improvement': 'mean'
    }).round(4)
    
    print(strategy_analysis)
    
    return patterns_df

def create_visualizations(df, best_df, patterns_df):
    """Create visualizations to understand epsilon trends."""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Recovery rate vs Epsilon for different strategies
    epsilon_algos = df[df['epsilon'].notna()].copy()
    
    if not epsilon_algos.empty:
        for strategy in ['Top-k', 'Prob-k', 'Gen-Prob']:
            strategy_data = epsilon_algos[epsilon_algos['strategy'] == strategy]
            if not strategy_data.empty:
                grouped = strategy_data.groupby('epsilon')['recovery_pct_float'].mean()
                axes[0, 0].plot(grouped.index, grouped.values, marker='o', label=strategy, alpha=0.7)
        
        axes[0, 0].set_xlabel('Epsilon')
        axes[0, 0].set_ylabel('Average Recovery %')
        axes[0, 0].set_title('Recovery Rate vs Epsilon by Strategy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Phase 1 difference vs Optimal Epsilon
    if not patterns_df.empty:
        scatter = axes[0, 1].scatter(patterns_df['p1_diff'], patterns_df['epsilon'], 
                                   c=patterns_df['k_ratio'], cmap='viridis', alpha=0.7)
        axes[0, 1].set_xlabel('Phase 1 Score Difference')
        axes[0, 1].set_ylabel('Optimal Epsilon')
        axes[0, 1].set_title('Phase 1 Difference vs Optimal Epsilon')
        plt.colorbar(scatter, ax=axes[0, 1], label='k/N Ratio')
    
    # Plot 3: k/N ratio vs Optimal Epsilon
    if not patterns_df.empty:
        axes[1, 0].scatter(patterns_df['k_ratio'], patterns_df['epsilon'], alpha=0.7)
        axes[1, 0].set_xlabel('k/N Ratio')
        axes[1, 0].set_ylabel('Optimal Epsilon')
        axes[1, 0].set_title('k/N Ratio vs Optimal Epsilon')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: True Winner Score vs Optimal Epsilon
    if not patterns_df.empty:
        axes[1, 1].scatter(patterns_df['p1_true_winner_score'], patterns_df['epsilon'], alpha=0.7)
        axes[1, 1].set_xlabel('Phase 1 True Winner Score')
        axes[1, 1].set_ylabel('Optimal Epsilon')
        axes[1, 1].set_title('True Winner Score vs Optimal Epsilon')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('epsilon_trends_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function."""
    print("Starting comprehensive epsilon trend analysis...")
    
    # Load and process data
    df = load_and_analyze_results()
    if df is None:
        return
    
    print(f"Loaded {len(df)} algorithm results across {len(df['scenario'].unique())} scenarios")
    
    # Analyze best performers
    best_df = analyze_epsilon_performance(df)
    
    if not best_df.empty:
        # Find patterns in epsilon selection
        patterns_df = find_epsilon_patterns(best_df)
        
        # Create visualizations
        create_visualizations(df, best_df, patterns_df)
        
        # Save detailed results
        if not patterns_df.empty:
            patterns_df.to_csv('epsilon_beating_parwis_patterns.csv', index=False)
            print(f"\nSaved detailed patterns to 'epsilon_beating_parwis_patterns.csv'")
        
        best_df.to_csv('best_performers_analysis.csv', index=False)
        print(f"Saved best performers analysis to 'best_performers_analysis.csv'")
        
        # Print summary insights
        print("\n=== SUMMARY INSIGHTS ===")
        if not patterns_df.empty:
            print(f"1. Epsilon-greedy beats PARWiS in {len(patterns_df)} scenarios")
            print(f"2. Average optimal epsilon: {patterns_df['epsilon'].mean():.3f}")
            print(f"3. Most common strategy: {patterns_df['strategy'].mode().iloc[0]}")
            print(f"4. Average improvement over PARWiS: {patterns_df['recovery_improvement'].mean():.2f}%")
            
            # Correlation analysis
            correlations = patterns_df[['epsilon', 'k_ratio', 'p1_diff', 'p1_true_winner_score']].corr()['epsilon']
            print(f"\n5. Correlations with optimal epsilon:")
            for var in ['k_ratio', 'p1_diff', 'p1_true_winner_score']:
                print(f"   - {var}: {correlations[var]:.3f}")
        else:
            print("No clear patterns found where epsilon-greedy consistently beats PARWiS")

if __name__ == "__main__":
    main()