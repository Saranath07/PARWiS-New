import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os

def comprehensive_recovery_analysis():
    """Analyze recovery rates based on k/N ratio and epsilon values from CSV files."""
    
    print("="*80)
    print("📊 RECOVERY RATE ANALYSIS: k/N vs Epsilon vs Performance")
    print("="*80)
    
    # Load all simulation results
    results_dir = "results/"
    csv_files = glob(os.path.join(results_dir, "*.csv"))
    
    all_results = []
    
    for file in csv_files:
        # Extract N and k from filename
        import re
        match = re.search(r'N_(\d+)_k_(\d+)_simulation_results\.csv', file)
        if match:
            n_val = int(match.group(1))
            k_val = int(match.group(2))
            
            df = pd.read_csv(file)
            df['N'] = n_val
            df['k'] = k_val
            df['k_N_ratio'] = k_val / n_val
            
            # Extract epsilon and algorithm details
            df['epsilon'] = df['Algorithm'].apply(extract_epsilon)
            df['m_value'] = df['Algorithm'].apply(extract_m_value)
            df['algorithm_type'] = df['Algorithm'].apply(parse_algorithm_type)
            
            # Convert recovery percentage to float
            df['recovery_rate'] = df['recovery_pct'].str.rstrip('%').astype(float)
            
            all_results.append(df)
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    print(f"📋 Data Overview: {len(combined_df)} algorithm results across {len(combined_df.groupby(['N', 'k']))} scenarios")
    
    return combined_df

def extract_epsilon(algorithm_name):
    """Extract epsilon value from algorithm name."""
    import re
    match = re.search(r'e=([\d.]+)', algorithm_name)
    return float(match.group(1)) if match else None

def extract_m_value(algorithm_name):
    """Extract m value from algorithm name."""
    import re
    match = re.search(r'm=(\d+)', algorithm_name)
    return int(match.group(1)) if match else None

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

def analyze_recovery_by_k_n_ratio(df):
    """Analyze how k/N ratio affects recovery rates."""
    
    print("\n🎯 RECOVERY RATES BY PROBLEM DIFFICULTY (k/N ratio)")
    print("="*60)
    
    # Create k/N ratio bins
    df['k_N_bin'] = pd.cut(df['k_N_ratio'], 
                          bins=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                          labels=['Very Easy (0-0.1)', 'Easy (0.1-0.3)', 'Medium (0.3-0.5)', 
                                 'Hard (0.5-0.7)', 'Very Hard (0.7-0.9)', 'Extreme (0.9-1.0)'])
    
    # Get PARWiS baseline for each scenario
    parwis_baseline = df[df['algorithm_type'] == 'Pure PARWiS'].groupby(['N', 'k'])['recovery_rate'].first()
    
    # Add baseline to dataframe
    df['parwis_baseline'] = df.apply(lambda row: parwis_baseline.get((row['N'], row['k']), 0), axis=1)
    df['improvement_over_parwis'] = df['recovery_rate'] - df['parwis_baseline']
    
    print("📊 Recovery Rates by Problem Difficulty:")
    print("-" * 50)
    
    difficulty_analysis = df.groupby('k_N_bin').agg({
        'recovery_rate': ['mean', 'max', 'std'],
        'parwis_baseline': 'mean',
        'improvement_over_parwis': 'mean',
        'k': ['min', 'max'],
        'N': 'mean'
    }).round(2)
    
    print(difficulty_analysis)
    
    return df

def analyze_epsilon_impact(df):
    """Analyze how epsilon values impact recovery rates."""
    
    print("\n⚙️ EPSILON IMPACT ON RECOVERY RATES")
    print("="*50)
    
    # Focus on epsilon-greedy algorithms
    epsilon_algos = df[df['epsilon'].notna()].copy()
    
    if epsilon_algos.empty:
        print("No epsilon-greedy algorithms found!")
        return
    
    print("📊 Recovery Rate by Epsilon Value (Gen-Prob E-Greedy only):")
    print("-" * 55)
    
    # Focus on Gen-Prob which performs best
    gen_prob = epsilon_algos[epsilon_algos['algorithm_type'] == 'Gen-Prob E-Greedy'].copy()
    
    epsilon_analysis = gen_prob.groupby('epsilon').agg({
        'recovery_rate': ['mean', 'max', 'std', 'count'],
        'improvement_over_parwis': ['mean', 'max'],
        'k_N_ratio': 'mean'
    }).round(3)
    
    print(epsilon_analysis)
    
    print(f"\n🎯 Best Epsilon Values:")
    print("-" * 25)
    
    # Find best epsilon for different difficulty ranges
    difficulty_ranges = [
        ('Very Easy', gen_prob[gen_prob['k_N_ratio'] <= 0.1]),
        ('Easy', gen_prob[(gen_prob['k_N_ratio'] > 0.1) & (gen_prob['k_N_ratio'] <= 0.3)]),
        ('Medium', gen_prob[(gen_prob['k_N_ratio'] > 0.3) & (gen_prob['k_N_ratio'] <= 0.5)]),
        ('Hard', gen_prob[(gen_prob['k_N_ratio'] > 0.5) & (gen_prob['k_N_ratio'] <= 0.9)]),
        ('Extreme', gen_prob[gen_prob['k_N_ratio'] > 0.9])
    ]
    
    for difficulty, subset in difficulty_ranges:
        if not subset.empty:
            best_eps = subset.groupby('epsilon')['recovery_rate'].mean().idxmax()
            best_rate = subset.groupby('epsilon')['recovery_rate'].mean().max()
            avg_improvement = subset.groupby('epsilon')['improvement_over_parwis'].mean().max()
            
            print(f"{difficulty} (k/N {subset['k_N_ratio'].min():.2f}-{subset['k_N_ratio'].max():.2f}):")
            print(f"  → Best epsilon: {best_eps}")
            print(f"  → Recovery rate: {best_rate:.1f}%")
            print(f"  → Improvement over PARWiS: +{avg_improvement:.1f}%")
            print()

def create_detailed_comparison_table(df):
    """Create a detailed table showing k/N, epsilon, and recovery rates."""
    
    print("\n📋 DETAILED COMPARISON TABLE")
    print("="*50)
    
    # Focus on scenarios where epsilon-greedy beats PARWiS
    epsilon_algos = df[df['epsilon'].notna()].copy()
    epsilon_algos['beats_parwis'] = epsilon_algos['improvement_over_parwis'] > 0
    
    winners = epsilon_algos[epsilon_algos['beats_parwis']].copy()
    
    if winners.empty:
        print("No epsilon-greedy configurations beat PARWiS!")
        return
    
    # Create summary table
    summary_table = winners.groupby(['k_N_ratio', 'epsilon', 'algorithm_type']).agg({
        'recovery_rate': 'mean',
        'parwis_baseline': 'mean',
        'improvement_over_parwis': 'mean',
        'k': 'first',
        'N': 'first'
    }).round(2).reset_index()
    
    # Sort by improvement
    summary_table = summary_table.sort_values('improvement_over_parwis', ascending=False)
    
    print("🏆 TOP PERFORMING CONFIGURATIONS:")
    print("-" * 40)
    print(f"{'k/N':>6} {'k':>4} {'N':>4} {'ε':>5} {'Strategy':>15} {'Recovery%':>10} {'PARWiS%':>9} {'Improve':>8}")
    print("-" * 75)
    
    for _, row in summary_table.head(20).iterrows():
        strategy_short = row['algorithm_type'].replace('Gen-Prob E-Greedy', 'Gen-Prob').replace('E-Greedy Top-k', 'Top-k').replace('E-Greedy Prob-k', 'Prob-k')
        print(f"{row['k_N_ratio']:>6.2f} {int(row['k']):>4} {int(row['N']):>4} {row['epsilon']:>5.1f} {strategy_short:>15} "
              f"{row['recovery_rate']:>9.1f}% {row['parwis_baseline']:>8.1f}% {row['improvement_over_parwis']:>+7.1f}%")
    
    return summary_table

def generate_recommendations_table(df):
    """Generate practical recommendations table."""
    
    print("\n🎯 PRACTICAL RECOMMENDATIONS TABLE")
    print("="*50)
    
    epsilon_algos = df[df['epsilon'].notna()].copy()
    gen_prob = epsilon_algos[epsilon_algos['algorithm_type'] == 'Gen-Prob E-Greedy'].copy()
    gen_prob['beats_parwis'] = gen_prob['improvement_over_parwis'] > 0
    winners = gen_prob[gen_prob['beats_parwis']].copy()
    
    if winners.empty:
        return
    
    # Create recommendation bins
    k_n_ranges = [
        (0.0, 0.1, "Very Easy"),
        (0.1, 0.3, "Easy"),
        (0.3, 0.5, "Medium"), 
        (0.5, 0.7, "Hard"),
        (0.7, 0.9, "Very Hard"),
        (0.9, 1.0, "Extreme")
    ]
    
    print("💡 QUICK SELECTION GUIDE:")
    print("-" * 30)
    print(f"{'Problem Type':>12} {'k/N Range':>12} {'Best ε':>8} {'Avg Recovery':>12} {'Improvement':>12}")
    print("-" * 70)
    
    for min_ratio, max_ratio, difficulty in k_n_ranges:
        subset = winners[(winners['k_N_ratio'] >= min_ratio) & (winners['k_N_ratio'] < max_ratio)]
        
        if not subset.empty:
            # Find best epsilon for this range
            best_config = subset.groupby('epsilon').agg({
                'recovery_rate': 'mean',
                'improvement_over_parwis': 'mean'
            }).sort_values('improvement_over_parwis', ascending=False)
            
            if not best_config.empty:
                best_eps = best_config.index[0]
                best_recovery = best_config.iloc[0]['recovery_rate']
                best_improvement = best_config.iloc[0]['improvement_over_parwis']
                
                print(f"{difficulty:>12} {min_ratio:.1f}-{max_ratio:.1f}      {best_eps:>5.1f} {best_recovery:>11.1f}% {best_improvement:>+11.1f}%")

def main():
    """Main analysis function."""
    
    # Load and analyze data
    df = comprehensive_recovery_analysis()
    
    # Analyze by k/N ratio
    df = analyze_recovery_by_k_n_ratio(df)
    
    # Analyze epsilon impact
    analyze_epsilon_impact(df)
    
    # Create detailed comparison
    summary_table = create_detailed_comparison_table(df)
    
    # Generate recommendations
    generate_recommendations_table(df)
    
    # Save results
    if summary_table is not None:
        summary_table.to_csv('recovery_rate_detailed_analysis.csv', index=False)
        print(f"\n💾 Saved detailed analysis to 'recovery_rate_detailed_analysis.csv'")

if __name__ == "__main__":
    main()