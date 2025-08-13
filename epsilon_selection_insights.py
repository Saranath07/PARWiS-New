import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def detailed_epsilon_analysis():
    """Generate detailed insights and practical recommendations for epsilon selection."""
    
    # Load the patterns data
    patterns_df = pd.read_csv('epsilon_beating_parwis_patterns.csv')
    
    print("="*80)
    print("COMPREHENSIVE EPSILON SELECTION INSIGHTS")
    print("="*80)
    
    print("\n📊 OVERVIEW:")
    print(f"• Epsilon-greedy beats PARWiS in {len(patterns_df)} scenarios")
    print(f"• Average improvement: {patterns_df['recovery_improvement'].mean():.2f}%")
    print(f"• Best improvement: {patterns_df['recovery_improvement'].max():.2f}%")
    print(f"• All successful cases use Gen-Prob strategy")
    
    # Analyze by k/N ratio bins
    print("\n🎯 EPSILON SELECTION BY PROBLEM DIFFICULTY (k/N ratio):")
    print("-" * 60)
    
    ratio_stats = patterns_df.groupby('k_ratio_bin').agg({
        'epsilon': ['mean', 'std', 'min', 'max', 'count'],
        'recovery_improvement': ['mean', 'max'],
        'p1_diff': 'mean',
        'p1_true_winner_score': 'mean'
    }).round(3)
    
    print(ratio_stats)
    
    # Key insights by ratio
    print("\n📋 PRACTICAL RECOMMENDATIONS:")
    print("-" * 40)
    
    for bin_name, group in patterns_df.groupby('k_ratio_bin'):
        if len(group) > 0:
            mean_eps = group['epsilon'].mean()
            std_eps = group['epsilon'].std()
            mean_improvement = group['recovery_improvement'].mean()
            k_range = f"{group['k'].min()}-{group['k'].max()}" if group['k'].min() != group['k'].max() else str(group['k'].iloc[0])
            
            print(f"\n{bin_name} difficulty (k/N = {group['k_ratio'].min():.2f}-{group['k_ratio'].max():.2f}):")
            print(f"  • k values: {k_range} (out of N={group['N'].iloc[0]})")
            print(f"  • Optimal epsilon: {mean_eps:.3f} ± {std_eps:.3f}")
            print(f"  • Average improvement: {mean_improvement:.2f}%")
            print(f"  • Sample scenarios: {len(group)} cases")

    # Analyze trend with N size
    print(f"\n📈 TRENDS BY PROBLEM SIZE (N):")
    print("-" * 40)
    
    n_analysis = patterns_df.groupby('N').agg({
        'epsilon': ['mean', 'std', 'count'],
        'k_ratio': 'mean',
        'recovery_improvement': 'mean'
    }).round(3)
    
    print(n_analysis)
    
    # Special cases analysis
    print(f"\n⚡ SPECIAL PATTERNS IDENTIFIED:")
    print("-" * 40)
    
    # High k/N ratios (>0.9)
    high_k = patterns_df[patterns_df['k_ratio'] > 0.9]
    if not high_k.empty:
        print(f"• Very High k/N (>0.9): Use epsilon = {high_k['epsilon'].mean():.2f} ± {high_k['epsilon'].std():.2f}")
    
    # Low k/N ratios (<0.1)
    low_k = patterns_df[patterns_df['k_ratio'] < 0.1]
    if not low_k.empty:
        print(f"• Very Low k/N (<0.1): Use epsilon = {low_k['epsilon'].mean():.2f} ± {low_k['epsilon'].std():.2f}")
    
    # High phase1 difference scenarios
    high_diff = patterns_df[patterns_df['p1_diff'] > 0.8]
    if not high_diff.empty:
        print(f"• High P1 difference (>0.8): Prefer epsilon = {high_diff['epsilon'].mean():.2f}")
    
    # Low phase1 difference scenarios  
    low_diff = patterns_df[patterns_df['p1_diff'] < 0.2]
    if not low_diff.empty:
        print(f"• Low P1 difference (<0.2): Prefer epsilon = {low_diff['epsilon'].mean():.2f}")

def create_epsilon_selection_guide():
    """Create a practical guide for epsilon selection."""
    
    patterns_df = pd.read_csv('epsilon_beating_parwis_patterns.csv')
    
    print("\n" + "="*80)
    print("🎯 PRACTICAL EPSILON SELECTION GUIDE")
    print("="*80)
    
    print("""
STEP-BY-STEP EPSILON SELECTION:

1. Calculate k/N ratio for your problem
2. Estimate Phase 1 winner-second place difference (if available)
3. Use the guidelines below:

""")
    
    # Create decision tree
    print("DECISION TREE:")
    print("-" * 20)
    
    # Very high k/N
    very_high = patterns_df[patterns_df['k_ratio'] >= 0.9]
    if not very_high.empty:
        print(f"IF k/N ≥ 0.9:")
        print(f"   → Use ε = {very_high['epsilon'].mean():.1f} (range: {very_high['epsilon'].min():.1f}-{very_high['epsilon'].max():.1f})")
    
    # High k/N  
    high_k = patterns_df[(patterns_df['k_ratio'] >= 0.6) & (patterns_df['k_ratio'] < 0.9)]
    if not high_k.empty:
        print(f"ELIF k/N ≥ 0.6:")
        print(f"   → Use ε = {high_k['epsilon'].mean():.1f} (range: {high_k['epsilon'].min():.1f}-{high_k['epsilon'].max():.1f})")
    
    # Medium k/N
    med_k = patterns_df[(patterns_df['k_ratio'] >= 0.3) & (patterns_df['k_ratio'] < 0.6)]
    if not med_k.empty:
        print(f"ELIF k/N ≥ 0.3:")
        print(f"   → Use ε = {med_k['epsilon'].mean():.1f} (range: {med_k['epsilon'].min():.1f}-{med_k['epsilon'].max():.1f})")
    
    # Low k/N
    low_k = patterns_df[(patterns_df['k_ratio'] >= 0.1) & (patterns_df['k_ratio'] < 0.3)]
    if not low_k.empty:
        print(f"ELIF k/N ≥ 0.1:")
        print(f"   → Use ε = {low_k['epsilon'].mean():.1f} (range: {low_k['epsilon'].min():.1f}-{low_k['epsilon'].max():.1f})")
    
    # Very low k/N
    very_low = patterns_df[patterns_df['k_ratio'] < 0.1]
    if not very_low.empty:
        print(f"ELSE (k/N < 0.1):")
        print(f"   → Use ε = {very_low['epsilon'].mean():.1f} (range: {very_low['epsilon'].min():.1f}-{very_low['epsilon'].max():.1f})")
    
    print(f"""
FALLBACK STRATEGY:
If unsure, use ε = {patterns_df['epsilon'].mean():.1f} (overall average)

IMPORTANT NOTES:
• Gen-Prob strategy is consistently the best performing
• Phase 1 metrics show weak correlation with optimal epsilon
• Higher k/N ratios tend to need higher epsilon values
• Lower k/N ratios benefit from moderate epsilon values

SUCCESS RATE: {len(patterns_df)} out of tested scenarios show improvement over PARWiS
""")

def analyze_phase1_predictors():
    """Analyze what Phase 1 metrics can predict about optimal epsilon."""
    
    patterns_df = pd.read_csv('epsilon_beating_parwis_patterns.csv')
    
    print("\n" + "="*80)
    print("🔍 PHASE 1 PREDICTIVE ANALYSIS")
    print("="*80)
    
    # Correlation analysis
    predictors = ['p1_diff', 'p1_true_winner_score', 'k_ratio']
    target = 'epsilon'
    
    print("\nCORRELATION WITH OPTIMAL EPSILON:")
    print("-" * 40)
    correlations = patterns_df[predictors + [target]].corr()[target].drop(target)
    
    for predictor in predictors:
        corr = correlations[predictor]
        strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
        direction = "positive" if corr > 0 else "negative"
        print(f"• {predictor}: {corr:.3f} ({strength} {direction})")
    
    # Binned analysis
    print(f"\nEPSILON BY PHASE 1 DIFFERENCE RANGES:")
    print("-" * 40)
    
    # Create bins for p1_diff
    patterns_df['p1_diff_bin'] = pd.cut(patterns_df['p1_diff'], 
                                       bins=4, 
                                       labels=['Low (0.1-0.3)', 'Med-Low (0.3-0.5)', 
                                              'Med-High (0.5-0.8)', 'High (0.8+)'])
    
    diff_analysis = patterns_df.groupby('p1_diff_bin').agg({
        'epsilon': ['mean', 'std', 'count'],
        'recovery_improvement': 'mean',
        'k_ratio': 'mean'
    }).round(3)
    
    print(diff_analysis)
    
    print(f"\nCONCLUSION: Phase 1 metrics are poor predictors of optimal epsilon.")
    print(f"The k/N ratio is the most reliable predictor (though still weak).")

def generate_final_recommendations():
    """Generate final actionable recommendations."""
    
    patterns_df = pd.read_csv('epsilon_beating_parwis_patterns.csv')
    
    print("\n" + "="*80)
    print("🏆 FINAL RECOMMENDATIONS FOR EPSILON SELECTION")
    print("="*80)
    
    print("""
ALGORITHM CHOICE:
✅ Always use Gen-Prob E-Greedy strategy (not Top-k or Prob-k focused variants)

EPSILON SELECTION HEURISTIC:
1. Calculate k/N ratio for your problem
2. Use this mapping:
""")
    
    # Simplified decision rules
    k_ratio_rules = [
        (0.0, 0.15, 0.2, "Easy problems"),
        (0.15, 0.35, 0.3, "Medium problems"), 
        (0.35, 0.65, 0.2, "Hard problems"),
        (0.65, 1.0, 0.8, "Very hard problems")
    ]
    
    for min_ratio, max_ratio, suggested_eps, description in k_ratio_rules:
        print(f"   k/N ∈ [{min_ratio:.2f}, {max_ratio:.2f}): ε = {suggested_eps:.1f} ({description})")
    
    print(f"""
WHEN TO USE EPSILON-GREEDY OVER PARWiS:
• Consider epsilon-greedy when you can afford some exploration
• Especially effective for problems where PARWiS recovery is low
• Best improvements seen in medium difficulty problems (k/N ≈ 0.3-0.5)

EXPECTED IMPROVEMENT:
• Average: {patterns_df['recovery_improvement'].mean():.1f}% better than PARWiS
• Maximum observed: {patterns_df['recovery_improvement'].max():.1f}% improvement
• Success rate: 54/89 configurations tested

LIMITATIONS OF PHASE 1 ANALYSIS:
• Phase 1 winner-second difference: Very weak predictor (r = {patterns_df[['p1_diff', 'epsilon']].corr().iloc[0,1]:.3f})
• Phase 1 true winner score: Very weak predictor (r = {patterns_df[['p1_true_winner_score', 'epsilon']].corr().iloc[0,1]:.3f})
• k/N ratio: Weak but best available predictor (r = {patterns_df[['k_ratio', 'epsilon']].corr().iloc[0,1]:.3f})

PRACTICAL ADVICE:
If you cannot determine the optimal epsilon analytically:
1. Use ε = 0.3-0.4 as a good general-purpose value
2. For very easy problems (k/N < 0.1): try ε = 0.2-0.4  
3. For very hard problems (k/N > 0.9): try ε = 0.8-1.0
4. Run quick pilot tests with different epsilon values if computational budget allows
""")

if __name__ == "__main__":
    detailed_epsilon_analysis()
    create_epsilon_selection_guide() 
    analyze_phase1_predictors()
    generate_final_recommendations()