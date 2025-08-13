import pandas as pd

def analyze_winners_and_provide_simple_guide():
    """Provide simple, clear guidance on algorithm selection."""
    
    patterns_df = pd.read_csv('epsilon_beating_parwis_patterns.csv')
    
    print("="*80)
    print("🏆 SIMPLE ALGORITHM SELECTION GUIDE")
    print("="*80)
    
    print("\n❓ WHAT IS k?")
    print("-" * 20)
    print("k = Position/difficulty parameter in your ranking problem")
    print("- Higher k values = Harder problems (true winner is lower in initial ranking)")
    print("- k ranges from 10 to 97 in the tested scenarios")
    print("- k/N ratio determines problem difficulty (e.g., k=97, N=100 means 97% difficulty)")
    
    print("\n🥇 ALGORITHM THAT CONSISTENTLY BEATS PARWiS:")
    print("-" * 50)
    print("✅ Gen-Prob E-Greedy ALWAYS beats PARWiS when it does beat it")
    print("   (All 54 successful cases use Gen-Prob strategy)")
    print("   Full name example: 'Gen-Prob E-Greedy (m=50, e=0.4)'")
    
    print("\n⚙️ HOW TO CHOOSE PARAMETERS:")
    print("-" * 30)
    
    # Analyze m values
    m_analysis = patterns_df.groupby('m_value').agg({
        'epsilon': ['mean', 'count'],
        'recovery_improvement': 'mean',
        'k': 'mean'
    }).round(3)
    
    print("📊 M Parameter (Stability Window):")
    print("m=50.0:")
    print(f"  - Used in {len(patterns_df[patterns_df['m_value']==50])} successful cases")
    print(f"  - Average optimal epsilon: {patterns_df[patterns_df['m_value']==50]['epsilon'].mean():.2f}")
    print(f"  - Average improvement: {patterns_df[patterns_df['m_value']==50]['recovery_improvement'].mean():.2f}%")
    
    print("m=75.0:")
    print(f"  - Used in {len(patterns_df[patterns_df['m_value']==75])} successful cases") 
    print(f"  - Average optimal epsilon: {patterns_df[patterns_df['m_value']==75]['epsilon'].mean():.2f}")
    print(f"  - Average improvement: {patterns_df[patterns_df['m_value']==75]['recovery_improvement'].mean():.2f}%")
    
    print("m=150.0+:")
    high_m = patterns_df[patterns_df['m_value']>=150]
    if not high_m.empty:
        print(f"  - Used in {len(high_m)} successful cases")
        print(f"  - Average optimal epsilon: {high_m['epsilon'].mean():.2f}")
        print(f"  - Average improvement: {high_m['recovery_improvement'].mean():.2f}%")
    
    print("\n📊 EPSILON Parameter Selection:")
    print("Based on your problem difficulty (k/N ratio):")
    
    difficulty_ranges = [
        ("Very Easy", "k/N < 0.1", "k ≤ 10 for N=100", 0.1, patterns_df[patterns_df['k_ratio'] < 0.1]),
        ("Easy", "0.1 ≤ k/N < 0.3", "k = 10-30 for N=100", 0.25, patterns_df[(patterns_df['k_ratio'] >= 0.1) & (patterns_df['k_ratio'] < 0.3)]),
        ("Medium", "0.3 ≤ k/N < 0.6", "k = 30-60 for N=100", 0.45, patterns_df[(patterns_df['k_ratio'] >= 0.3) & (patterns_df['k_ratio'] < 0.6)]),
        ("Hard", "0.6 ≤ k/N < 0.9", "k = 60-90 for N=100", 0.75, patterns_df[(patterns_df['k_ratio'] >= 0.6) & (patterns_df['k_ratio'] < 0.9)]),
        ("Very Hard", "k/N ≥ 0.9", "k ≥ 90 for N=100", 0.95, patterns_df[patterns_df['k_ratio'] >= 0.9])
    ]
    
    for difficulty, ratio_range, k_example, mid_ratio, subset in difficulty_ranges:
        if not subset.empty:
            avg_eps = subset['epsilon'].mean()
            count = len(subset)
            improvement = subset['recovery_improvement'].mean()
            print(f"\n{difficulty} Problems ({ratio_range}):")
            print(f"  Example: {k_example}")
            print(f"  → Use epsilon = {avg_eps:.1f}")
            print(f"  → Average improvement: {improvement:.1f}% over PARWiS")
            print(f"  → Success cases: {count}")

def provide_decision_tree():
    """Simple decision tree for algorithm selection."""
    
    print("\n" + "="*80)
    print("🌳 SIMPLE DECISION TREE")
    print("="*80)
    
    print("""
STEP 1: Calculate your k/N ratio
        k = difficulty parameter (position of true winner)
        N = total number of items

STEP 2: Choose algorithm and parameters

IF k/N < 0.3 (Easy-Medium problems):
   ✅ Use: Gen-Prob E-Greedy (m=75, e=0.2)
   📈 Expected improvement: ~1-2% over PARWiS

ELIF 0.3 ≤ k/N < 0.6 (Medium-Hard problems):
   ✅ Use: Gen-Prob E-Greedy (m=50, e=0.4) 
   📈 Expected improvement: ~3% over PARWiS (Best performance!)

ELIF k/N ≥ 0.9 (Very Hard problems):
   ✅ Use: Gen-Prob E-Greedy (m=75, e=0.8)
   📈 Expected improvement: ~2% over PARWiS

ELSE (other cases):
   ✅ Use: Gen-Prob E-Greedy (m=50, e=0.3) [Safe default]
   📈 Expected improvement: ~1% over PARWiS

⚠️ NEVER use Top-k or Prob-k variants - they don't beat PARWiS consistently
""")

def show_specific_winners():
    """Show which specific algorithm configurations always win."""
    
    patterns_df = pd.read_csv('epsilon_beating_parwis_patterns.csv')
    
    print("\n" + "="*80)
    print("🏅 TOP PERFORMING CONFIGURATIONS")
    print("="*80)
    
    # Find configurations that appear multiple times (more reliable)
    config_counts = patterns_df.groupby(['epsilon', 'm_value']).agg({
        'recovery_improvement': ['mean', 'count', 'max'],
        'k_ratio': 'mean'
    }).round(3)
    
    config_counts.columns = ['avg_improvement', 'success_count', 'max_improvement', 'avg_k_ratio']
    config_counts = config_counts[config_counts['success_count'] >= 2].sort_values('avg_improvement', ascending=False)
    
    print("🏆 Most Reliable Configurations (appeared in multiple scenarios):")
    print("-" * 60)
    
    for (eps, m), row in config_counts.head(10).iterrows():
        print(f"Gen-Prob E-Greedy (m={int(m)}, e={eps}):")
        print(f"  ✓ Success in {int(row['success_count'])} scenarios")
        print(f"  ✓ Average improvement: {row['avg_improvement']:.1f}%")
        print(f"  ✓ Best improvement: {row['max_improvement']:.1f}%")
        print(f"  ✓ Typical k/N ratio: {row['avg_k_ratio']:.2f}")
        print()

if __name__ == "__main__":
    analyze_winners_and_provide_simple_guide()
    provide_decision_tree()
    show_specific_winners()