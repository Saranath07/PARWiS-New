import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def load_all_results():
    """
    Loads all CSV files from the results folder and combines them with N and k columns.
    """
    all_dataframes = []
    
    # Get all CSV files from results folder
    csv_files = glob.glob('results/N_*_k_*_simulation_results.csv')
    
    for file_path in csv_files:
        try:
            # Extract N and k from filename
            filename = os.path.basename(file_path)
            match = re.match(r'N_(\d+)_k_(\d+)_simulation_results\.csv', filename)
            if match:
                N = int(match.group(1))
                k = int(match.group(2))
                
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Add N and k columns
                df['N'] = N
                df['k'] = k
                
                all_dataframes.append(df)
                print(f"Loaded {file_path}: N={N}, k={k}, rows={len(df)}")
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"Combined data: {len(combined_df)} total rows")
        return combined_df
    else:
        print("No data files found!")
        return None

def analyze_and_plot(results_df):
    """
    Processes the combined dataframe and generates analysis plots.
    """
    if results_df is None:
        print("No data to process!")
        return
        
    # Clean up percentage columns by removing '%' and converting to float
    for col in ['recovery_pct', 'p1_correct_pct', 'P2 Changes %', 'P2 Fixes %', 'P2 Errors %']:
        if col in results_df.columns and results_df[col].dtype == 'object':
            results_df[col] = results_df[col].str.replace('%', '').astype(float)

    # --- Feature Engineering ---
    # Define a function to extract parameters from the 'Algorithm' column
    def extract_params(algo_name):
        m_match = re.search(r'm=(\d+)', algo_name)
        e_match = re.search(r'e=([\d.]+)', algo_name)
        
        m = int(m_match.group(1)) if m_match else np.nan
        e = float(e_match.group(1)) if e_match else np.nan
        
        family = 'Unknown'
        if 'Gen-Prob E-Greedy' in algo_name:
            family = 'Gen-Prob E-Greedy'
        elif 'PARWiS-P2' in algo_name:
            family = 'PARWiS-P2'
        elif 'Pure PARWiS' in algo_name:
            family = 'Baseline'
            
        return m, e, family

    # Apply the function to create new columns
    results_df[['m', 'epsilon', 'family']] = results_df['Algorithm'].apply(lambda x: pd.Series(extract_params(x)))

    # Calculate performance gain over the PARWiS baseline for each 'N' and 'k' combination
    parwis_baseline = results_df[results_df['family'] == 'Baseline'].set_index(['N', 'k'])['recovery_pct']
    results_df['recovery_gain'] = results_df.apply(lambda row: row['recovery_pct'] - parwis_baseline.get((row['N'], row['k']), 0), axis=1)

    # --- Visualization ---

    # 1. Create separate heatmaps for each N value
    gen_prob_df = results_df[results_df['family'] == 'Gen-Prob E-Greedy'].copy()
    n_values = sorted(results_df['N'].unique())
    
    if not gen_prob_df.empty:
        for n_val in n_values:
            try:
                n_data = gen_prob_df[gen_prob_df['N'] == n_val]
                
                if not n_data.empty:
                    heatmap_data = n_data.pivot_table(index=['m', 'epsilon'], columns='k', values='recovery_gain').sort_index(ascending=False)

                    plt.figure(figsize=(16, 10))
                    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.1f', linewidths=.5, cbar_kws={'label': 'Performance Gain (%)'})
                    plt.title(f'Performance Gain (%) vs. PARWiS Baseline for Gen-Prob E-Greedy (N={n_val})', fontsize=16)
                    plt.ylabel('Algorithm Parameters (m, epsilon)', fontsize=12)
                    plt.xlabel('k (Problem Difficulty)', fontsize=12)
                    plt.tight_layout()
                    plt.savefig(f'heatmap_recovery_gain_N{n_val}.png', dpi=300, bbox_inches='tight')
                    print(f"Successfully generated: heatmap_recovery_gain_N{n_val}.png")
                    plt.close()

            except Exception as e:
                print(f"Could not generate heatmap for N={n_val}: {e}")

    # 2. Plotting best algorithm vs baseline for each N value
    try:
        baseline_df = results_df[results_df['family'] == 'Baseline']
        
        # Create separate plots for each N value
        n_values = sorted(results_df['N'].unique())
        
        fig, axes = plt.subplots(len(n_values), 1, figsize=(14, 6*len(n_values)))
        if len(n_values) == 1:
            axes = [axes]
            
        for i, n_val in enumerate(n_values):
            n_data = results_df[results_df['N'] == n_val]
            n_baseline = baseline_df[baseline_df['N'] == n_val]
            
            if not n_data.empty and not n_baseline.empty:
                best_algos = n_data.loc[n_data.groupby('k')['recovery_pct'].idxmax()]
                
                axes[i].plot(n_baseline['k'], n_baseline['recovery_pct'], 'ro--', label='PARWiS Baseline', markersize=8)
                axes[i].plot(best_algos['k'], best_algos['recovery_pct'], 'go-', label='Best Performing Algorithm', markersize=8)
                
                # Annotate the best performing algorithm's parameters
                for j, row in best_algos.iterrows():
                    if row['family'] != 'Baseline':
                        axes[i].text(row['k'], row['recovery_pct'] + 1, f"m={int(row['m'])}, e={row['epsilon']:.1f}", fontsize=8, ha='center')

                axes[i].set_title(f'Best Algorithm vs. PARWiS Baseline (N={n_val})', fontsize=14)
                axes[i].set_xlabel('k (Problem Difficulty)', fontsize=12)
                axes[i].set_ylabel('Recovery Percentage (%)', fontsize=12)
                axes[i].legend()
                axes[i].grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig('best_vs_baseline_by_N.png', dpi=300, bbox_inches='tight')
        print("Successfully generated: best_vs_baseline_by_N.png")
    except Exception as e:
        print(f"Could not generate best vs baseline plot: {e}")

    # Save the processed data for your own inspection
    results_df.to_csv("processed_simulation_results.csv", index=False)
    print("Successfully generated: processed_simulation_results.csv")


# --- Main execution ---
if __name__ == "__main__":
    print("Loading data from results folder...")
    combined_data = load_all_results()
    
    if combined_data is not None:
        print("\nProcessing and generating plots...")
        analyze_and_plot(combined_data)
        print("\nAnalysis complete!")
    else:
        print("Failed to load data!")