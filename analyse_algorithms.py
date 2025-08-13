import pandas as pd
import os
import re
from collections import defaultdict

def create_comparison_table(n_values: list, k_values: list, base_path: str = 'results') -> pd.DataFrame:
    """
    Analyzes and aggregates simulation results from multiple CSV files.

    This function iterates through combinations of N and k values to find CSV files
    named like '{base_path}/N_{N}_k_{k}_simulation_results.csv'. It compares the
    'recovery_pct' of 'Gen-Prob E-Greedy' algorithms against the 'Pure PARWiS (Baseline)'.

    The results are aggregated based on the algorithm's epsilon value (e) and whether
    its memory parameter (m) corresponds to 50% or 75% of N for that file.

    Args:
        n_values: A list of N values to iterate through (e.g., [100, 200]).
        k_values: A list of k values to iterate through (e.g., [10, 20, ...]).
        base_path: The directory path where the result files are stored.

    Returns:
        A pandas DataFrame summarizing the comparison across all files, where:
        - The index is the Epsilon value.
        - Columns show the total counts for 'beats' and 'exactly equal'
          for both 50% and 75% of N.
    """
    # This dictionary will store the aggregated counts for each epsilon.
    # e.g., {0.1: {'50% N beats': 2, '75% N beats': 5, ...}}
    final_counts = defaultdict(lambda: {
        '50% N beats': 0,
        '75% N beats': 0,
        '50% N (exactly equal)': 0,
        '75% N (exactly equal)': 0
    })

    # --- 1. Loop through all specified file combinations ---
    for N in n_values:
        # Calculate the m values that correspond to 50% and 75% of the current N
        n_50_pct_m = 0.5 * N
        n_75_pct_m = 0.75 * N

        for k in k_values:
            filename = os.path.join(base_path, f'N_{N}_k_{k}_simulation_results.csv')

            if not os.path.exists(filename):
                # Silently skip files that don't exist
                continue

            try:
                df = pd.read_csv(filename)
            except Exception as e:
                print(f"Error reading {filename}, skipping. Error: {e}")
                continue

            # --- 2. Data Cleaning and Preparation ---
            if 'recovery_pct' not in df.columns or 'Algorithm' not in df.columns:
                print(f"Warning: Missing required columns in {filename}, skipping.")
                continue
            
            if df['recovery_pct'].dtype == 'object':
                df['recovery_pct'] = df['recovery_pct'].str.replace('%', '', regex=False).astype(float)

            # --- 3. Get Baseline Performance ---
            baseline_row = df[df['Algorithm'] == 'Pure PARWiS (Baseline)']
            if baseline_row.empty:
                print(f"Warning: Baseline algorithm not found in {filename}, skipping.")
                continue
            baseline_pct = baseline_row['recovery_pct'].iloc[0]

            # --- 4. Filter and Extract Parameters for Comparison Algos ---
            # Filter for only 'Gen-Prob E-Greedy' algorithms
            greedy_algos = df[df['Algorithm'].str.contains("Gen-Prob E-Greedy", na=False)].copy()

            # Use regex to extract 'm' and 'epsilon' from the algorithm name
            params = greedy_algos['Algorithm'].str.extract(
                r"\(m=(?P<m>[\d\.]+),\s*e=(?P<epsilon>[\d\.]+)\)"
            )
            
            # Join extracted params back and convert to numeric types
            greedy_algos = greedy_algos.join(params)
            greedy_algos.dropna(subset=['m', 'epsilon'], inplace=True)
            greedy_algos[['m', 'epsilon']] = greedy_algos[['m', 'epsilon']].apply(pd.to_numeric)


            # --- 5. Compare and Aggregate Results ---
            for _, row in greedy_algos.iterrows():
                epsilon = row['epsilon']
                m = row['m']
                
                # Determine which m-percentage this row corresponds to
                m_category = None
                if m == n_50_pct_m:
                    m_category = '50% N'
                elif m == n_75_pct_m:
                    m_category = '75% N'
                else:
                    # Skip if m doesn't match the expected 50% or 75% of N
                    continue

                # Compare to baseline and update counts in our main dictionary
                if row['recovery_pct'] > baseline_pct:
                    final_counts[epsilon][f'{m_category} beats'] += 1
                elif row['recovery_pct'] == baseline_pct:
                    final_counts[epsilon][f'{m_category} (exactly equal)'] += 1
    
    # --- 6. Create Final DataFrame from Aggregated Data ---
    if not final_counts:
        return pd.DataFrame() # Return empty DataFrame if no data was processed

    result_df = pd.DataFrame.from_dict(final_counts, orient='index')
    
    # Ensure all columns exist, fill missing with 0, and set column order
    final_columns = ['50% N beats', '75% N beats', '50% N (exactly equal)', '75% N (exactly equal)']
    result_df = result_df.reindex(columns=final_columns).fillna(0).astype(int)

    # Sort by epsilon value for a clean, ordered table
    result_df.sort_index(inplace=True)
    result_df.index.name = 'Epsilon'

    return result_df

# --- Example of how to use the function ---

# 1. Define the N and k values for the files you want to process.
N_values_to_process = [100, 200, 300, 400, 500]
k_values_to_process = [10, 20, 30, 40, 50, 60, 70, 90, 93, 95, 97]

# 2. Specify the path to the directory containing your CSV files.
#    (Create a dummy 'results' folder with some sample files to test this)
results_directory = 'results' 

    # You would place your actual N_{N}_k_{k}_simulation_results.csv files here.
    # For now, the script will just print warnings that files are not found.


# 3. Call the function to get the aggregated table.
final_table = create_comparison_table(
    n_values=N_values_to_process,
    k_values=k_values_to_process,
    base_path=results_directory
)

# 4. Print the final result and save it to a new CSV file.
if not final_table.empty:
    print("--- Aggregated Comparison Results ---")
    print(final_table)
    
    output_filename = 'aggregated_comparison_results.csv'
    final_table.to_csv(output_filename)
    print(f"\nResults saved to '{output_filename}'")
else:
    print("No data processed. Ensure your files are in the 'results' directory and match the naming convention.")

