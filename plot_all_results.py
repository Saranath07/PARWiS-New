import os
import csv
import matplotlib.pyplot as plt
import re
from collections import defaultdict

def extract_algorithm_name(filename):
    """
    Extract algorithm name from filename pattern: N{N}_k{k}_budget{budget}_{algorithm_name}.csv
    Example: N100_k10_budget500_E-Greedy_(m=50__e=0.0__Prob-k).csv -> E-Greedy_(m=50__e=0.0__Prob-k)
    """
    # Remove the .csv extension first
    filename_no_ext = filename.replace('.csv', '')
    
    # Pattern to match N{number}_k{number}_budget{number}_
    pattern = r'N\d+_k\d+_budget\d+_'
    match = re.search(pattern, filename_no_ext)
    
    if match:
        # Extract everything after the matched pattern
        algorithm_name = filename_no_ext[match.end():]
        return algorithm_name
    return None

def extract_k_value(filename):
    """
    Extract k value from filename
    Example: N100_k10_budget500_E-Greedy_(m=50__e=0.0__Prob-k).csv -> 10
    """
    match = re.search(r'_k(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def plot_algorithm_results(algorithm_name, algorithm_files, folder_path, output_dir):
    """
    Plot results for a specific algorithm across all k values
    """
    plt.figure(figsize=(12, 8))
    
    # Define colors for different k values
    colors = plt.cm.tab10.colors + plt.cm.Set2.colors + plt.cm.Set3.colors
    
    # Sort files by k value for consistent ordering
    algorithm_files.sort(key=lambda x: extract_k_value(x))
    
    for i, file in enumerate(algorithm_files):
        k_value = extract_k_value(file)
        if k_value is None:
            continue
            
        # Read the CSV file
        file_path = os.path.join(folder_path, file)
        budgets = []
        recovery_rates = []
        
        try:
            with open(file_path, 'r') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header row
                for row in csv_reader:
                    budgets.append(int(row[0]))
                    recovery_rates.append(float(row[1]))
            
            # Plot the data
            plt.plot(budgets, recovery_rates,
                     label=f'k={k_value}',
                     color=colors[i % len(colors)],
                     linewidth=2)
                     
        except FileNotFoundError:
            print(f"Warning: Could not find file {file_path}")
            continue
        except Exception as e:
            print(f"Warning: Error reading file {file_path}: {e}")
            continue
    
    # Add labels and title
    plt.xlabel('Budget', fontsize=14)
    plt.ylabel('Average Recovery Rate', fontsize=14)
    plt.title(f'{algorithm_name} - Recovery Rate by Budget for Different k Values', fontsize=16)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    
    # Add a horizontal line at y=1.0 to represent perfect recovery
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Recovery')
    
    # Create safe filename for the algorithm
    safe_algo_name = re.sub(r'[^\w\-.()\s]', '_', algorithm_name)
    safe_algo_name = safe_algo_name.replace(' ', '_')
    
    # Save the plot
    plt.tight_layout()
    output_filename = os.path.join(output_dir, f'{safe_algo_name}_comparison.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print(f"Plot saved: {output_filename}")

def main():
    # Define the folder path containing CSV files
    folder_path = 'all_algos_recovery_by_budget'
    
    # Create output directory for plots
    output_dir = 'algorithm_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Group files by algorithm name
    algorithm_files = defaultdict(list)
    
    for file in csv_files:
        algorithm_name = extract_algorithm_name(file)
        if algorithm_name:
            algorithm_files[algorithm_name].append(file)
        else:
            print(f"Warning: Could not extract algorithm name from {file}")
    
    print(f"Found {len(algorithm_files)} unique algorithms:")
    for algo_name in sorted(algorithm_files.keys()):
        print(f"  - {algo_name} ({len(algorithm_files[algo_name])} files)")
    
    print(f"\nGenerating plots...")
    
    # Generate plot for each algorithm
    for algorithm_name, files in algorithm_files.items():
        print(f"Processing algorithm: {algorithm_name}")
        plot_algorithm_results(algorithm_name, files, folder_path, output_dir)
    
    print(f"\nAll plots have been generated and saved in the '{output_dir}' directory.")

if __name__ == "__main__":
    main()