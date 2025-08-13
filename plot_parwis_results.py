import os
import csv
import matplotlib.pyplot as plt
import re

# Define the folder path
folder_path = 'all_algos_recovery_by_budget'

# Get all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Set up the plot
plt.figure(figsize=(12, 8))

# Define a colormap for different lines
colors = plt.cm.tab10.colors + plt.cm.Set2.colors + plt.cm.Set3.colors

# Plot each CSV file with a different color
for i, file in enumerate(sorted(csv_files)):
    # Extract N and k values from filename using regex
    match = re.search(r'N_(\d+)_k_(\d+)_', file)
    if match:
        n_value = match.group(1)
        k_value = match.group(2)
        label = f'N={n_value}, k={k_value}'
    else:
        label = file.replace('_parwis_recovery_by_budget.csv', '')
    
    # Read the CSV file using standard csv module
    file_path = os.path.join(folder_path, file)
    budgets = []
    recovery_rates = []
    
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            budgets.append(int(row[0]))
            recovery_rates.append(float(row[1]))
    
    # Plot the data
    plt.plot(budgets, recovery_rates,
             label=label,
             color=colors[i % len(colors)],
             linewidth=2)

# Add labels and title
plt.xlabel('Budget', fontsize=14)
plt.ylabel('Average Recovery Rate', fontsize=14)
plt.title('PARWiS Recovery Rate by Budget for Different k Values (N=100)', fontsize=16)

# Add grid and legend
plt.grid(True, alpha=0.3)
plt.legend(loc='best', fontsize=10)

# Add a horizontal line at y=1.0 to represent perfect recovery
plt.axhline(y=0.1, color='black', linestyle='--', alpha=0.5, label='Perfect Recovery')

# Save the plot
plt.tight_layout()
plt.savefig('parwis_recovery_comparison.png', dpi=300)

# Show the plot
plt.show()

print("Plot has been saved as 'parwis_recovery_comparison.png'")