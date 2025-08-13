#!/usr/bin/env python3
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re

def extract_k_value(filename):
    """Extract k value from filename pattern N_100_k_XX_..."""
    match = re.search(r'N_\d+_k_(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return 0

# Set the figure size and style
plt.figure(figsize=(12, 8))
plt.style.use('ggplot')

# Get all CSV files in the parwis-results directory
csv_files = sorted(glob.glob('parwis-results/N_*_k_*_parwis_recovery_by_budget.csv'), 
                  key=extract_k_value)

# Color map for different lines
cmap = plt.get_cmap('viridis', len(csv_files))

# Process each CSV file
for i, csv_file in enumerate(csv_files):
    # Extract k value for the legend
    k_value = extract_k_value(csv_file)
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Get the recovery rate at budget 500 (last value)
    recovery_at_500 = df.iloc[-1, 1]
    
    # Calculate difference: recovery_at_500 - recovery_at_each_budget
    df['diff'] = recovery_at_500 - df['Average Recovery Rate']
    
    # Plot the difference vs budget
    plt.plot(df['Budget'], df['diff'], label=f'k={k_value}', 
             color=cmap(i/len(csv_files)), linewidth=2)

# Add labels and title
plt.xlabel('Budget', fontsize=14)
plt.ylabel('Recovery Improvement (Budget 500 - Current Budget)', fontsize=14)
plt.title('PARWiS Recovery Improvement with Budget', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend with smaller font size and multiple columns if needed
if len(csv_files) > 10:
    plt.legend(fontsize=10, ncol=2)
else:
    plt.legend(fontsize=10)

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Add text describing the plot
plt.figtext(0.5, 0.01, 
            'This plot shows how recovery improves as budget increases.\n'
            'The y-axis represents the difference between recovery at max budget (500) and each budget level.',
            ha='center', fontsize=10)

# Tight layout
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# Save the figure
plt.savefig('parwis_budget_recovery_difference.png', dpi=300, bbox_inches='tight')
print(f"Plot saved as 'parwis_budget_recovery_difference.png'")

# Display the figure
plt.show()