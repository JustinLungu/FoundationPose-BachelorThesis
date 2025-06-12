import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the results
df = pd.read_csv('plots/results_summary.csv')

# Create output directory for comparison plots
os.makedirs('plots/comparisons', exist_ok=True)

# Set style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]

# Define the fixed order of methods
METHOD_ORDER = ['original', 'gaussian', 'normal', 'outlier', 'speckle', 'zero123', 'magic123', 'dreamfusion']

# Define metrics, their units, and fixed y-axis limits
metrics = {
    'Rotation Error (deg)': {
        'unit': 'Degrees (lower is better)',
        'ymax': 150
    },
    'Translation Error (m)': {
        'unit': 'Meters (lower is better)',
        'ymax': 0.04
    },
    'Pose Error (Frobenius norm)': {
        'unit': 'Error (lower is better)',
        'ymax': 2.5
    },
    'ADD (m)': {
        'unit': 'Meters (lower is better)',
        'ymax': 0.08
    }
}

# Create a plot for each object
for obj_id in df['Object'].unique():
    obj_df = df[df['Object'] == obj_id]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Performance Comparison for Object {obj_id}', fontsize=16)
    
    # Plot each metric
    for ax, (metric, config) in zip(axes.flatten(), metrics.items()):
        # Filter to only include methods in our specified order
        filtered_df = obj_df[obj_df['Method'].isin(METHOD_ORDER)]
        
        # Create bar plot with fixed method order
        sns.barplot(
            data=filtered_df,
            x='Method',
            y=metric,
            order=METHOD_ORDER,
            palette='viridis',
            ax=ax
        )
        
        # Add value labels
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.2f}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 5),
                textcoords='offset points'
            )
        
        ax.set_title(metric)
        ax.set_ylabel(config['unit'])
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        
        # Set consistent y-axis limits
        ax.set_ylim(0, config['ymax'])
    
    plt.tight_layout()
    plt.savefig(f'plots/comparisons/object_{obj_id}_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[✓] Saved comparison plot for object {obj_id}")

print("\n[✓] All comparison plots saved to plots/comparisons/")