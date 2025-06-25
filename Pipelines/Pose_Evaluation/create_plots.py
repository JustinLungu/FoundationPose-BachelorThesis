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

# Define metrics, their units, y-limits, and both acceptable & outlier thresholds
metrics = {
    'Rotation Error (deg)': {
        'unit': 'Degrees (lower is better)',
        'ymax': 150,
        'lower_thresh':  5.0,    # start of “acceptable” band (°)
        'upper_thresh': 10.0     # outlier cutoff (°)
    },
    'Translation Error (m)': {
        'unit': 'Meters (lower is better)',
        'ymax': 0.1,
        'lower_thresh': 0.02,    # start of “acceptable” band (m)
        'upper_thresh': 0.05     # outlier cutoff (m)
    },
    'Pose Error (Frobenius norm)': {
        'unit': 'Error (lower is better)',
        'ymax': 2.5,
        'lower_thresh': 0.05,    # start of “acceptable” band (unitless)
        'upper_thresh': 0.13     # outlier cutoff (unitless)
    },
    'ADD (m)': {
        'unit': 'Meters (lower is better)',
        'ymax': 0.08,
        'lower_thresh': 0.02,    # start of “acceptable” band (m)
        'upper_thresh': 0.05     # outlier cutoff (m)
    }
}

# Create a plot for each object
for obj_id in df['Object'].unique():
    obj_df = df[df['Object'] == obj_id]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Performance Comparison for Object {obj_id}', fontsize=16)
    
    for ax, (metric, cfg) in zip(axes.flatten(), metrics.items()):
        filtered_df = obj_df[obj_df['Method'].isin(METHOD_ORDER)]
        
        sns.barplot(
            data=filtered_df,
            x='Method', y=metric,
            order=METHOD_ORDER,
            palette='viridis',
            ax=ax
        )
        
        # Annotate bar values
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.2f}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points'
            )
        
        # Draw “acceptable” lower bound
        ax.axhline(cfg['lower_thresh'], linestyle='--', linewidth=2, color='orange',
                   label=f'Acceptable ≥ {cfg["lower_thresh"]:.2f}')
        # Draw “outlier” upper bound
        ax.axhline(cfg['upper_thresh'], linestyle='--', linewidth=2, color='red',
                   label=f'Outlier ≥ {cfg["upper_thresh"]:.2f}')
        
        ax.legend(loc='upper right')
        ax.set_title(metric)
        ax.set_ylabel(cfg['unit'])
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, cfg['ymax'])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'plots/comparisons/object_{obj_id}_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[✓] Saved comparison plot for object {obj_id}")

print("\n[✓] All comparison plots saved to plots/comparisons/")
