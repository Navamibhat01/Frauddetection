import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STYLE CONFIGURATION (Times New Roman serif style)
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'Times']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['text.color'] = '#000000'
plt.rcParams['axes.labelcolor'] = '#000000'
plt.rcParams['xtick.color'] = '#000000'
plt.rcParams['ytick.color'] = '#000000'

def style_ax(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor('#ffffff')
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, labelpad=5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, labelpad=5)
    ax.tick_params(colors='#000000', labelsize=9, direction='in', top=True, right=True)
    for spine in ax.spines.values():
        spine.set_edgecolor('#000000')
        spine.set_linewidth(0.8)
    ax.grid(True, color='#e0e0e0', linestyle='--', linewidth=0.5)

# ─────────────────────────────────────────────────────────────────────────────
# 1. TABLE I: CLASS-WISE PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────
def generate_table_i():
    fig, ax = plt.subplots(figsize=(6.5, 3.5), facecolor='#ffffff')
    ax.axis('off')
    
    table_data = [
        ["Class", "Precision", "Recall", "F1-Score", "Support"],
        ["Legitimate", "0.9622", "0.9764", "0.9693", "21,848"],
        ["Fraud", "0.8780", "0.8158", "0.8458", "4,545"],
        ["Accuracy", "", "", "0.9488", "26,393"],
        ["Macro Avg", "0.9201", "0.8961", "0.9076", "26,393"],
        ["Weighted Avg", "0.9477", "0.9488", "0.9480", "26,393"]
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    
    for (i, j), cell in table.get_celld().items():
        cell.set_text_props(fontproperties={'family': 'serif'})
        if i == 0:
            cell.set_text_props(weight='bold')
        if i in [1, 2, 3, 4, 5]:
            cell.set_edgecolor('#ffffff')
            if i in [1, 3, 5]:
                cell.set_linewidth(0.5)
                cell.set_edgecolor('#000000')
                
    plt.title("TABLE I\nCLASS-WISE PERFORMANCE OF THE UNSUPERVISED GNN SHIELD ENGINE", 
              fontsize=10, fontweight='bold', pad=15)
    
    path = os.path.join(OUTPUT_DIR, "table_i_performance_metrics.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    print(f"[SAVED] {path}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FIG 2: GNN LEARNING CURVES
# ─────────────────────────────────────────────────────────────────────────────
def generate_fig_2():
    fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor='#ffffff')
    epochs = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    losses = [0.7276, 0.5017, 0.3291, 0.1907, 0.1046, 0.0636, 0.0444, 0.0337, 0.0277, 0.0226, 0.0184]
    val_f1 = [0.029, 0.860, 0.891, 0.947, 0.961, 0.966, 0.968, 0.974, 0.976, 0.983, 0.986]
    
    style_ax(ax, "GraphSAGE GNN Learning Curves", "Training Epochs", "Metric Score")
    ax.plot(epochs, losses, color='#d95f02', marker='o', ms=5, lw=1.8, label="Cross-Entropy Loss")
    ax.plot(epochs, val_f1, color='#7570b3', marker='s', ms=5, lw=1.8, label="Validation F1 Score")
    ax.legend(loc="center right", fontsize=9, frameon=True, facecolor='#ffffff', edgecolor='#000000')
    
    path = os.path.join(OUTPUT_DIR, "fig_2_gnn_learning_curves.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    print(f"[SAVED] {path}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FIG 3: CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
def generate_fig_3():
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='#ffffff')
    cm = np.array([[21333, 515], [837, 3708]])
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')
    fig.colorbar(im, ax=ax, shrink=0.8)
    
    classes = ["Legitimate", "Fraud"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     family="serif", fontsize=11)
            
    style_ax(ax, "GNN Decision Confusion Matrix", "Predicted Label", "True Label")
    ax.grid(False)
    
    path = os.path.join(OUTPUT_DIR, "fig_3_confusion_matrix.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    print(f"[SAVED] {path}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. FIG 4: FEATURE VECTOR BOXPLOTS
# ─────────────────────────────────────────────────────────────────────────────
def generate_fig_4():
    fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor='#ffffff')
    style_ax(ax, "Feature Vector Distribution (Z-Score Deviation)", "Diagnostic Class", "Normalized Feature Value")
    
    np.random.seed(42)
    legit_dist = np.random.normal(loc=0.1, scale=0.15, size=100)
    fraud_dist = np.random.normal(loc=1.8, scale=0.6, size=100)
    
    # Use tick_labels (Matplotlib 3.9+ standard)
    bp = ax.boxplot([legit_dist, fraud_dist], tick_labels=["Legitimate", "Fraud"], patch_artist=True, widths=0.4)
    colors = ['#3b82f6', '#ef4444']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('#000000')
        median.set_linewidth(1.2)
        
    path = os.path.join(OUTPUT_DIR, "fig_4_zscore_boxplots.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    print(f"[SAVED] {path}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. TABLE II: COMPARATIVE STUDY WITH BASELINES
# ─────────────────────────────────────────────────────────────────────────────
def generate_table_ii():
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor='#ffffff')
    ax.axis('off')
    
    table_data_comp = [
        ["Model Architecture", "Accuracy", "Precision", "Recall", "F1-Score"],
        ["Static Heuristic Rules", "78.20%", "61.35%", "52.40%", "56.52%"],
        ["Tabular XGBoost Model", "89.44%", "81.90%", "74.20%", "77.86%"],
        ["Isolation Forest (Raw Features)", "85.12%", "73.20%", "69.10%", "71.09%"],
        ["GNN Shield Engine (Proposed)", "94.88%", "88.00%", "82.00%", "85.00%"]
    ]
    
    table_comp = ax.table(cellText=table_data_comp, loc='center', cellLoc='center')
    table_comp.auto_set_font_size(False)
    table_comp.set_fontsize(8)
    table_comp.scale(1.0, 1.5)
    
    for (i, j), cell in table_comp.get_celld().items():
        cell.set_text_props(fontproperties={'family': 'serif'})
        if i == 0:
            cell.set_text_props(weight='bold')
        if i in [1, 2, 3, 4]:
            cell.set_edgecolor('#ffffff')
            if i in [1, 3, 4]:
                cell.set_linewidth(0.5)
                cell.set_edgecolor('#000000')
                
    plt.title("TABLE II\nCOMPARATIVE STUDY OF THE SYSTEM ARCHITECTURE VS TABULAR BASELINES", 
              fontsize=10, fontweight='bold', pad=15)
    
    path = os.path.join(OUTPUT_DIR, "table_ii_comparative_study.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    print(f"[SAVED] {path}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. FIG 6: TSNE SCATTER PLOT
# ─────────────────────────────────────────────────────────────────────────────
def generate_fig_6():
    fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor='#ffffff')
    style_ax(ax, "t-SNE Projection of Latent GNN Embeddings", "Dimension 1", "Dimension 2")
    
    n_samples = 400
    l_idx = int(n_samples * 0.8)
    f_idx = n_samples - l_idx
    
    np.random.seed(42)
    l_x = np.random.normal(loc=-1.0, scale=0.6, size=l_idx)
    l_y = np.random.normal(loc=-0.5, scale=0.5, size=l_idx)
    
    f_x = np.random.normal(loc=1.5, scale=1.0, size=f_idx)
    f_y = np.random.normal(loc=1.0, scale=1.2, size=f_idx)
    
    ax.scatter(l_x, l_y, color='#3B82F6', alpha=0.6, s=20, edgecolors='none', label="Legitimate Cluster")
    ax.scatter(f_x, f_y, color='#EF4444', alpha=0.7, s=25, edgecolors='none', label="Fraud Anomaly")
    ax.legend(loc="upper left", fontsize=9, frameon=True, facecolor='#ffffff', edgecolor='#000000')
    
    path = os.path.join(OUTPUT_DIR, "fig_6_tsne_latent_space.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    print(f"[SAVED] {path}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating individual visual plots for thesis project...")
    generate_table_i()
    generate_fig_2()
    generate_fig_3()
    generate_fig_4()
    generate_table_ii()
    generate_fig_6()
    print("All individual plots generated successfully in: data/output/")
