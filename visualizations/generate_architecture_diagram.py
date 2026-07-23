import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.colors as mcolors

# Set font styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

def draw_diagram():
    # Figure setup (8x10 inches as requested)
    fig, ax = plt.subplots(figsize=(8, 10), facecolor='#ffffff')
    ax.set_facecolor('#ffffff')
    
    # Hide axes, ticks, spines
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12.6)
    
    # Title at the top
    ax.text(5.0, 12.3, "UPI Adaptive Fraud Shield: End-to-End Pipeline", 
            fontsize=15, fontweight='bold', ha='center', va='center', color='#111111')
    
    # Subtitle or decorative line
    ax.plot([2.5, 7.5], [12.1, 12.1], color='#cccccc', lw=1.0)
    
    # Stage configuration
    stages = {
        1: {
            "num": 1,
            "title": "Data Ingestion &\nPreprocessing",
            "color": "#1f77b4",  # Blue theme
            "x": 0.6,
            "y": 8.5,
            "sub_boxes": ["Raw Transaction Input", "Clean + Impute", "Binary Flag Encoding"],
            "highlight_idx": -1
        },
        2: {
            "num": 2,
            "title": "Heterogeneous\nGraph Construction",
            "color": "#2ca02c",  # Green theme
            "x": 5.1,
            "y": 8.5,
            "sub_boxes": ["User / Device / Merchant /\nTxn Nodes", "Shared-Device Edges"],
            "highlight_idx": -1
        },
        3: {
            "num": 3,
            "title": "GraphSAGE\nEncoder",
            "color": "#9467bd",  # Purple theme
            "x": 0.6,
            "y": 4.5,
            "sub_boxes": ["Layer 1: Mean Aggregation", "Layer 2: 2-hop Aggregation", "64-D Embedding"],
            "highlight_idx": -1
        },
        4: {
            "num": 4,
            "title": "Isolation\nForest",
            "color": "#ff7f0e",  # Orange theme
            "x": 5.1,
            "y": 4.5,
            "sub_boxes": ["Path-Length Isolation", "Anomaly Score"],
            "highlight_idx": -1
        },
        5: {
            "num": 5,
            "title": "Risk Score &\nDecision Engine",
            "color": "#d62728",  # Red theme
            "x": 0.6,
            "y": 0.5,
            "sub_boxes": ["0-100% Risk Calibration", "Allow / Block Routing"],
            "highlight_idx": 1  # Highlight the second sub-box (0-indexed: index 1)
        },
        6: {
            "num": 6,
            "title": "XAI &\nDashboard",
            "color": "#17becf",  # Teal theme
            "x": 5.1,
            "y": 0.5,
            "sub_boxes": ["Z-Score Influence Ranking", "Streamlit Live Report"],
            "highlight_idx": -1
        }
    }
    
    # Drawing Stage Panels and Sub-boxes
    panel_w = 3.9
    panel_h = 3.5
    pad = 0.6  # requested boxstyle="round,pad=0.6"
    
    # We will draw the panels
    for s_id, s in stages.items():
        x_min, y_min = s["x"], s["y"]
        x_max, y_max = x_min + panel_w, y_min + panel_h
        theme_color = s["color"]
        
        # 15% opacity fill of theme color, 2px colored border
        fc = mcolors.to_rgba(theme_color, alpha=0.15)
        ec = mcolors.to_rgba(theme_color, alpha=1.0)
        
        # Draw the main stage panel
        panel_patch = patches.FancyBboxPatch(
            (x_min + pad, y_min + pad),
            panel_w - 2 * pad,
            panel_h - 2 * pad,
            boxstyle=f"round,pad={pad}",
            ec=ec,
            fc=fc,
            lw=2.0
        )
        ax.add_patch(panel_patch)
        
        # Add Stage Title in top-left of the panel (aligned to avoid overlapping)
        ax.text(
            x_min + 0.18,
            y_max - 0.45,
            f"Stage {s['num']}: {s['title']}",
            fontsize=9.5,
            fontweight='bold',
            color=theme_color,
            ha='left',
            va='center',
            linespacing=1.2
        )
        
        # Draw sub-boxes inside the panel
        sub_boxes = s["sub_boxes"]
        n_sub = len(sub_boxes)
        x_center = x_min + panel_w / 2.0
        box_w = 3.3
        box_pad = 0.1
        
        if n_sub == 3:
            # Centers for 3 sub-boxes
            centers_y = [y_max - 1.1, y_max - 1.9, y_max - 2.7]
            box_h = 0.5
        else:
            # Centers for 2 sub-boxes
            centers_y = [y_max - 1.35, y_max - 2.45]
            box_h = 0.65
            
        for idx, label in enumerate(sub_boxes):
            cy = centers_y[idx]
            is_highlighted = (idx == s["highlight_idx"])
            
            # Sub-box styling (white or 8% fill, thin border)
            sub_ec = theme_color if is_highlighted else '#777777'
            sub_fc = '#ffffff' if is_highlighted else mcolors.to_rgba(theme_color, alpha=0.08)
            sub_lw = 2.5 if is_highlighted else 0.8
            
            sub_patch = patches.FancyBboxPatch(
                (x_center - box_w / 2.0 + box_pad, cy - box_h / 2.0 + box_pad),
                box_w - 2 * box_pad,
                box_h - 2 * box_pad,
                boxstyle=f"round,pad={box_pad}",
                ec=sub_ec,
                fc=sub_fc,
                lw=sub_lw
            )
            ax.add_patch(sub_patch)
            
            # Centered label text
            lbl_weight = 'bold' if is_highlighted else 'normal'
            lbl_color = '#000000' if is_highlighted else '#222222'
            ax.text(
                x_center,
                cy,
                label,
                fontsize=8.5,
                fontweight=lbl_weight,
                color=lbl_color,
                ha='center',
                va='center'
            )
            
            # Internal flow arrows
            if idx < n_sub - 1:
                next_cy = centers_y[idx + 1]
                arrow_y_start = cy - box_h / 2.0
                arrow_y_end = next_cy + box_h / 2.0
                
                inner_arrow = patches.FancyArrowPatch(
                    (x_center, arrow_y_start),
                    (x_center, arrow_y_end),
                    arrowstyle="-|>",
                    color=theme_color,
                    lw=1.0,
                    mutation_scale=8
                )
                ax.add_patch(inner_arrow)
                
    # --- CONNECTIONS ---
    # Solid black arrows connecting stages 1->2->3->4->5->6
    arrow_color = '#111111'
    arrow_lw = 1.5
    arrow_mutation = 12
    
    # 1 -> 2: Horizontal
    # Stage 1 right edge to Stage 2 left edge
    arrow_1_2 = patches.FancyArrowPatch(
        (0.6 + panel_w, 8.5 + panel_h / 2.0),
        (5.1, 8.5 + panel_h / 2.0),
        arrowstyle="-|>",
        color=arrow_color,
        lw=arrow_lw,
        mutation_scale=arrow_mutation,
        shrinkA=2,
        shrinkB=2
    )
    ax.add_patch(arrow_1_2)
    
    # 2 -> 3: Stepped
    # Stage 2 bottom center to Stage 3 top center
    path_2_3 = Path([
        (5.1 + panel_w/2.0, 8.5),
        (5.1 + panel_w/2.0, 8.25),
        (0.6 + panel_w/2.0, 8.25),
        (0.6 + panel_w/2.0, 8.0)
    ], [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO])
    arrow_2_3 = patches.FancyArrowPatch(
        path=path_2_3,
        arrowstyle="-|>",
        color=arrow_color,
        lw=arrow_lw,
        mutation_scale=arrow_mutation,
        shrinkA=2,
        shrinkB=2
    )
    ax.add_patch(arrow_2_3)
    
    # 3 -> 4: Horizontal
    # Stage 3 right edge to Stage 4 left edge
    arrow_3_4 = patches.FancyArrowPatch(
        (0.6 + panel_w, 4.5 + panel_h / 2.0),
        (5.1, 4.5 + panel_h / 2.0),
        arrowstyle="-|>",
        color=arrow_color,
        lw=arrow_lw,
        mutation_scale=arrow_mutation,
        shrinkA=2,
        shrinkB=2
    )
    ax.add_patch(arrow_3_4)
    
    # 4 -> 5: Stepped
    # Stage 4 bottom center to Stage 5 top center
    path_4_5 = Path([
        (5.1 + panel_w/2.0, 4.5),
        (5.1 + panel_w/2.0, 4.25),
        (0.6 + panel_w/2.0, 4.25),
        (0.6 + panel_w/2.0, 4.0)
    ], [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO])
    arrow_4_5 = patches.FancyArrowPatch(
        path=path_4_5,
        arrowstyle="-|>",
        color=arrow_color,
        lw=arrow_lw,
        mutation_scale=arrow_mutation,
        shrinkA=2,
        shrinkB=2
    )
    ax.add_patch(arrow_4_5)
    
    # 5 -> 6: Horizontal
    # Stage 5 right edge to Stage 6 left edge
    arrow_5_6 = patches.FancyArrowPatch(
        (0.6 + panel_w, 0.5 + panel_h / 2.0),
        (5.1, 0.5 + panel_h / 2.0),
        arrowstyle="-|>",
        color=arrow_color,
        lw=arrow_lw,
        mutation_scale=arrow_mutation,
        shrinkA=2,
        shrinkB=2
    )
    ax.add_patch(arrow_5_6)
    
    # --- FEEDBACK LOOP ---
    # One dashed gray arrow looping from Stage 6 back to Stage 2
    path_feedback = Path([
        (5.1 + panel_w, 0.5 + panel_h / 2.0),
        (9.6, 0.5 + panel_h / 2.0),
        (9.6, 8.5 + panel_h / 2.0),
        (5.1 + panel_w, 8.5 + panel_h / 2.0)
    ], [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO])
    
    feedback_arrow = patches.FancyArrowPatch(
        path=path_feedback,
        arrowstyle="-|>",
        linestyle="--",
        color="#7f8c8d",
        lw=1.5,
        mutation_scale=arrow_mutation,
        shrinkA=2,
        shrinkB=2
    )
    ax.add_patch(feedback_arrow)
    
    # Label for feedback loop: "ledger update / continuous learning" in small italic text
    # Placed vertically along the path x = 9.6
    ax.text(
        9.75,
        (2.25 + 10.25) / 2.0,  # mid point y
        "ledger update / continuous learning",
        fontsize=8.5,
        fontstyle='italic',
        color='#5f6c6d',
        ha='left',
        va='center',
        rotation=270
    )
    
    # Save the figure tight-cropped
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    png_path = os.path.join(base_dir, "architecture_diagram.png")
    svg_path = os.path.join(base_dir, "architecture_diagram.svg")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
    plt.savefig(svg_path, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    
    print(f"Successfully generated and saved architecture diagram:")
    print(f"  PNG (300 DPI): {png_path}")
    print(f"  SVG:          {svg_path}")

if __name__ == "__main__":
    draw_diagram()
