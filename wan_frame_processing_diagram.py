import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Define colors
color_input = '#FFE5CC'
color_vae = '#FFD4D4'
color_patch = '#D4E5FF'
color_seq = '#D4FFD4'
color_attn = '#FFD4FF'
color_pad = '#E0E0E0'

# Define positions
y_levels = [10, 8, 6, 4, 2, 0]
labels = ['Input Video', 'VAE Encoded', 'Patch Embedded', 'Flattened Sequence', 'Attention Processing', 'Output']

# Example with 81 frames at 480x832
frames = 81
height = 480
width = 832

# Calculate dimensions through pipeline
vae_h = height // 8  # 60
vae_w = width // 8   # 104
patch_h = vae_h // 2  # 30
patch_w = vae_w // 2  # 52

# Sequence length calculation
seq_len = int(np.ceil((vae_h * vae_w) / (2 * 2) * frames))  # 126,360
seq_len_padded = 32760  # Fixed for 480x832

# Draw the pipeline
# 1. Input Video
rect1 = mpatches.FancyBboxPatch((1, y_levels[0]-0.3), 8, 0.6, 
                                boxstyle="round,pad=0.05", 
                                facecolor=color_input, edgecolor='black', linewidth=2)
ax.add_patch(rect1)
ax.text(5, y_levels[0], f'Input: [3, {frames}, {height}, {width}]', 
        ha='center', va='center', fontsize=11, weight='bold')

# 2. VAE Encoded
rect2 = mpatches.FancyBboxPatch((1, y_levels[1]-0.3), 8, 0.6, 
                                boxstyle="round,pad=0.05", 
                                facecolor=color_vae, edgecolor='black', linewidth=2)
ax.add_patch(rect2)
ax.text(5, y_levels[1], f'VAE: [16, {frames}, {vae_h}, {vae_w}]', 
        ha='center', va='center', fontsize=11, weight='bold')

# 3. Patch Embedded
rect3 = mpatches.FancyBboxPatch((1, y_levels[2]-0.3), 8, 0.6, 
                                boxstyle="round,pad=0.05", 
                                facecolor=color_patch, edgecolor='black', linewidth=2)
ax.add_patch(rect3)
ax.text(5, y_levels[2], f'Patches: [2048, {frames}, {patch_h}, {patch_w}]', 
        ha='center', va='center', fontsize=11, weight='bold')

# 4. Flattened Sequence
# Show actual tokens and padding
actual_tokens = frames * patch_h * patch_w
rect4a = mpatches.FancyBboxPatch((1, y_levels[3]-0.3), 5, 0.6, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor=color_seq, edgecolor='black', linewidth=2)
ax.add_patch(rect4a)
rect4b = mpatches.FancyBboxPatch((6, y_levels[3]-0.3), 3, 0.6, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor=color_pad, edgecolor='black', linewidth=2, linestyle='--')
ax.add_patch(rect4b)
ax.text(3.5, y_levels[3], f'Actual: {actual_tokens:,} tokens', 
        ha='center', va='center', fontsize=10)
ax.text(7.5, y_levels[3], f'Padding', 
        ha='center', va='center', fontsize=10, style='italic')

# 5. Attention Processing
rect5 = mpatches.FancyBboxPatch((1, y_levels[4]-0.3), 8, 0.6, 
                                boxstyle="round,pad=0.05", 
                                facecolor=color_attn, edgecolor='black', linewidth=2)
ax.add_patch(rect5)
ax.text(5, y_levels[4], f'Attention on FULL {seq_len_padded:,} tokens', 
        ha='center', va='center', fontsize=11, weight='bold')

# Add arrows
arrow_props = dict(arrowstyle='->', lw=2, color='black')
for i in range(len(y_levels)-1):
    ax.annotate('', xy=(5, y_levels[i+1]+0.4), xytext=(5, y_levels[i]-0.4),
                arrowprops=arrow_props)

# Add labels
for i, label in enumerate(labels[:-1]):
    ax.text(0.5, y_levels[i], label, ha='right', va='center', fontsize=10, weight='bold')

# Add key insight box
insight_box = mpatches.FancyBboxPatch((10, 2), 3.5, 8, 
                                      boxstyle="round,pad=0.1", 
                                      facecolor='#FFFFCC', edgecolor='red', linewidth=3)
ax.add_patch(insight_box)
ax.text(11.75, 9, 'KEY INSIGHT', ha='center', va='center', fontsize=14, weight='bold', color='red')
ax.text(11.75, 8, 'Sequence Length Formula:', ha='center', va='center', fontsize=11, weight='bold')
ax.text(11.75, 7.3, 'seq_len = (H×W×F) / 256', ha='center', va='center', fontsize=11, family='monospace')
ax.text(11.75, 6.5, 'For 480×832:', ha='center', va='center', fontsize=10)
ax.text(11.75, 6, 'ALWAYS 32,760 tokens', ha='center', va='center', fontsize=11, weight='bold', color='red')
ax.text(11.75, 5.2, 'Regardless of frame count!', ha='center', va='center', fontsize=10, style='italic')

ax.text(11.75, 4, 'Result:', ha='center', va='center', fontsize=11, weight='bold')
ax.text(11.75, 3.3, '• 10 frames = 32,760 tokens', ha='center', va='center', fontsize=9)
ax.text(11.75, 2.8, '• 81 frames = 32,760 tokens', ha='center', va='center', fontsize=9)
ax.text(11.75, 2.3, '• ALL frames = 32,760 tokens', ha='center', va='center', fontsize=9)

# Set axis properties
ax.set_xlim(0, 14)
ax.set_ylim(-0.5, 10.5)
ax.axis('off')

# Add title
plt.title('WAN Model Frame Processing Pipeline\nWhy Fewer Frames Don\'t Speed Up Processing', 
          fontsize=16, weight='bold', pad=20)

# Add note at bottom
ax.text(7, -0.3, 'Note: The model always processes the full sequence length by padding with zeros', 
        ha='center', va='center', fontsize=10, style='italic', color='gray')

plt.tight_layout()
plt.savefig('/Users/agmajima/Downloads/VACE/wan_frame_processing_pipeline.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a second diagram showing the computation flow
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))

# Show different frame counts but same computation
frame_counts = [10, 30, 81]
y_positions = [6, 4, 2]

for i, (frames, y_pos) in enumerate(zip(frame_counts, y_positions)):
    # Input frames
    for j in range(min(frames, 10)):
        frame_rect = mpatches.Rectangle((0.5 + j*0.3, y_pos-0.2), 0.25, 0.4, 
                                       facecolor=color_input, edgecolor='black')
        ax2.add_patch(frame_rect)
    if frames > 10:
        ax2.text(3.5, y_pos, f'... ({frames} frames)', ha='center', va='center', fontsize=9)
    
    # Arrow
    ax2.annotate('', xy=(5, y_pos), xytext=(4.5, y_pos),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Fixed sequence box
    seq_rect = mpatches.FancyBboxPatch((5.5, y_pos-0.3), 4, 0.6, 
                                       boxstyle="round,pad=0.05", 
                                       facecolor=color_attn, edgecolor='black', linewidth=2)
    ax2.add_patch(seq_rect)
    ax2.text(7.5, y_pos, '32,760 tokens\n(ALWAYS)', ha='center', va='center', fontsize=10, weight='bold')
    
    # Computation time
    ax2.text(10.5, y_pos, '= SAME computation time', ha='left', va='center', fontsize=11, color='red', weight='bold')

# Labels
ax2.text(2, 7, 'Input Frames', ha='center', va='center', fontsize=12, weight='bold')
ax2.text(7.5, 7, 'Transformer Processing', ha='center', va='center', fontsize=12, weight='bold')

# Title
ax2.text(6, 8, 'Computation is Independent of Frame Count', ha='center', va='center', fontsize=14, weight='bold')

ax2.set_xlim(0, 12)
ax2.set_ylim(1, 9)
ax2.axis('off')

plt.tight_layout()
plt.savefig('/Users/agmajima/Downloads/VACE/wan_computation_comparison.png', dpi=300, bbox_inches='tight')
plt.show()