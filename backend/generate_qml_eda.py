import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle, Ellipse

plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 10), facecolor='#0d1117')

# 1. Feature Space Mapping (Classical vs Quantum)
ax1 = plt.subplot(2, 2, 1)
ax1.set_title('Feature Mapping: ML (Linear) vs QML (Hilbert Space)', color='white', fontsize=14, pad=15)
ax1.axis('off')
# Classical
ax1.text(0.2, 0.9, 'Classical ML\nLow-Dimensional Space', color='#a5b4fc', ha='center', fontsize=12)
ax1.plot([0.1, 0.3], [0.2, 0.8], 'w-', alpha=0.3)
ax1.scatter([0.1, 0.2, 0.3], [0.2, 0.5, 0.8], color='#38bdf8', s=100)
# Quantum
ax1.text(0.8, 0.9, 'Quantum ML\nExponential Hilbert Space', color='#fca5a5', ha='center', fontsize=12)
circle = Circle((0.8, 0.5), 0.3, fill=False, color='#fca5a5', alpha=0.5, linestyle='--')
ax1.add_patch(circle)
ax1.plot([0.8, 0.8], [0.2, 0.8], 'w-', alpha=0.3)
ax1.plot([0.5, 1.1], [0.5, 0.5], 'w-', alpha=0.3)
ax1.scatter([0.7, 0.9, 0.8, 0.85], [0.6, 0.4, 0.7, 0.3], color='#fb7185', s=100)
ax1.text(0.5, -0.1, "Explanation: Classical ML maps features to flat spaces.\nQML wraps data around complex quantum spheres (Bloch sphere),\nallowing it to see hidden 'minor' connections.", 
         color='lightgray', ha='center', fontsize=10, bbox=dict(facecolor='#1e293b', alpha=0.5, edgecolor='none'))

# 2. Decision Boundary (Macro vs Micro)
ax2 = plt.subplot(2, 2, 2)
ax2.set_title('Decision Boundaries: Macro vs Micro-level', color='white', fontsize=14, pad=15)
np.random.seed(42)
x = np.random.rand(100)
y = np.random.rand(100)
c = np.where(x**2 + y**2 > 0.6, '#38bdf8', '#fb7185') # Some complex boundary
ax2.scatter(x, y, c=c, alpha=0.6, s=50)
# Classical Boundary
theta = np.linspace(0, np.pi/2, 50)
ax2.plot(np.cos(theta)*0.77, np.sin(theta)*0.77, color='#38bdf8', linewidth=3, label='Classical ML (Smooth)')
# Quantum Boundary
q_x = np.cos(theta)*0.77 + np.sin(theta*20)*0.05
q_y = np.sin(theta)*0.77 + np.cos(theta*20)*0.05
ax2.plot(q_x, q_y, color='#fb7185', linewidth=2, linestyle='--', label='QML (Micro-Entangled)')
ax2.legend(loc='upper right', frameon=False, labelcolor='white')
ax2.axis('off')
ax2.text(0.5, -0.1, "Explanation: ML draws broad, smooth lines, often missing tiny anomalies.\nQML uses entanglement to draw highly complex, intricate boundaries,\npicking up 'minor level' features that ML ignores.", 
         color='lightgray', ha='center', fontsize=10, bbox=dict(facecolor='#1e293b', alpha=0.5, edgecolor='none'))

# 3. Entanglement vs Independence
ax3 = plt.subplot(2, 2, 3)
ax3.set_title('Feature Relationship: Independence vs Entanglement', color='white', fontsize=14, pad=15)
ax3.axis('off')
ax3.text(0.25, 0.8, 'Classical ML\nIsolated Features', color='#a5b4fc', ha='center', fontsize=12)
ax3.scatter([0.2, 0.3, 0.25], [0.4, 0.4, 0.6], color='#38bdf8', s=200)
ax3.text(0.75, 0.8, 'Quantum ML\nEntangled Features', color='#fca5a5', ha='center', fontsize=12)
ax3.scatter([0.7, 0.8, 0.75], [0.4, 0.4, 0.6], color='#fb7185', s=200)
ax3.plot([0.7, 0.8], [0.4, 0.4], color='#fb7185', linestyle='-', alpha=0.5)
ax3.plot([0.7, 0.75], [0.4, 0.6], color='#fb7185', linestyle='-', alpha=0.5)
ax3.plot([0.8, 0.75], [0.4, 0.6], color='#fb7185', linestyle='-', alpha=0.5)
ax3.text(0.5, 0.1, "Explanation: Classical ML treats variables (like Age, Vitals) mostly independently.\nQML literally entangles them. If Heart Rate and Opacity have a microscopic\ncorrelation, Quantum Entanglement locks them together instantly.", 
         color='lightgray', ha='center', fontsize=10, bbox=dict(facecolor='#1e293b', alpha=0.5, edgecolor='none'))

# 4. Processing Power (Matrix vs Tensor)
ax4 = plt.subplot(2, 2, 4)
ax4.set_title('Capacity: Sequential vs Parallel Superposition', color='white', fontsize=14, pad=15)
ax4.axis('off')
# ML
ax4.text(0.25, 0.8, 'Classical Neural Net\nEvaluates 1 state at a time', color='#a5b4fc', ha='center', fontsize=10)
for i in range(3):
    ax4.add_patch(plt.Rectangle((0.15, 0.3 + i*0.15), 0.2, 0.1, color='#38bdf8', alpha=0.5))
# QML
ax4.text(0.75, 0.8, 'Quantum Circuit\nSuperposition (Evaluates ALL at once)', color='#fca5a5', ha='center', fontsize=10)
ax4.add_patch(plt.Rectangle((0.65, 0.3), 0.2, 0.4, color='#fb7185', alpha=0.5))
ax4.text(0.75, 0.5, '2^N States\nSimultaneously', color='white', ha='center', va='center', fontsize=12, fontweight='bold')

ax4.text(0.5, 0.1, "Explanation: ML has to process minor details step-by-step (slow for high detail).\nQML explores all possible minor states simultaneously via superposition,\nmaking it exceptionally fast at detecting subtle X-Ray anomalies.", 
         color='lightgray', ha='center', fontsize=10, bbox=dict(facecolor='#1e293b', alpha=0.5, edgecolor='none'))

plt.tight_layout(pad=4.0)
out_path = r"C:\Users\shubham dixit\.gemini\antigravity-ide\brain\dedeba13-1582-4504-b4ba-569de608b834\qml_vs_ml_eda.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'EDA Saved successfully to {out_path}')
