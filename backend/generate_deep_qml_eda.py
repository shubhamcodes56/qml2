import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 7), facecolor='#0d1117')
fig.suptitle('Deep Dive: Why Quantum ML Destroys Classical ML in Complexity', color='white', fontsize=18, y=1.05)

# ==========================================
# Panel 1: Optimization (Local Minima vs Quantum Tunneling)
# ==========================================
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.set_facecolor('#0d1117')
ax1.set_title('1. Optimization & Learning', color='#38bdf8', fontsize=14, pad=10)

X = np.linspace(-5, 5, 50)
Y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(X, Y)
# A complex loss landscape with a fake barrier
Z = np.sin(X) + np.cos(Y) + 0.1*(X**2 + Y**2)

ax1.plot_surface(X, Y, Z, cmap='magma', alpha=0.6, edgecolor='none')
ax1.axis('off')

# Classical ML path (stuck in local minima)
ax1.scatter([-2], [-2], [Z[15, 15] + 0.5], color='#38bdf8', s=200, label='Classical ML\n(Stuck in Local Minima)')
# QML path (Quantum Tunneling)
ax1.scatter([0], [0], [Z[25, 25]], color='#fca5a5', s=300, marker='*', label='Quantum ML\n(Tunnels to Global Minima)')

# Draw a line showing tunneling
ax1.plot([-2, 0], [-2, 0], [Z[15, 15] + 0.5, Z[25, 25]], color='#fca5a5', linestyle='--', linewidth=2)

ax1.legend(loc='upper right', facecolor='#1e293b', edgecolor='none', labelcolor='white')

ax1.text2D(0.5, -0.1, "ML uses Gradient Descent (rolling down a hill)\nand gets stuck in 'Local Minima' (fake solutions).\nQML uses 'Quantum Tunneling' to literally pass\nthrough mathematical walls to find the perfect solution.", 
           transform=ax1.transAxes, color='lightgray', ha='center', fontsize=10, bbox=dict(facecolor='#1e293b', alpha=0.8, edgecolor='none'))

# ==========================================
# Panel 2: Computational Complexity Scaling
# ==========================================
ax2 = fig.add_subplot(1, 3, 2)
ax2.set_facecolor('#0d1117')
ax2.set_title('2. Processing Power Scaling', color='#38bdf8', fontsize=14, pad=10)

n_features = np.linspace(1, 50, 100)
ml_time = 2**(n_features/10) # Exponential
qml_time = n_features * 0.1 # Linear/Polynomial

ax2.plot(n_features, ml_time, color='#38bdf8', linewidth=3, label='Classical ML (Exponential Time)')
ax2.plot(n_features, qml_time, color='#fca5a5', linewidth=3, label='Quantum ML (Linear Time)')

ax2.set_xlabel('Number of Complex Medical Features (Variables)', color='gray')
ax2.set_ylabel('Compute Time Required', color='gray')
ax2.tick_params(colors='gray')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_color('gray')
ax2.spines['left'].set_color('gray')

ax2.fill_between(n_features, ml_time, qml_time, where=(ml_time>qml_time), color='red', alpha=0.1)
ax2.text(25, 15, 'Intractable Zone\n(ML crashes or takes years)', color='#fb7185', ha='center', fontsize=12)

ax2.legend(loc='upper left', facecolor='#1e293b', edgecolor='none', labelcolor='white')

ax2.text(0.5, -0.2, "As you add more variables (Vitals + Xray + Blood),\nClassical ML combinations explode exponentially.\nQML handles infinite combinations linearly because\nqubits process ALL combinations simultaneously.", 
         transform=ax2.transAxes, color='lightgray', ha='center', fontsize=10, bbox=dict(facecolor='#1e293b', alpha=0.8, edgecolor='none'))

# ==========================================
# Panel 3: State Representation (Bit vs Qubit)
# ==========================================
ax3 = fig.add_subplot(1, 3, 3)
ax3.set_facecolor('#0d1117')
ax3.set_title('3. Data Capacity (8 Bits vs 8 Qubits)', color='#38bdf8', fontsize=14, pad=10)
ax3.axis('off')

# Classical bits
for i in range(8):
    ax3.add_patch(plt.Rectangle((0.1, 0.85 - i*0.1), 0.1, 0.05, color='#38bdf8'))
ax3.text(0.15, 0.95, '8 Classical Bits\n= 8 Pieces of Info', color='#a5b4fc', ha='center')

# Quantum bits (Network of entangled states)
import math
angles = np.linspace(0, 2*math.pi, 16, endpoint=False)
q_x = 0.7 + 0.2 * np.cos(angles)
q_y = 0.5 + 0.3 * np.sin(angles)

for i in range(len(q_x)):
    for j in range(i+1, len(q_x)):
        ax3.plot([q_x[i], q_x[j]], [q_y[i], q_y[j]], color='#fb7185', alpha=0.1, linewidth=0.5)

ax3.scatter(q_x, q_y, color='#fca5a5', s=50, zorder=5)
ax3.text(0.7, 0.95, '8 Qubits (Superposition)\n= 256 Simultaneous States', color='#fca5a5', ha='center')
ax3.text(0.7, 0.45, 'Highly Entangled\nInformation Web', color='white', ha='center', fontsize=10, fontweight='bold')

ax3.text(0.5, -0.2, "8 normal bits can only hold 1 pattern at a time.\n8 Qubits hold 256 patterns AT THE SAME TIME.\nThis 'Entangled Web' lets QML instantly see how\na 0.5% shift in Opacity links to a tiny fever spike.", 
         transform=ax3.transAxes, color='lightgray', ha='center', fontsize=10, bbox=dict(facecolor='#1e293b', alpha=0.8, edgecolor='none'))

plt.tight_layout()
out_path = r"C:\Users\shubham dixit\.gemini\antigravity-ide\brain\dedeba13-1582-4504-b4ba-569de608b834\deep_qml_eda.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'Deep EDA Saved successfully to {out_path}')
