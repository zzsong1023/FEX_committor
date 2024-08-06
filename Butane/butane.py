import numpy as np
import matplotlib.pyplot as plt

# Define the function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the range for x0 which is cos(\Phi), where \Phi is the dihedral angle of butane
phi = np.linspace(0, 2 * np.pi, 1000)
x0 = np.cos(phi)

# Compute the y values
y = sigmoid(16.7625 * (x0)**3 + 2.6791)
y = sigmoid(10.0278 * x0 + 4.209)

fs = 20

# Plot the function
plt.figure(figsize=(10, 7))
plt.plot(phi, y, label='committor')
plt.xlabel(r'Dihedral angle $\Phi$ (degrees)', fontsize=fs)
plt.ylabel('Committer', fontsize=fs)
plt.title('Committer as a Function of Dihedral Angle $\Phi$', fontsize=fs)
plt.legend(fontsize=fs)
plt.grid(True)

# Set x ticks to degrees
deg_ticks = np.linspace(0, 2 * np.pi, 9)
deg_labels = np.linspace(0, 360, 9).astype(int)
plt.xticks(deg_ticks, deg_labels, fontsize=fs)
plt.yticks(fontsize=fs)

# Annotate 'A' at 180 degrees and 'B' at 60 and 300 degrees
phi_60 = np.pi / 3
phi_180 = np.pi
phi_300 = 5 * np.pi / 3


y = sigmoid(10.0278 * x0 + 4.209)

y_60 = sigmoid(10.0278 * (np.cos(phi_60)) +  4.209)
y_180 = sigmoid(10.0278 * (np.cos(phi_180)) +  4.209)
y_300 = sigmoid(10.0278 * (np.cos(phi_300)) +  4.209)

plt.text(phi_60, y_60 - 0.1, 'B', fontsize=fs, ha='center')
plt.text(phi_180, y_180 + 0.05, 'A', fontsize=fs, ha='center')
plt.text(phi_300, y_300 - 0.1, 'B', fontsize=fs, ha='center')

# Save the figure
plt.savefig('butane_committor.pdf', dpi=300)

plt.show()