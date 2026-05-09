import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the shape functions N1 to N8 as functions of xi, eta, zeta
def N1(xi, eta, zeta): return 1/8 * (1 - xi) * (1 - eta) * (1 - zeta)
def N2(xi, eta, zeta): return 1/8 * (1 + xi) * (1 - eta) * (1 - zeta)
def N3(xi, eta, zeta): return 1/8 * (1 + xi) * (1 + eta) * (1 - zeta)
def N4(xi, eta, zeta): return 1/8 * (1 - xi) * (1 + eta) * (1 - zeta)
def N5(xi, eta, zeta): return 1/8 * (1 - xi) * (1 - eta) * (1 + zeta)
def N6(xi, eta, zeta): return 1/8 * (1 + xi) * (1 - eta) * (1 + zeta)
def N7(xi, eta, zeta): return 1/8 * (1 + xi) * (1 + eta) * (1 + zeta)
def N8(xi, eta, zeta): return 1/8 * (1 - xi) * (1 + eta) * (1 + zeta)

# List of shape functions and titles for plotting
shape_functions = [N1, N2, N3, N4, N5, N6, N7, N8]
titles = [f"N{i}" for i in range(1, 9)]

# Define a grid in the (xi, eta) space for zeta=0 as a slice
xi = np.linspace(-1, 1, 20)
eta = np.linspace(-1, 1, 20)
XI, ETA = np.meshgrid(xi, eta)
zeta = 0  # Set zeta to 0 to visualize the mid-plane

# Plot each shape function for the hexahedral element
fig = plt.figure(figsize=(16, 10))
for i, shape_func in enumerate(shape_functions):
    ax = fig.add_subplot(2, 4, i + 1, projection='3d')
    Z = shape_func(XI, ETA, zeta)  # Evaluate shape function on this slice
    ax.plot_surface(XI, ETA, Z, cmap='viridis', edgecolor='k', rstride=1, cstride=1, alpha=0.75)
    ax.set_title(titles[i])
    ax.set_xlabel(r'$\xi$')
    ax.set_ylabel(r'$\eta$')
    ax.set_zlabel('N')
    ax.set_zlim(0, 1)  # Shape functions vary from 0 to 1 in a single element

plt.tight_layout()
plt.savefig('Shape_function')
#plt.show()
