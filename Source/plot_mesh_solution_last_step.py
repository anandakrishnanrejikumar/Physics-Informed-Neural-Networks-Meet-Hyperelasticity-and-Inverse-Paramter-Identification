import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Load nodes and connectivity data from text files
nodes = np.loadtxt('nodes.txt', delimiter=',')  # Each row contains X, Y, Z coordinates
connectivity = np.loadtxt('connectivity.txt', delimiter=',', dtype=int)  # Adjust to zero-based indexing
displacement_500000 = np.loadtxt('displacement_0.500000.txt', delimiter=',', dtype=float)

# Function to fetch coordinates of nodes for each element based on connectivity
def get_element_coordinates(nodes, connectivity):
    element_coordinates = []
    displacement_coordinates = []
    for element in connectivity:
        # For each element, retrieve the node coordinates based on connectivity indices
        element_nodes = nodes[element]  # Fetch rows from nodes for this element
        displacement_constraint = displacement_500000[element]
        displacement_constraint_trimmed = displacement_constraint[:,6:10]
        element_coordinates.append(element_nodes)
        displacement_coordinates.append(displacement_constraint_trimmed)
    return element_coordinates, displacement_coordinates

# Get coordinates for each element
elements_with_coordinates,displacement_with_coordinates = get_element_coordinates(nodes, connectivity)

# Plot each element as a wireframe box in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for idx, element_coords in enumerate(elements_with_coordinates):
    # Define vertices for each element according to the specified order
    vertices = [
        [element_coords[0], element_coords[1], element_coords[3], element_coords[2]],  # bottom face
        [element_coords[4], element_coords[5], element_coords[7], element_coords[6]],  # top face
        [element_coords[0], element_coords[4], element_coords[6], element_coords[2]],  # side face (left)
        [element_coords[1], element_coords[5], element_coords[7], element_coords[3]],  # side face (right)
        [element_coords[0], element_coords[1], element_coords[5], element_coords[4]],  # front face
        [element_coords[2], element_coords[3], element_coords[7], element_coords[6]]  # back face
    ]


    # Create a Poly3DCollection for each element
    poly = Poly3DCollection(vertices, alpha=0.25, edgecolor='k')
    ax.add_collection3d(poly)

    # # Annotate nodes within each element for clarity
    # for i, coord in enumerate(element_coords):
    #     ax.text(*coord, f"{i}", color='red')  # Label nodes from 0 to 7

# Scatter plot of all nodes
ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], color='blue', marker='o')

# Set axis labels and title
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.title('3D Hexahedral Element Visualization with Labeled Nodes')

plt.savefig('Mesh_Elements')

