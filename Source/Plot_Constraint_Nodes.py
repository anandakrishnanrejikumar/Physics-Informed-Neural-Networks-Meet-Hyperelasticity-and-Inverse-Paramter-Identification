import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Load nodes and connectivity data
nodes = np.loadtxt('nodes.txt', delimiter=',')  # Each row contains X, Y, Z coordinates
connectivity = np.loadtxt('connectivity.txt', delimiter=',', dtype=int)  # Adjust to zero-based indexing
displacement_500000 = np.loadtxt('displacement_0.500000.txt', delimiter=',', dtype=float)

# Function to fetch coordinates of nodes and constraints for each element
def get_element_coordinates(nodes, connectivity, displacement_data):
    element_coordinates = []
    displacement_constraints = []
    for element in connectivity:
        element_nodes = nodes[element]  # Retrieve rows from nodes for this element
        element_constraints = displacement_data[element, 6:10]  # Extract constraints from columns 6 to 10
        element_coordinates.append(element_nodes)
        displacement_constraints.append(element_constraints)
    return element_coordinates, displacement_constraints

# Identify constrained nodes for specific directions
def find_constrained_nodes_by_direction(nodes, connectivity, displacement_with_coordinates):
    constrained_x = set()  # Nodes constrained in x-direction
    constrained_y = set()  # Nodes constrained in y-direction
    constrained_z = set()  # Nodes constrained in z-direction

    for element_idx, element_constraints in enumerate(displacement_with_coordinates):
        for node_idx, constraints in enumerate(element_constraints):
            global_node_idx = connectivity[element_idx][node_idx]  # Get global node index
            if constraints[0] == 1:  # Check if constrained in x
                constrained_x.add(global_node_idx)
            if constraints[1] == 1:  # Check if constrained in y
                constrained_y.add(global_node_idx)
            if constraints[2] == 1:  # Check if constrained in z
                constrained_z.add(global_node_idx)

    return list(constrained_x), list(constrained_y), list(constrained_z)

# Get element coordinates and constraints
elements_with_coordinates, displacement_with_coordinates = get_element_coordinates(
    nodes, connectivity, displacement_500000
)

# Find constrained nodes for each direction
constrained_x, constrained_y, constrained_z = find_constrained_nodes_by_direction(
    nodes, connectivity, displacement_with_coordinates
)

# Plot each element and highlight constrained nodes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for idx, element_coords in enumerate(elements_with_coordinates):
    # Define vertices for each element according to the specified order
    vertices = [
        [element_coords[0], element_coords[1], element_coords[3], element_coords[2]],  # Bottom face
        [element_coords[4], element_coords[5], element_coords[7], element_coords[6]],  # Top face
        [element_coords[0], element_coords[4], element_coords[6], element_coords[2]],  # Side face (left)
        [element_coords[1], element_coords[5], element_coords[7], element_coords[3]],  # Side face (right)
        [element_coords[0], element_coords[1], element_coords[5], element_coords[4]],  # Front face
        [element_coords[2], element_coords[3], element_coords[7], element_coords[6]],  # Back face
    ]

    # Create a Poly3DCollection for each element
    poly = Poly3DCollection(vertices, alpha=0.25, edgecolor='k')
    ax.add_collection3d(poly)

# Highlight constrained nodes in x, y, and z directions
offset = 0.002  # Offset to prevent marker overlap
legend_set = set()  # Track legend entries to avoid duplicates

if constrained_x:
    constrained_x_coords = nodes[constrained_x]
    ax.scatter(
        constrained_x_coords[:, 0] + offset,
        constrained_x_coords[:, 1],
        constrained_x_coords[:, 2],
        color='red',
        marker='^',
        label='Constrained in X' if 'Constrained in X' not in legend_set else None
    )
    legend_set.add('Constrained in X')

if constrained_y:
    constrained_y_coords = nodes[constrained_y]
    ax.scatter(
        constrained_y_coords[:, 0],
        constrained_y_coords[:, 1] + offset,
        constrained_y_coords[:, 2],
        color='green',
        marker='s',
        label='Constrained in Y' if 'Constrained in Y' not in legend_set else None
    )
    legend_set.add('Constrained in Y')

if constrained_z:
    constrained_z_coords = nodes[constrained_z]
    ax.scatter(
        constrained_z_coords[:, 0],
        constrained_z_coords[:, 1],
        constrained_z_coords[:, 2] + offset,
        color='purple',
        marker='*',
        label='Constrained in Z' if 'Constrained in Z' not in legend_set else None
    )
    legend_set.add('Constrained in Z')

# Set axis labels and title
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.title('3D Hexahedral Element Visualization with Directional Constraints')
plt.legend()

# Save the figure to a file
plt.savefig('Directional_Constraints.png', dpi=300, bbox_inches='tight')
