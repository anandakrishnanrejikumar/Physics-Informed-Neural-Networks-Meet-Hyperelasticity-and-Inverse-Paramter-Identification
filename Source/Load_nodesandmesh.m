% Load node coordinates and element connectivity data
nodes = load('nodes.txt');  % Each row is [X, Y, Z] for a node
connectivity = load('connectivity.txt');  % Each row has 8 global node indices for an element

% Create a new figure for 3D visualization
figure;
hold on;
axis equal;
xlabel('X');
ylabel('Y');
zlabel('Z');
title('3D Plot of Mesh Elements with Global Node Ordering');

% Plot each node with its global node index
scatter3(nodes(:,1), nodes(:,2), nodes(:,3), 'filled', 'MarkerFaceColor', 'blue');
for idx = 1:size(nodes, 1)
    text(nodes(idx, 1), nodes(idx, 2), nodes(idx, 3), sprintf('%d', idx-1), 'Color', 'red', 'FontSize', 8);
end

% Loop through each element in the connectivity matrix
for i = 1:size(connectivity, 1)
    % Get the global indices for the nodes of the current element
    element_nodes = connectivity(i, :);
    
    % Retrieve the coordinates for each vertex of the element
    element_coords = nodes(element_nodes + 1, :);  % MATLAB indexing starts at 1

    % Define the vertex ordering to create a closed polyhedron
    verts = element_coords([1 2 3 4 1 5 6 7 8 5 6 2 3 7 8 4], :);

    % Plot each face of the element
    fill3(verts(:, 1), verts(:, 2), verts(:, 3), 'cyan', 'FaceAlpha', 0.3, 'EdgeColor', 'k');
end

hold off;
