import numpy as np
import plotly.graph_objects as go

# Define the 8 vertices of a cube in 3D space
cube = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]
])

# Create a list of edges to build the cube
edges = [(0, 1), (1, 2), (2, 3), (3, 0),  # edges on the bottom face
         (4, 5), (5, 6), (6, 7), (7, 4),  # edges on the top face
         (0, 4), (1, 5), (2, 6), (3, 7)]  # edges on the sides

# Define a 2x3 transformation matrix
A = np.array([[1, 2, 1], [2, 1, 0]])

# Apply the transformation
transformed_cube = np.dot(A, cube.T).T

# Create a 3D plot for the original cube
fig1 = go.Figure()

for edge in edges:
    x0, y0, z0 = cube[edge[0]]
    x1, y1, z1 = cube[edge[1]]
    fig1.add_trace(go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1], mode='lines'))

fig1.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                   width=700, margin=dict(r=20, l=10, b=10, t=10))

fig1.show()

# Create a 2D plot for the transformed cube
fig2 = go.Figure()

for edge in edges:
    x0, y0 = transformed_cube[edge[0]]
    x1, y1 = transformed_cube[edge[1]]
    fig2.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines'))

fig2.update_layout(xaxis_title='X', yaxis_title='Y',
                   width=700, margin=dict(r=20, l=10, b=10, t=10))

fig2.show()



import numpy as np

a = np.array([[1,2,1],[2,0,1]])

U,S,V_t = np.linalg.svd(a)

S_full = np.zeros((U.shape[1], V_t.shape[0]))  # Create a 3x2 matrix filled with zeros
S_full[:S.shape[0], :S.shape[0]] = np.diag(S)

S_full[:,[0,1]]






import numpy as np
import plotly.graph_objects as go

# define eight vertices for the unit cube
vertices = np.array([
    [-2.5, -2.5, -2.5],
    [2.5, -2.5, -2.5],
    [2.5, 2.5, -2.5],
    [-2.5, 2.5, -2.5],
    [-2.5, -2.5, 2.5],
    [2.5, -2.5, 2.5],
    [2.5, 2.5, 2.5],
    [-2.5, 2.5, 2.5]
])

# define the twelve edges of the cube
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # edges in the bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # edges in the top face
    (0, 4), (1, 5), (2, 6), (3, 7)   # edges connecting top and bottom faces
]

# create a plotly graph object
fig = go.Figure()

# add each edge to the graph
for edge in edges:
    x_values = [vertices[edge[0], 0], vertices[edge[1], 0]]
    y_values = [vertices[edge[0], 1], vertices[edge[1], 1]]
    z_values = [vertices[edge[0], 2], vertices[edge[1], 2]]
    fig.add_trace(go.Scatter3d(x=x_values, y=y_values, z=z_values, mode='lines', line=dict(color='black')))

# Draw axis lines
for i, color in enumerate(['red', 'green', 'blue']):  # i=0 is X, i=1 is Y, i=2 is Z
    axis = np.zeros((2, 3))
    axis[1, i] = 1
    axis_line = axis * 5  # scale to the desired length
    fig.add_trace(go.Scatter3d(x=axis_line[:, 0], y=axis_line[:, 1], z=axis_line[:, 2], mode='lines', line=dict(color=color)))

fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), autosize=False, width=500, height=500,
                  margin=dict(l=50, r=50, b=100, t=100, pad=4))

fig.show()


# Transformation matrix
T = np.array([
    [-0.70341305, -0.5221579 , -0.482246  ],
    [ 0.56101149, -0.82445866,  0.07439108],
    [-0.43643578, -0.21821789,  0.87287156]
])

# create a plotly graph object
fig = go.Figure()

# Add each edge of the original cube to the graph
for edge in edges:
    x_values = [vertices[edge[0], 0], vertices[edge[1], 0]]
    y_values = [vertices[edge[0], 1], vertices[edge[1], 1]]
    z_values = [vertices[edge[0], 2], vertices[edge[1], 2]]
    fig.add_trace(go.Scatter3d(x=x_values, y=y_values, z=z_values, mode='lines', line=dict(color='black')))

# Apply transformation
vertices_transformed = np.dot(vertices, T.T)

# Add each edge of the transformed cube to the graph
for edge in edges:
    x_values = [vertices_transformed[edge[0], 0], vertices_transformed[edge[1], 0]]
    y_values = [vertices_transformed[edge[0], 1], vertices_transformed[edge[1], 1]]
    z_values = [vertices_transformed[edge[0], 2], vertices_transformed[edge[1], 2]]
    fig.add_trace(go.Scatter3d(x=x_values, y=y_values, z=z_values, mode='lines', line=dict(color='blue')))

# Draw axis lines
for i, color in enumerate(['red', 'green', 'blue']):  # i=0 is X, i=1 is Y, i=2 is Z
    axis = np.zeros((2, 3))
    axis[1, i] = 1
    axis_line = axis * 5  # scale to the desired length
    fig.add_trace(go.Scatter3d(x=axis_line[:, 0], y=axis_line[:, 1], z=axis_line[:, 2], mode='lines', line=dict(color=color)))

fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), autosize=False, width=500, height=500,
                  margin=dict(l=50, r=50, b=100, t=100, pad=4))

fig.show()


# Transformation matrix for projection onto X-Y plane
P = np.array([
    [1, 0, 0],
    [0, 1, 0]
])

# Apply transformation
vertices_transformed_2d = np.dot(vertices_transformed, P.T)

# create a plotly graph object for 2D
fig2d = go.Figure()

# Add each edge of the transformed cube to the graph
for edge in edges:
    x_values = [vertices_transformed_2d[edge[0], 0], vertices_transformed_2d[edge[1], 0]]
    y_values = [vertices_transformed_2d[edge[0], 1], vertices_transformed_2d[edge[1], 1]]
    fig2d.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', line=dict(color='blue')))

# Draw axis lines
axis = np.array([[-5, 5], [0, 0]])
fig2d.add_trace(go.Scatter(x=axis[0, :], y=axis[1, :], mode='lines', line=dict(color='red')))  # X-axis
axis = np.array([[0, 0], [-5, 5]])
fig2d.add_trace(go.Scatter(x=axis[0, :], y=axis[1, :], mode='lines', line=dict(color='green')))  # Y-axis

fig2d.update_layout(xaxis_title='X', yaxis_title='Y', autosize=False, width=500, height=500, margin=dict(l=50, r=50, b=100, t=100, pad=4))

fig2d.show()


# Transformation matrix for scaling
S = np.array([
    [2.92256416, 0],
    [0, 1.56799832]
])

# Apply transformation
vertices_transformed_2d_scaled = np.dot(vertices_transformed_2d, S.T)

# create a new plotly graph object for 2D
fig2d_scaled = go.Figure()

# Add each edge of the transformed cube to the graph
for edge in edges:
    x_values = [vertices_transformed_2d_scaled[edge[0], 0], vertices_transformed_2d_scaled[edge[1], 0]]
    y_values = [vertices_transformed_2d_scaled[edge[0], 1], vertices_transformed_2d_scaled[edge[1], 1]]
    fig2d_scaled.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', line=dict(color='blue')))

# Draw axis lines
axis = np.array([[-15, 15], [0, 0]])
fig2d_scaled.add_trace(go.Scatter(x=axis[0, :], y=axis[1, :], mode='lines', line=dict(color='red')))  # X-axis
axis = np.array([[0, 0], [-15, 15]])
fig2d_scaled.add_trace(go.Scatter(x=axis[0, :], y=axis[1, :], mode='lines', line=dict(color='green')))  # Y-axis

fig2d_scaled.update_layout(xaxis_title='X', yaxis_title='Y', autosize=False, width=500, height=500, margin=dict(l=50, r=50, b=100, t=100, pad=4))

fig2d_scaled.show()


# Transformation matrix for rotation
R = np.array([
    [-0.76301998, -0.6463749],
    [-0.6463749, 0.76301998]
])

# Apply transformation
vertices_transformed_2d_scaled_rotated = np.dot(vertices_transformed_2d_scaled, R.T)

# create a new plotly graph object for 2D
fig2d_scaled_rotated = go.Figure()

# Add each edge of the transformed cube to the graph
for edge in edges:
    x_values = [vertices_transformed_2d_scaled_rotated[edge[0], 0], vertices_transformed_2d_scaled_rotated[edge[1], 0]]
    y_values = [vertices_transformed_2d_scaled_rotated[edge[0], 1], vertices_transformed_2d_scaled_rotated[edge[1], 1]]
    fig2d_scaled_rotated.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', line=dict(color='blue')))

# Draw axis lines
axis = np.array([[-15, 15], [0, 0]])
fig2d_scaled_rotated.add_trace(go.Scatter(x=axis[0, :], y=axis[1, :], mode='lines', line=dict(color='red')))  # X-axis
axis = np.array([[0, 0], [-15, 15]])
fig2d_scaled_rotated.add_trace(go.Scatter(x=axis[0, :], y=axis[1, :], mode='lines', line=dict(color='green')))  # Y-axis

fig2d_scaled_rotated.update_layout(xaxis_title='X', yaxis_title='Y', autosize=False, width=500, height=500, margin=dict(l=50, r=50, b=100, t=100, pad=4))

fig2d_scaled_rotated.show()
