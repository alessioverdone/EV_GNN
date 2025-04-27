import numpy as np

# Define a toy graph with 4 nodes and their initial features
num_nodes = 4
num_features = 2
adjacency_matrix = np.array([[0, 1, 0, 1],
                             [1, 0, 1, 1],
                             [0, 1, 0, 0],
                             [1, 1, 0, 0]])  # Adjacency matrix

node_features = np.random.rand(num_nodes, num_features)  # Random node features

# Define a simple message passing function
def message_passing(adj_matrix, node_feats):
    updated_feats = np.zeros_like(node_feats)
    num_nodes = len(node_feats)
    
    # Iterate over each node
    for i in range(num_nodes):
        # Gather neighboring nodes based on adjacency matrix
        neighbors = np.where(adj_matrix[i] == 1)[0]
        
        # Aggregate messages from neighbors
        message = np.sum(node_feats[neighbors], axis=0)
        
        # Update node representation
        updated_feats[i] = node_feats[i] + message
    
    return updated_feats

num_iterations = 3
for i in range(num_iterations):
    updated_features = message_passing(adjacency_matrix, node_features)
     
# Perform message passing for one iteration
#updated_features = message_passing(adjacency_matrix, node_features)
print("Updated Node Features after Message Passing:")
print(updated_features)