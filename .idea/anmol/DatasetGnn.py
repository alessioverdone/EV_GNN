# Import libraries for data handling, deep learning, and GNNs
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np

# 1. Load and preprocess a small part of the dataset
def load_traffic_data(folder_path):
    # Get list of CSV files in the given folder
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    dfs = []
    for file in files[:10]:  # Load only first 10 files for demo (can increase later)
        df = pd.read_csv(file)
        dfs.append(df)
    data = pd.concat(dfs)  # Combine all files into one big dataframe
    return data

# Instead of folder loading, read a single CSV directly
folder_path = "/Users/anmolpreetsingh/Desktop/trafficData/"  # Provide path to your downloaded CSV
traffic_data = load_traffic_data(folder_path)

print("Loaded data:", traffic_data.shape)  # Print how much data was loaded

# 2. Basic preprocessing
traffic_data = traffic_data.dropna()  # Remove any rows with missing data
traffic_data['TIMESTAMP'] = pd.to_datetime(traffic_data['TIMESTAMP'])  # Convert timestamp to datetime
traffic_data = traffic_data.sort_values(by='TIMESTAMP')  # Sort data by time

# Take one snapshot (a single timestamp) to create a graph
snapshot = traffic_data[traffic_data['TIMESTAMP'] == traffic_data['TIMESTAMP'].iloc[0]]

# 3. Create node features
# Pick avgSpeed and vehicleCount as features for each sensor (node)
node_features = snapshot[['avgSpeed', 'vehicleCount']].values
x = torch.tensor(node_features, dtype=torch.float)  # Convert to PyTorch tensor

# 4. Build safe edges between nodes (so that GNN can learn)
num_nodes = x.shape[0]
edge_index = []

if num_nodes > 1:  # Only if there are at least 2 nodes
    k = min(3, num_nodes - 1)  # Each node will connect to up to 3 neighbors

    for i in range(num_nodes):
        neighbors = np.random.choice(
            [j for j in range(num_nodes) if j != i],  # Pick neighbors (no self-connections)
            size=k,
            replace=False
        )
        for j in neighbors:
            edge_index.append([i, j])  # Create directed edges

    # Convert edge list to tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
else:
    print("Not enough nodes to create edges.")  # Safe fallback
    edge_index = torch.empty((2, 0), dtype=torch.long)

print("Edge index shape:", edge_index.shape)  # Check how many edges were created

#----------------------------- we are trying to preditct vehicle count at each node ---------------#
# 5. Define target output
# What we are trying to predict: vehicle count at each node
y = torch.tensor(snapshot['vehicleCount'].values, dtype=torch.float)

# 6. Create a PyTorch Geometric Data object
# Wrap features, edges, and labels into one graph object
data = Data(x=x, edge_index=edge_index, y=y)

# 7. Define the GCN Model
class TrafficGCN(torch.nn.Module):
    def __init__(self):
        super(TrafficGCN, self).__init__()
        # First Graph Convolution layer: input 2 features -> 16 hidden features
        self.conv1 = GCNConv(2, 16)
        # Second Graph Convolution layer: 16 hidden features -> 1 output (vehicle count prediction)
        self.conv2 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index  # Extract node features and edge list
        x = self.conv1(x, edge_index)  # First GCN layer
        x = F.relu(x)  # Activation function
        x = self.conv2(x, edge_index)  # Second GCN layer
        return x.squeeze()  # Remove extra dimensions for output

# 8. Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
model = TrafficGCN().to(device)  # Move model to device
data = data.to(device)  # Move data to device
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Optimizer setup

model.train()  # Set model to training mode
for epoch in range(201):  # Train for 200 epochs
    optimizer.zero_grad()  # Reset gradients
    out = model(data)  # Forward pass (predict outputs)
    loss = F.mse_loss(out, data.y)  # Calculate Mean Squared Error loss
    loss.backward()  # Backpropagation (compute gradients)
    optimizer.step()  # Update model parameters

    if epoch % 20 == 0:
        print(f'Epoch {epoch} | Loss: {loss.item():.4f}')  # Print loss every 20 epochs

# 9. Done! 🚀
# After training, the model can predict vehicle counts at each sensor based on graph structure + features.
