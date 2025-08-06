# ============================
# 1. Import necessary libraries
# ============================
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# ============================
# 2. Load and preprocess traffic data
# ============================
def load_traffic_data(csv_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    df = df.dropna()  # Remove missing values

    # Convert TIMESTAMP to datetime format
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

    return df

# Load your dataset
csv_path = "/Users/anmolpreetsingh/Desktop/trafficData/trafficData.csv"  # Change if needed
df = load_traffic_data(csv_path)

# ============================
# 3. Take a snapshot at a single timestamp
# ============================
# Select the earliest timestamp
selected_time = df['TIMESTAMP'].min()
snapshot = df[df['TIMESTAMP'] == selected_time]

# Sort by sensor ID for consistency
snapshot = snapshot.sort_values('extID')

# ============================
# 4. Extract features and labels
# ============================
# Use 3 numerical features: avgMeasuredTime, avgSpeed, medianMeasuredTime
node_features = snapshot[['avgMeasuredTime', 'avgSpeed', 'medianMeasuredTime']].values
x = torch.tensor(node_features, dtype=torch.float)

# Target to predict: vehicleCount
y = torch.tensor(snapshot['vehicleCount'].values, dtype=torch.float)

# ============================
# 5. Generate synthetic edges (each node connects to 3 others)
# ============================
num_nodes = x.shape[0]
edge_index = []

k = min(3, num_nodes - 1)  # 3 neighbors per node (or fewer if <4 nodes)

for i in range(num_nodes):
    neighbors = np.random.choice([j for j in range(num_nodes) if j != i], size=k, replace=False)
    for j in neighbors:
        edge_index.append([i, j])

# Convert edge list to tensor format (2, num_edges)
edge_index_list = []

if num_nodes > 1:
    k = min(3, num_nodes - 1)
    for i in range(num_nodes):
        neighbors = np.random.choice([j for j in range(num_nodes) if j != i], size=k, replace=False)
        for j in neighbors:
            edge_index_list.append([i, j])
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
else:
    print("Warning: Not enough nodes to create edges.")
    edge_index = torch.empty((2, 0), dtype=torch.long)

#edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# ============================
# 6. Wrap into a PyTorch Geometric data object
# ============================
data = Data(x=x, edge_index=edge_index, y=y)

# ============================
# 7. Define the GCN model
# ============================
class TrafficGCN(torch.nn.Module):
    def __init__(self):
        super(TrafficGCN, self).__init__()
        self.conv1 = GCNConv(3, 16)  # 3 input features
        self.conv2 = GCNConv(16, 1)  # Predict 1 output per node

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x.squeeze()  # Remove extra dimension from output

# ============================
# 8. Train the model
# ============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TrafficGCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(201):
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out, data.y)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# ============================
# 9. Done! Model can now predict vehicleCount from graph structure
# ============================
model.eval()
with torch.no_grad():
    predictions = model(data)
    print("\nSample predictions:")
    print(predictions if predictions.dim() > 0 else predictions.unsqueeze(0))

