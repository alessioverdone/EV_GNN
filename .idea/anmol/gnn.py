import torch 
from torch_geometric.data import Data 
from torch_geometric.utils import to_networkx 
import matplotlib.pyplot as plt 
import networkx as nx 

edge_index = torch.tensor([[0, 1], [1,2]], dtype = torch.long)
x = torch.tensor([[0], [1], [2]], dtype = torch.float)

data = Data(x = x, edge_index = edge_index.t().contiguous())

G = to_networkx(data)

nx.draw(G, with_labels = True)
plt.title("Graph Representation Using pytorch geometric")
plt.show()