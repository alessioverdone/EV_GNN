import networkx as nx 
import matplotlib.pyplot as plt 

social_network = nx.Graph()
nodes = [1,2,3,4,5]

social_network.add_nodes_from(nodes)

edges = [(1,2), (2,3), (3,4), (4,5)]
social_network.add_edges_from(edges)

pos = nx.spring_layout(social_network)
nx.draw(social_network, pos, with_labels = True, font_size = 12, node_color = 'red')
plt.title("GNN")
plt.show()


# nx draws and creates graph , the only function of matplotlib is to show that graph on screen 
