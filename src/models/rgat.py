import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class RGAT(nn.Module):
    """
    A Graph Attention Network (GAT) module operated on graphs.

    Example usage:
    x = torch.randn(4, 128)
    node_type = torch.tensor([0, 0, 1, 2]) # GAT typically doesn't directly use edge_type like RGCN
                                          # but node_type can be concatenated to features.
    edge_index = torch.tensor([
        [0, 1, 0, 1, 3],
        [3, 3, 2, 2, 3]
        ])

    model = GAT(128, 128, num_node_type=3, type_dim=20, num_layers=2, heads=8)

    out = model(x, node_type, edge_index)
    """

    def __init__(self, in_dim, hidden_dim , num_node_type=3, type_dim=20, num_layers=2, heads=8, dropout=0.2):
        super(RGAT, self).__init__()
        self.activation = nn.ELU() # GAT often uses ELU activation hidden_dim
        self.num_layers = num_layers
        self.num_node_type = num_node_type
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim 
        self.heads = heads # Number of attention heads
        self.dropout = dropout

        self.node_emb = torch.nn.Embedding(num_node_type, type_dim)

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim + type_dim, hidden_dim, heads=heads, dropout=dropout, add_self_loops=True))
        
        for i in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, add_self_loops=True))
        
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout, add_self_loops=True))
        else: # If num_layers is 1, the first layer is also the output layer
            self.convs = nn.ModuleList([GATConv(in_dim + type_dim, hidden_dim, heads=1, concat=False, dropout=dropout, add_self_loops=True)])


    def forward(self, x, node_type, edge_index, edge_type):
        """
        x: Node feature matrix [num_nodes, in_channels]
        node_type: Node type labels [num_nodes] (used for embedding)
        edge_index: Graph connectivity [2, num_edges]
        """
        x = torch.cat([x, self.node_emb(node_type)], dim=-1)
        
        output = list()

        for i, conv in enumerate(self.convs):
            if i == len(self.convs) - 1:
                x = conv(x, edge_index)
            else:
                x = self.activation(conv(x, edge_index))
            if i == 0: 
                output.append(x)
        
        if self.num_layers != 1:
            output.append(x)

        return output
