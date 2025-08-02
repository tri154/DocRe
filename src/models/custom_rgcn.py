import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv

class CustomRGCN(nn.Module):
    """
    Example usage:
    x = torch.randn(4, 128)
    node_type = torch.tensor([0, 0, 1, 2])
    edge_index = torch.tensor([
        [0, 1, 0, 1, 3],
        [3, 3, 2, 2, 3]
        ])

    edge_type = torch.tensor([0, 0, 1, 1, 2])
    model = RGCN(128, 128, 128, 3, 128)

    out = model(x, node_type, edge_index, edge_type)
    """

    def __init__(self, in_dim, hidden_dim, num_relations=6, num_node_type=3, type_dim=20, low_layers=1, high_layers=2, num_bases=0):
        super(CustomRGCN, self).__init__()
        self.activation = nn.ReLU()
        self.low_layers = low_layers
        self.high_layers = high_layers
        self.num_node_type = num_node_type
        self.num_relations = num_relations
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        if num_bases == 0:
            self.num_bases = None
        else:
            self.num_bases = num_bases

        self.node_emb = torch.nn.Embedding(num_node_type, type_dim)

        self.convs = nn.ModuleList()
        for layer in range(low_layers + high_layers):
            input_dim = in_dim + type_dim if layer == 0 else hidden_dim
            self.convs.append(RGCNConv(input_dim, hidden_dim, num_relations, num_bases=self.num_bases))
           

    def forward(self, embs, node_type, edges):
        """
        x: Node feature matrix [num_nodes, in_channels]
        edge_index: Graph connectivity [2, num_edges]
        edge_type: Edge type labels [num_edges]
        """
        device = embs.device
        output = list()
        x = torch.cat([embs, self.node_emb(node_type)], dim=-1).to(device)

        #low layers
        edge_index = torch.cat(edges[:-1], dim=-1)
        edge_type = torch.arange(len(edges[:-1]), device=device).repeat_interleave(torch.tensor([ts.shape[-1] for ts in edges[:-1]], device=device))
        for idx in range(self.low_layers):
            conv = self.convs[idx]
            x = self.activation(conv(x, edge_index, edge_type))
            output.append(x)

        #high layers
        x = output[-1]
        temp = [edges[1], edges[5]]
        edge_type = torch.tensor([1, 5], device=device).repeat_interleave(torch.tensor([ts.shape[-1] for ts in temp], device=device))
        edge_index = torch.cat(temp, dim=-1).to(device)
        for idx in range(self.low_layers, self.low_layers + self.high_layers):
            conv = self.convs[idx]
            x = self.activation(conv(x, edge_index, edge_type))
            output.append(x)

        return [output[0], output[-1]]
