import torch
import torch.nn as nn
from torch_geometric.nn.conv import SAGEConv


class NostrSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        return self.conv2(x, edge_index)
