import torch
from torch import nn
from torch_geometric.nn import MessagePassing


class CustomTransformer(nn.Module):
    def __init__(self, feat_dim, nhead, num_encoder_layers, dim_feedforward, dropout, first_seq=1, second_seq=1):
        super(CustomTransformer, self).__init__()
        self.seq_len = first_seq + second_seq + 2
        self.first_seq = first_seq
        self.second_seq = second_seq

        encoder_layer = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers,
                                                         enable_nested_tensor=False)

        self.cls_token_param = nn.Parameter(torch.ones(1, 1, feat_dim))
        self.sep_token_param = nn.Parameter(torch.zeros(1, 1, feat_dim))

        self.pos_param = nn.Parameter(torch.zeros(1, self.seq_len, feat_dim))

    def forward(self, *x):
        first_seq = [x.unsqueeze(1) for x in x[:self.first_seq]]  # (Be, 1, H)
        second_seq = [x.unsqueeze(1) for x in x[self.first_seq:]]  # (Be, 1, H)

        cls_token = self.cls_token_param.expand(first_seq[0].size(0), -1, -1)
        sep_token = self.sep_token_param.expand(first_seq[0].size(0), -1, -1)

        x = torch.cat([cls_token] + first_seq + [sep_token] + second_seq, dim=1)
        x += self.pos_param

        x = self.transformer_encoder(x)
        return x[:, 0, :]


class GNNTransformModel(MessagePassing):
    def __init__(self, num_node_features, num_edge_features,
                 output_node_features=None, output_edge_features=None,
                 dropout_rate=.1, hid_dim=128):
        super(GNNTransformModel, self).__init__(aggr='add')

        self.node_encoder = nn.Sequential(
            nn.Linear(num_node_features, hid_dim),
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(num_edge_features, hid_dim),
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim),
        )

        self.node_decoder = nn.Linear(hid_dim, output_node_features or num_node_features)
        self.edge_decoder = nn.Linear(hid_dim, output_edge_features or num_edge_features)

        self.node_message_passing = CustomTransformer(
            hid_dim, 4, 4, hid_dim, dropout_rate,
            first_seq=1
        )
        self.edge_message_passing = CustomTransformer(
            hid_dim, 4, 4, hid_dim, dropout_rate,
            first_seq=2
        )


    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)  # (Bn, H)
        edge_attr = self.edge_encoder(edge_attr) if len(edge_attr) > 0 else edge_attr  # (Be, H)

        # Handling graphs with no edges
        if len(edge_index) > 0 and edge_index.shape[1] > 0:
            x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            edge_attr = self.edge_updater(edge_index, x=x, edge_attr=edge_attr) if len(edge_attr) > 0 else edge_attr

        out_node_features = self.node_decoder(x)
        out_edge_features = self.edge_decoder(edge_attr) if len(edge_attr) > 0 else edge_attr

        return out_node_features, out_edge_features

    def message(self, x_j, edge_attr):
        return self.node_message_passing(x_j, edge_attr)

    def edge_update(self, x_i, x_j, edge_attr):
        return self.edge_message_passing(x_i, x_j, edge_attr)

    def update(self, aggr_out):
        return aggr_out
