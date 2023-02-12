import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolutionLayer(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907, add trainable mask.
    """

    def __init__(self, in_features, out_features, num_nodes, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        # Add mask here, as trainable parameters.
        self.mask = Parameter(torch.ones(num_nodes, num_nodes))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.einsum('bnc,cd -> bnd', input, self.weight)
        output = torch.einsum('mn,bnd -> bmd', adj * self.mask, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, num_nodes, dropout, bias=True):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolutionLayer(nfeat, nhid, num_nodes, bias=bias)
        self.gc2 = GraphConvolutionLayer(nhid, nout, num_nodes, bias=bias)
        self.dropout = dropout

    def forward(self, x, adj):
        """
        x in shape [batch_size, num_node, feat_dim]
        adj in shape [num_node, num_node]
        """
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return self.gc2(x, adj)


class Graph_ConvGRUCell(nn.Module):

    def __init__(self, input_dim, gnn_hidden_dim, rnn_hidden_dim, num_nodes, gnn_bias, dropout):
        """
        input_dim: int
            Number of channels of input tensor.
        gnn_hidden_dim: int
            Number of channels of hidden state of GNN.
        rnn_hidden_dim: int
            Number of channels of hidden state of GRU.
        num_nodes: int
            Number of graph nodes.
        gnn_bias: bool
            Whether or not to add the bias for GCN.
        dropout: float
            Dropout probability for GNN.
        """

        super(Graph_ConvGRUCell, self).__init__()
        self.input_dim = input_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_nodes = num_nodes
        self.gnn_bias = gnn_bias
        self.dropout = dropout

        self.graph_net_1 = GCN(nfeat=self.input_dim + self.rnn_hidden_dim,
                               nhid=self.gnn_hidden_dim,
                               nout=2 * self.rnn_hidden_dim,
                               num_nodes=self.num_nodes,
                               dropout=self.dropout,
                               bias=self.gnn_bias)

        self.graph_net_2 = GCN(nfeat=self.input_dim + self.rnn_hidden_dim,
                               nhid=self.gnn_hidden_dim,
                               nout=self.rnn_hidden_dim,
                               num_nodes=self.num_nodes,
                               dropout=self.dropout,
                               bias=self.gnn_bias)

    def forward(self, input_tensor, cur_state, adj):
        h_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=-1)  # concatenate along channel axis

        combined_conv = self.graph_net_1(combined, adj)  # B, N, Cout

        gamma, beta = torch.split(combined_conv, self.rnn_hidden_dim, dim=-1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=-1)
        cc_ct = self.graph_net_2(combined, adj)
        ct = torch.tanh(cc_ct)

        h_next = (1 - update_gate) * h_cur + update_gate * ct
        return h_next

    def init_hidden(self, batch_size, num_nodes, device=torch.device("cuda")):  # todo: device
        return torch.zeros(batch_size, num_nodes, self.rnn_hidden_dim, device=device)


class Graph_ConvRNN(nn.Module):
    """
        input_dim: Number of channels in input
        num_layers: Number of RNN layers stacked on each other
        gnn_hidden_dim: Number of hidden channels for GNN
        rnn_hidden_dim: Number of hidden channels for RNN
        num_nodes: Number of graph nodes
        gnn_bias: Bias or no bias in GNN
        gnn_dropout: dropout probability in GNN
        return_all_layers: Return the list of computations for all layers
        batch_first: Whether or not dimension 0 is the batch or not
    """

    def __init__(self, input_dim, num_layers, gnn_hidden_dim, rnn_hidden_dim, num_nodes,
                 gnn_bias=True, gnn_dropout=0.5, batch_first=True, return_all_layers=True):
        super(Graph_ConvRNN, self).__init__()

        gnn_hidden_dim = self._extend_for_multilayer(gnn_hidden_dim, num_layers)
        rnn_hidden_dim = self._extend_for_multilayer(rnn_hidden_dim, num_layers)
        if not len(gnn_hidden_dim) == len(rnn_hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.gnn_hidden_dim = gnn_hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_nodes = num_nodes
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers
        self.gnn_dropout = gnn_dropout
        self.gnn_bias = gnn_bias

        cell = Graph_ConvGRUCell

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.rnn_hidden_dim[i - 1]

            cell_list.append(cell(input_dim=cur_input_dim,
                                  rnn_hidden_dim=self.rnn_hidden_dim[i],
                                  gnn_hidden_dim=self.gnn_hidden_dim[i],
                                  num_nodes=self.num_nodes,
                                  dropout=self.gnn_dropout,
                                  gnn_bias=self.gnn_bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, adj, init_state=None):
        """
        input_tensor: 4-D Tensor either of shape [B, T, N, C] or [T, B, N, C]
        adj: adjacency matrix， 2-D Tensor of shape (N, N)
        init_state: initial hidden state for RNN
        """
        if not self.batch_first:
            # (t, b, n, c) -> (b, t, n, c)
            input_tensor = input_tensor.permute(1, 0, 2, 3)

        b, _, n, _ = input_tensor.size()
        if not n == adj.size()[0] == self.num_nodes:
            raise ValueError('Inconsistent number of nodes in graph.')
        if init_state is None:
            init_state = self._init_hidden(batch_size=b, num_nodes=n)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            hidden_state = init_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                state_list = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :],
                                                       cur_state=hidden_state,
                                                       adj=adj)
                output_inner.append(state_list)
                hidden_state = state_list

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(state_list)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, num_nodes):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, num_nodes))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class GGCNet(nn.Module):

    def __init__(self):
        super(GGCNet, self).__init__()
        self.gcrnn = Graph_ConvRNN(input_dim=4, num_layers=4, gnn_hidden_dim=[32, 32, 16, 16],
                                   rnn_hidden_dim=[8, 8, 4, 4], num_nodes=14, gnn_bias=True, gnn_dropout=0.5,
                                   batch_first=True, return_all_layers=True)  # [56, 28, 14];[8, 16, 4]
        self.fc = nn.Sequential(nn.Linear(30*14*24, 100), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.5)  # 0.2
        self.fc1 = nn.Sequential(nn.Linear(100, 1))  # outputchannel=1
        self.fc2 = nn.Sequential(nn.Linear(100, 1))
        self.fc3 = nn.Sequential(nn.Linear(100, 1))
        self.fc_std = nn.Sequential(nn.Linear(100, 1), nn.Softplus())

    def forward(self, input_tensor, adj):
        layer_output_list, _ = self.gcrnn(input_tensor, adj)
        layer_output_list = torch.cat(layer_output_list, dim=3)
        flatten_h = layer_output_list.view(layer_output_list.size(0), -1)
        x = self.fc(flatten_h)
        x = self.dropout(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        return x1, x2, x3
