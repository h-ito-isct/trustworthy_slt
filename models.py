import torch.nn as nn
import torch.nn.functional as F
from utils.slt_modules import SLT_Conv2d, SLT_Linear
from utils.slt_modules import PartialFrozenConv2d, PartialFrozenLinear
from utils.mc_dropout import MC_Dropout
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from typing import Optional, Tuple, Union
from torch_geometric.typing import OptTensor, OptPairTensor, Adj
from torch import Tensor, nn
from torch_sparse import SparseTensor
from torch_geometric.utils import spmm
from torch_geometric.nn.inits import zeros
from torch.nn import Parameter
import torch
from torch_geometric.nn.dense.linear import Linear


class LeNet(nn.Module):
    def __init__(self, args):
        super(LeNet, self).__init__()

        self.args = args
        self.num_bayes_layers = args.num_bayes_layers

        # Scale hidden dimensions by scaling_rate
        conv1_out = int(20 * args.scaling_rate)
        conv2_out = int(20 * args.scaling_rate)
        fc1_out = int(100 * args.scaling_rate)
        fc2_out = int(10)

        # Set input channels based on dataset
        in_channels = 3 if args.dataset == "cifar10" else 1

        # Choose Conv2d implementation based on args
        if args.partial_frozen_slt:
            Conv2d = PartialFrozenConv2d
            Linear = PartialFrozenLinear
            conv_args = {'sparsity': args.pruning_rate[0], 'algo': 'global_ep', 'scale_method': 'dynamic_scaled'}
            linear_args = {'sparsity': args.pruning_rate[0], 'algo': 'global_ep', 'scale_method': 'dynamic_scaled'}
        elif args.slt:
            Conv2d = SLT_Conv2d
            Linear = SLT_Linear
            conv_args = {'args': args}
            linear_args = {'args': args}
        else:
            Conv2d = nn.Conv2d
            Linear = nn.Linear
            conv_args = {}
            linear_args = {}

        self.conv1 = Conv2d(in_channels, conv1_out, kernel_size=5, padding="same", **conv_args)
        self.conv2 = Conv2d(conv1_out, conv2_out, kernel_size=5, padding="same", **conv_args)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(7)
        self.softmax = nn.Softmax(dim=1)

        self.fc1_input_size = int(conv2_out * 2 * 2)

        self.fc1 = Linear(self.fc1_input_size, fc1_out, **linear_args)
        self.fc2 = Linear(fc1_out, fc2_out, **linear_args)

        self.dropout_layers = nn.ModuleList([
            MC_Dropout(p=args.dropout_rate) for _ in range(args.num_bayes_layers)
        ])

    def forward(self, x, threshold=None):
        if self.args.dataset == "mnist":
            x = F.pad(x, (2, 2, 2, 2))

        if self.args.partial_frozen_slt:
            x = self.conv1(x, threshold)
            x = self.relu1(x)
            if self.num_bayes_layers >= 3:
                x = self.dropout_layers[2](x)
            x = self.maxpool1(x)

            x = self.conv2(x, threshold)
            x = self.relu2(x)
            if self.num_bayes_layers >= 2:
                x = self.dropout_layers[1](x)
            x = self.maxpool2(x)

            x = x.view(x.size(0), -1)

            x = self.fc1(x, threshold)
            x = self.relu3(x)
            if self.num_bayes_layers >= 1:
                x = self.dropout_layers[0](x)

            x = self.fc2(x, threshold)
        else:
            x = self.conv1(x, threshold) if self.args.slt else self.conv1(x)
            x = self.relu1(x)
            if self.num_bayes_layers >= 3:
                x = self.dropout_layers[2](x)
            x = self.maxpool1(x)

            x = self.conv2(x, threshold) if self.args.slt else self.conv2(x)
            x = self.relu2(x)
            if self.num_bayes_layers >= 2:
                x = self.dropout_layers[1](x)
            x = self.maxpool2(x)

            x = x.view(x.size(0), -1)

            x = self.fc1(x, threshold) if self.args.slt else self.fc1(x)
            x = self.relu3(x)
            if self.num_bayes_layers >= 1:
                x = self.dropout_layers[0](x)

            x = self.fc2(x, threshold) if self.args.slt else self.fc2(x)

        return self.softmax(x)

    def __str__(self):
        layers = []
        in_channels = 3 if self.args.dataset == "cifar10" else 1
        layers.append(f"Input: ({in_channels}, 28, 28)")
        if self.args.dataset == "mnist":
            layers.append("Padding: (2, 2, 2, 2)")
        layers.append(f"Conv2d: {self.conv1}")
        layers.append("ReLU")
        if self.num_bayes_layers >= 3:
            layers.append(f"MC_Dropout: p={self.dropout_layers[0].p}")
        layers.append("MaxPool2d: kernel_size=2")
        layers.append(f"Conv2d: {self.conv2}")
        layers.append("ReLU")
        if self.num_bayes_layers >= 2:
            dropout_index = 1 if self.num_bayes_layers >= 3 else 0
            layers.append(f"MC_Dropout: p={self.dropout_layers[dropout_index].p}")
        layers.append("MaxPool2d: kernel_size=7")
        layers.append("Flatten")
        layers.append(f"Linear: {self.fc1}")
        layers.append("ReLU")
        if self.num_bayes_layers >= 1:
            if self.num_bayes_layers == 3:
                dropout_index = 2
            elif self.num_bayes_layers == 2:
                dropout_index = 1
            else:
                dropout_index = 0
            layers.append(f"MC_Dropout: p={self.dropout_layers[dropout_index].p}")
        layers.append(f"Linear: {self.fc2}")
        layers.append("Softmax")
        layers.append("Output: (10,)")

        return "\n".join(layers)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, args=None, conv_args=None):
        super(BasicBlock, self).__init__()
        if args and args.partial_frozen_slt:
            Conv2d = PartialFrozenConv2d
        elif args and args.slt:
            Conv2d = SLT_Conv2d
        else:
            Conv2d = nn.Conv2d

        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, **conv_args)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, **conv_args)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, **conv_args),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.args = args

    def forward(self, x, threshold=None):
        if self.args and self.args.partial_frozen_slt:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            shortcut = self.shortcut[0](x) if len(self.shortcut) > 0 else x
            if len(self.shortcut) > 0:
                shortcut = self.shortcut[1](shortcut)
            out += shortcut
            out = F.relu(out)
        else:
            out = F.relu(self.bn1(self.conv1(x, threshold) if self.args and self.args.slt else self.conv1(x)))
            out = self.bn2(self.conv2(out, threshold) if self.args and self.args.slt else self.conv2(out))
            shortcut = self.shortcut[0](x, threshold) if (self.args and self.args.slt and len(self.shortcut)>0) else (self.shortcut(x) if len(self.shortcut)>0 else x)
            if len(self.shortcut) > 0:
                shortcut = self.shortcut[1](shortcut)
            out += shortcut
            out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, args, num_classes=10):
        super(ResNet18, self).__init__()
        self.args = args
        base = int(64 * args.scaling_rate)  # scaling_rate はここだけで使う
        self.in_planes = base

        # Choose Conv2d and Linear implementation based on args
        if args.partial_frozen_slt:
            Conv2d = PartialFrozenConv2d
            Linear = PartialFrozenLinear
            conv_args = {'sparsity': args.pruning_rate[0], 'algo': 'local_ep', 'scale_method': 'dynamic_scaled'}
            linear_args = {'sparsity': args.pruning_rate[0], 'algo': 'local_ep', 'scale_method': 'dynamic_scaled'}
        elif args.slt:
            Conv2d = SLT_Conv2d
            Linear = SLT_Linear
            conv_args = {'args': args}
            linear_args = {'args': args}
        else:
            Conv2d = nn.Conv2d
            Linear = nn.Linear
            conv_args = {}
            linear_args = {}

        # Initial convolution
        self.conv1 = Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, **conv_args)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        # Create layers with proper filter scaling
        self.layer1 = self._make_layer(base, 2, stride=1, conv_args=conv_args)
        self.layer2 = self._make_layer(base * 2, 2, stride=2, conv_args=conv_args)
        self.layer3 = self._make_layer(base * 4, 2, stride=2, conv_args=conv_args)
        self.layer4 = self._make_layer(base * 8, 2, stride=2, conv_args=conv_args)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_input_size = base * 8
        self.fc = Linear(self.fc_input_size, num_classes, **linear_args)

        # Initialize dropout layers for Bayesian inference
        self.dropout_layers = nn.ModuleList([
            MC_Dropout(p=args.dropout_rate) for _ in range(args.num_bayes_layers)
        ])
        self.num_bayes_layers = args.num_bayes_layers

    def _make_layer(self, planes, num_blocks, stride, conv_args):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride, self.args, conv_args))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x, threshold=None):
        # Initial convolution
        if self.args.partial_frozen_slt:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.bn1(self.conv1(x, threshold) if self.args.slt else self.conv1(x)))

        # Layer 1
        out = self.layer1[0](out, threshold)
        if self.num_bayes_layers >= 8:
            out = self.dropout_layers[7](out)

        out = self.layer1[1](out, threshold)
        if self.num_bayes_layers >= 7:
            out = self.dropout_layers[6](out)

        # Layer 2
        out = self.layer2[0](out, threshold)
        if self.num_bayes_layers >= 6:
            out = self.dropout_layers[5](out)

        out = self.layer2[1](out, threshold)
        if self.num_bayes_layers >= 5:
            out = self.dropout_layers[4](out)

        # Layer 3
        out = self.layer3[0](out, threshold)
        if self.num_bayes_layers >= 4:
            out = self.dropout_layers[3](out)

        out = self.layer3[1](out, threshold)
        if self.num_bayes_layers >= 3:
            out = self.dropout_layers[2](out)

        # Layer 4
        out = self.layer4[0](out, threshold)
        if self.num_bayes_layers >= 2:
            out = self.dropout_layers[1](out)

        out = self.layer4[1](out, threshold)
        if self.num_bayes_layers >= 1:
            out = self.dropout_layers[0](out)

        # Final layers
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        if self.args.partial_frozen_slt:
            out = self.fc(out)
        else:
            out = self.fc(out, threshold) if self.args.slt else self.fc(out)
        return F.softmax(out, dim=1)

    def __str__(self):
        layers = []
        layers.append("Input: (3, 32, 32)")
        layers.append(f"Conv2d: {self.conv1}")
        layers.append("BatchNorm2d")
        layers.append("ReLU")

        # Layer 1
        layers.append("Layer 1:")
        layers.append(f"BasicBlock: {self.layer1[0]}")
        if self.num_bayes_layers >= 8:
            layers.append(f"MC_Dropout: p={self.dropout_layers[7].p}")
        layers.append(f"BasicBlock: {self.layer1[1]}")
        if self.num_bayes_layers >= 7:
            layers.append(f"MC_Dropout: p={self.dropout_layers[6].p}")

        # Layer 2
        layers.append("Layer 2:")
        layers.append(f"BasicBlock: {self.layer2[0]}")
        if self.num_bayes_layers >= 6:
            layers.append(f"MC_Dropout: p={self.dropout_layers[5].p}")
        layers.append(f"BasicBlock: {self.layer2[1]}")
        if self.num_bayes_layers >= 5:
            layers.append(f"MC_Dropout: p={self.dropout_layers[4].p}")

        # Layer 3
        layers.append("Layer 3:")
        layers.append(f"BasicBlock: {self.layer3[0]}")
        if self.num_bayes_layers >= 4:
            layers.append(f"MC_Dropout: p={self.dropout_layers[3].p}")
        layers.append(f"BasicBlock: {self.layer3[1]}")
        if self.num_bayes_layers >= 3:
            layers.append(f"MC_Dropout: p={self.dropout_layers[2].p}")

        # Layer 4
        layers.append("Layer 4:")
        layers.append(f"BasicBlock: {self.layer4[0]}")
        if self.num_bayes_layers >= 2:
            layers.append(f"MC_Dropout: p={self.dropout_layers[1].p}")
        layers.append(f"BasicBlock: {self.layer4[1]}")
        if self.num_bayes_layers >= 1:
            layers.append(f"MC_Dropout: p={self.dropout_layers[0].p}")

        # Final layers
        layers.append("AdaptiveAvgPool2d: (1, 1)")
        layers.append("Flatten")
        layers.append(f"Linear: {self.fc}")
        layers.append("Softmax")
        layers.append("Output: (10,)")

        return "\n".join(layers)


class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        
        # Get input features and number of classes from dataset
        if args.dataset == 'cora':
            self.num_features = 1433
            self.num_classes = 7
        else:
            raise ValueError(f"Unsupported dataset for GCN: {args.dataset}")

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(self.num_features, args.hidden_channels, args=args))
        
        for _ in range(args.num_layers - 2):
            self.convs.append(GCNConv(args.hidden_channels, args.hidden_channels, args=args))
            
        self.convs.append(GCNConv(args.hidden_channels, self.num_classes, args=args))
        
        # Initialize dropout layers for Bayesian inference
        self.num_bayes_layers = args.num_bayes_layers
        self.dropout_layers = nn.ModuleList([
            MC_Dropout(p=args.dropout_rate) for _ in range(self.num_bayes_layers)
        ])
        
    def forward(self, x, edge_index, threshold=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, threshold)
            x = F.relu(x)
            # Apply dropout in reverse order from the end of layers
            dropout_index = self.num_bayes_layers - (self.args.num_layers - i - 1)
            if dropout_index >= 0 and dropout_index < self.num_bayes_layers:
                x = self.dropout_layers[dropout_index](x)
        x = self.convs[-1](x, edge_index, threshold)
        return F.log_softmax(x, dim=1)


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}^{\top} \sum_{j \in
        \mathcal{N}(i) \cup \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j
        \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. By default, self-loops will be added
            in case :obj:`normalize` is set to :obj:`True`, and not added
            otherwise. (default: :obj:`None`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on-the-fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`
          or sparse matrix :math:`(|\mathcal{V}|, |\mathcal{V}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        args=None,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: Optional[bool] = None,
        normalize: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize

        if add_self_loops and not normalize:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"adding self-loops to the graph when no "
                             f"on-the-fly normalization is applied")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.args = args
        self._cached_edge_index = None
        self._cached_adj_t = None

        if args.partial_frozen_slt:
            self.lin = PartialFrozenLinear(
                in_channels, out_channels, sparsity=args.pruning_rate[0], algo='global_ep', scale_method='dynamic_scaled')
            self.bias = None
        elif args.slt:
            self.lin = SLT_Linear(
                in_channels, out_channels, args=args
            )
            self.bias = None
        else:
            self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')
            if bias:
                self.bias = Parameter(torch.empty(out_channels))
                zeros(self.bias)
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()
            
    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, threshold=None) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        if self.args.slt:
            x = self.lin(x, threshold)
        else:
            x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

