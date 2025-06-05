import torch.nn as nn
import torch.nn.functional as F
from utils.slt_modules import SLT_Conv2d, SLT_Linear
from utils.slt_modules import PartialFrozenConv2d, PartialFrozenLinear
from utils.mc_dropout import MC_Dropout


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

