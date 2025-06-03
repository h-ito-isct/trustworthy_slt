import math
import numbers
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter


class GetSubnet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, threshold, zeros, ones):
        out = torch.where(
            scores < threshold, zeros.to(scores.device), ones.to(scores.device)
        )
        return out

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


class SparseModule(nn.Module):
    def init_param_(
        self,
        param,
        init_mode=None,
        scale=None,
        sparse_value=None,
        gain="relu",
        args=None,
    ):
        if init_mode == "kaiming_normal":
            nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity=gain)
            param.data *= scale
        elif init_mode == "uniform":
            nn.init.uniform_(param, a=-1, b=1)
            param.data *= scale
        elif init_mode == "kaiming_uniform":
            nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity=gain)
            param.data *= scale
        elif init_mode == "kaiming_normal_SF":
            fan = nn.init._calculate_correct_fan(param, "fan_in")
            gain = nn.init.calculate_gain(gain)
            scale_fan = fan * (1 - sparse_value)
            std = gain / math.sqrt(scale_fan)
            param.data.normal_(0, std)
        elif init_mode == "signed_constant":
            fan = nn.init._calculate_correct_fan(param, "fan_in")
            gain = nn.init.calculate_gain(gain)
            std = gain / math.sqrt(fan)
            nn.init.kaiming_normal_(param)  # use only its sign
            param.data = param.data.sign() * std
            param.data *= scale
        elif init_mode == "signed_kaiming_constant":
            fan = nn.init._calculate_correct_fan(param, "fan_in")
            gain = nn.init.calculate_gain(gain)
            scale_fan = fan * (1 - sparse_value)
            std = gain / math.sqrt(scale_fan)
            nn.init.kaiming_normal_(param)  # use only its sign
            param.data = param.data.sign() * std
            param.data *= scale  # scale value is defined in defualt as 1.0
        elif init_mode == "signed_xavier_uniform_constant_SF":
            gain = nn.init.calculate_gain(gain)
            nn.init.xavier_uniform_(param, gain)
            std = torch.std(param)
            scaled_std = std * math.sqrt(1 / (1 - sparse_value))
            nn.init.kaiming_normal_(param)
            param.data = param.data.sign() * scaled_std
            param.data *= scale
        else:
            raise NotImplementedError


class SLT_Linear(SparseModule):
    def __init__(self, in_ch, out_ch, args):
        super().__init__()

        self.sparsity = args.pruning_rate
        self.init_mode_weight = args.init_mode_weight
        self.init_mode_score = args.init_mode_score
        self.init_scale_weight = args.init_scale_weight
        self.init_scale_score = args.init_scale_score

        # 重みの形状を修正: (out_features, in_features)
        self.weight = nn.Parameter(torch.ones(out_ch, in_ch))
        self.weight.requires_grad = False
        self.weight_score = nn.Parameter(torch.ones(self.weight.size()))
        self.weight_score.is_score = True
        self.weight_score.sparsity = self.sparsity

        self.init_param_(
            self.weight,
            init_mode=self.init_mode_weight,
            scale=self.init_scale_weight,
            sparse_value=self.sparsity[0],
            args=args,
        )

        self.init_param_(
            self.weight_score,
            init_mode=self.init_mode_score,
            scale=self.init_scale_score,
            sparse_value=self.sparsity[0],
            args=args,
        )

        self.weight_zeros = torch.zeros(self.weight.size())
        self.weight_ones = torch.ones(self.weight.size())
        self.weight_zeros.requires_grad = False
        self.weight_ones.requires_grad = False
        self.args = args

    def reset_parameters(self):
        self.init_param_(
            self.weight,
            init_mode=self.init_mode_weight,
            scale=self.init_scale_weight,
            sparse_value=self.sparsity[0],
            args=self.args,
        )
        self.init_param_(
            self.weight_score,
            init_mode=self.init_mode_score,
            scale=self.init_scale_score,
            sparse_value=self.sparsity[0],
            args=self.args,
        )

    def forward(self, x, threshold):
        subnet = GetSubnet.apply(torch.abs(self.weight_score),
            threshold,
            self.weight_zeros,
            self.weight_ones,
        )
        pruned_weight = self.weight * subnet
        ret = F.linear(x, pruned_weight, None)
        return ret


class SLT_Conv2d(SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, args=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        if isinstance(padding, str):
            self.padding = padding
        else:
            self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups

        # 重みの初期化
        self.weight = nn.Parameter(torch.ones(out_channels, in_channels // groups, *self.kernel_size))
        self.weight.requires_grad = False

        # スコアの初期化
        self.weight_score = nn.Parameter(torch.ones(self.weight.size()))
        self.weight_score.is_score = True
        self.weight_score.sparsity = args.conv_sparsity if hasattr(args, 'conv_sparsity') else args.pruning_rate

        # 初期化パラメータの設定
        self.init_mode_weight = args.init_mode_weight
        self.init_mode_score = args.init_mode_score
        self.init_scale_weight = args.init_scale_weight
        self.init_scale_score = args.init_scale_score

        # 重みとスコアの初期化
        self.init_param_(
            self.weight,
            init_mode=self.init_mode_weight,
            scale=self.init_scale_weight,
            sparse_value=self.weight_score.sparsity[0],
            args=args,
        )

        self.init_param_(
            self.weight_score,
            init_mode=self.init_mode_score,
            scale=self.init_scale_score,
            sparse_value=self.weight_score.sparsity[0],
            args=args,
        )

        # 疎化用の定数テンソル
        self.weight_zeros = torch.zeros(self.weight.size())
        self.weight_ones = torch.ones(self.weight.size())
        self.weight_zeros.requires_grad = False
        self.weight_ones.requires_grad = False
        self.args = args

    def reset_parameters(self):
        self.init_param_(
            self.weight,
            init_mode=self.init_mode_weight,
            scale=self.init_scale_weight,
            sparse_value=self.weight_score.sparsity[0],
            args=self.args,
        )
        self.init_param_(
            self.weight_score,
            init_mode=self.init_mode_score,
            scale=self.init_scale_score,
            sparse_value=self.weight_score.sparsity[0],
            args=self.args,
        )

    def forward(self, x, threshold):
        # スコアに基づいてマスクを生成
        subnet = GetSubnet.apply(
            torch.abs(self.weight_score),
            threshold,
            self.weight_zeros,
            self.weight_ones,
        )

        # 重みにマスクを適用
        pruned_weight = self.weight * subnet

        # 畳み込み演算を実行（バイアスなし）
        return F.conv2d(
            x,
            pruned_weight,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )


def calculate_sparsities(sparsity, epoch, max_epoch_half):
    return [value * (epoch / max_epoch_half) for value in sparsity]


def percentile(t, q):
    k = 1 + round(0.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()


def get_threshold(model, epoch=None, args=None):
    if not args.slt:
        return None

    max_epoch_half = args.epochs // 2
    sparsity_value = args.pruning_rate[0] * (min(epoch, max_epoch_half) / max_epoch_half)
    local = torch.cat(
        [
            p.detach().flatten()
            for name, p in model.named_parameters()
            if hasattr(p, "is_score") and p.is_score
        ]
    )
    threshold = percentile(
        local.abs(),
        sparsity_value * 100,
    )
    return threshold
