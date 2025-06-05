import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(description='LeNet with SLT training')

    # Model parameters
    parser.add_argument('--model', type=str, default='lenet',
                        choices=['lenet', 'resnet18'],
                        help='model architecture (default: lenet)')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'cifar10'],
                        help='dataset to use (default: mnist)')
    parser.add_argument('--num_repeats', type=int, default=1,
                        help='number of repeats (default: 1)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--scaling_rate', type=float, default=1.0,
                        help='scale factor for model size (default: 1.0)')

    # Optimizer parameters
    parser.add_argument('--optimizer_name', type=str, default='sgd',
                        choices=['sgd', 'adamw'],
                        help='optimizer (default: sgd)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD optimizer (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay for SGD optimizer (default: 1e-4)')
    parser.add_argument('--filter_bias_and_bn', action='store_true', default=False,
                        help='filter bias and batch normalization parameters (default: False)')

    # Scheduler parameters
    parser.add_argument('--scheduler_name', type=str, default='cosine_lr',
                        choices=['cosine_lr', 'cosine_lr_warmup', 'multi_step_lr'],
                        help='scheduler (default: cosine_lr)')
    parser.add_argument('--milestones', type=int, nargs='+', default=[50, 75],
                        help='milestones for scheduler (default: [50, 75])')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='gamma for scheduler (default: 0.1)')
    parser.add_argument('--min_lr', type=float, default=None,
                        help='minimum learning rate (default: None)')
    parser.add_argument('--warmup_lr_init', type=float, default=None,
                        help='warmup learning rate (default: None)')
    parser.add_argument('--warmup_t', type=int, default=None,
                        help='warmup epochs (default: None)')
    parser.add_argument('--warmup_prefix', action='store_true', default=False,
                        help='warmup prefix (default: False)')

    # SLT parameters
    parser.add_argument('--slt', action='store_true',
                        help='use SLT (Sparse Learning)')
    parser.add_argument('--partial_frozen_slt', action='store_true',
                        help='use Partial Frozen SLT')
    parser.add_argument('--pruning_rate', type=float, nargs='+', default=[0.5],
                        help='sparsity for linear layers (default: [0.5])')
    parser.add_argument('--init_mode_weight', type=str, default='signed_kaiming_constant',
                        help='weight initialization mode (default: signed_kaiming_constant)')
    parser.add_argument('--init_mode_score', type=str, default='kaiming_normal',
                        help='score initialization mode (default: kaiming_normal)')
    parser.add_argument('--init_scale_weight', type=float, default=1.0,
                        help='weight initialization scale (default: 1.0)')
    parser.add_argument('--init_scale_score', type=float, default=1.0,
                        help='score initialization scale (default: 1.0)')
    parser.add_argument('--p_ratio', type=float, default=0.25,
                        help='p ratio (default: 0.25)')
    parser.add_argument('--r_ratio', type=float, default=0.25,
                        help='r ratio (default: 0.25)')
    parser.add_argument('--algo', type=str, default='global_ep',
                        help='pruning algorithm (default: global_ep)')

    # Miscellaneous parameters
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--seed', type=int, default=110,
                        help='random seed (default: 110)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='disables CUDA training')

    # Bayesian optimization parameters
    parser.add_argument('--mc_samples', type=int, default=10,
                        help='number of MC samples for inference (default: 10)')
    parser.add_argument('--num_bins', type=int, default=10,
                        help='number of bins for ECE calculation (default: 10)')
    parser.add_argument('--num_bayes_layers', type=int, default=1,
                        help='number of Bayesian layers (default: 1)')
    parser.add_argument('--dropout_rate', type=float, default=0.05,
                        help='dropout rate for MC dropout (default: 0.05)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
