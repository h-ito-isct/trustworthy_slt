import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='LeNet with SLT training')

    # Training parameters
    parser.add_argument('--num_repeats', type=int, default=1,
                        help='number of repeats (default: 1)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.08,
                        help='learning rate (default: 0.08)')
    parser.add_argument('--momentum', type=float, default=0.85,
                        help='momentum for SGD optimizer (default: 0.85)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay for SGD optimizer (default: 1e-4)')

    # Model parameters
    parser.add_argument('--scaling_rate', type=float, default=1.0,
                        help='scale factor for model size (default: 1.0)')
    parser.add_argument('--num_bayes_layers', type=int, default=1,
                        help='number of Bayesian layers (default: 1)')
    parser.add_argument('--dropout_rate', type=float, default=0.05,
                        help='dropout rate for MC dropout (default: 0.05)')

    # SLT parameters
    parser.add_argument('--slt', action='store_true',
                        help='use SLT (Sparse Learning)')
    parser.add_argument('--pruning_rate', type=float, nargs='+', default=[0.5],
                        help='sparsity for linear layers (default: [0.5])')
    parser.add_argument('--init_mode_weight', type=str, default='signed_kaiming_constant',
                        help='weight initialization mode (default: signed_kaiming_constant)')
    parser.add_argument('--init_mode_score', type=str, default='kaiming_uniform',
                        help='score initialization mode (default: kaiming_uniform)')
    parser.add_argument('--init_scale_weight', type=float, default=1.0,
                        help='weight initialization scale (default: 1.0)')
    parser.add_argument('--init_scale_score', type=float, default=1.0,
                        help='score initialization scale (default: 1.0)')
    parser.add_argument('--model', type=str, default='lenet',
                        choices=['lenet', 'resnet18'],
                        help='model architecture (default: lenet)')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'cifar10'],
                        help='dataset to use (default: mnist)')

    # Hardware parameters
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--seed', type=int, default=110,
                        help='random seed (default: 110)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='disables CUDA training')

    # MC Dropout parameters
    parser.add_argument('--mc_samples', type=int, default=10,
                        help='number of MC samples for inference (default: 10)')

    # New parameters
    parser.add_argument('--num_bins', type=int, default=10,
                        help='number of bins for ECE calculation (default: 10)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
