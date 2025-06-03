import datetime
import os
import time
import uuid

# Suppress warnings
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ptflops import get_model_complexity_info
from metrics import calculate_ece
from fvcore.nn import FlopCountAnalysis, flop_count_table

from scipy.stats import entropy
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# import argparse

from args import get_args
from datasets import get_mnist_loaders, get_cifar10_loaders
from slt_modules import get_threshold

from data_utils import random_noise_data
from models import LeNet, ResNet18


def count_flops(model, input_size=(1, 1, 28, 28)):
    # 入力テンソルを準備
    if isinstance(input_size, torch.Size):
        input_size = tuple(input_size)
    device = next(model.parameters()).device
    dummy_input = torch.zeros(input_size, device=device)

    # モデルを評価モードに設定
    model.eval()

    # FLOPS計算
    flops = FlopCountAnalysis(model, dummy_input)
    print("\nFLOPS Analysis:")
    print("-" * 80)
    print(flop_count_table(flops))
    print(f"Total FLOPS: {flops.total()/1e6:.2f}M")
    print("-" * 80)

    return flops.total()


def create_checkpoint_path(model_name="lenet", iteration=0, base_dir="checkpoints"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    uid = uuid.uuid4().hex[:6]
    checkpoint_path = f"{base_dir}/{model_name}_best_{timestamp}_pid{pid}_{uid}_iter{iteration}.pth"
    return checkpoint_path


def mc_predict(model, data, threshold=None, mc_samples=10, slt=False):
    outputs = [model(data, threshold) if slt else model(data) for _ in range(mc_samples)]
    return torch.stack(outputs).mean(0)


def train(args, model, device, train_loader, optimizer, epoch, threshold):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, threshold) if args.slt else model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    return 100.0 * correct / total


def evaluate(
    args,
    model,
    device,
    data_loader,
    phase="val",
    threshold=None,
):
    model.eval()
    loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            outputs = []
            output = model(data, threshold) if args.slt else model(data)
            outputs.append(output)
            # Average predictions
            output = torch.stack(outputs).mean(0)
            loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def train_iteration(args, device, train_loader, val_loader, test_loader, iteration):
    # print args
    if iteration == 0:
        print("Args:")
        print("-" * 40)
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")

    # Create model
    if args.model == "lenet":
        model = LeNet(args).to(device)
    elif args.model == "resnet18":
        model = ResNet18(args).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Print model architecture
    if iteration == 0:
        print("-" * 40)
        print("Model Architecture:")
        print(model)
        print("-" * 40)

    # Create optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # print training results
    best_val_acc = 0
    print("Training Results:")
    print("Epoch, train_acc, val_acc, test_acc")
    print("-" * 40)

    # create checkpoint directory
    checkpoint_path = create_checkpoint_path(model_name=args.model, iteration=iteration)
    
    # Training loop 
    for epoch in range(1, args.epochs + 1):
        
        # train
        if args.slt:
            threshold = get_threshold(model, epoch, args)
        train_acc = train(args, model, device, train_loader, optimizer, epoch, threshold)

        # evaluate
        if args.slt:
            threshold = get_threshold(model, epoch, args)
        val_acc = evaluate(
            args,
            model,
            device,
            val_loader,
            phase="val",
            threshold=threshold,
        )
        test_acc = evaluate(
            args,
            model,
            device,
            test_loader,
            phase="test",
            threshold=threshold,
        )

        print(f"{epoch}\t{train_acc:.2f}\t{val_acc:.2f}\t{test_acc:.2f}")

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    threshold = get_threshold(model, args.epochs, args)

    # calculate ECE 
    all_probs, all_labels = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = mc_predict(model, data, threshold, args.mc_samples, args.slt)
            all_probs.append(output.detach().cpu().numpy())
            all_labels.append(target.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    mean_acc = accuracy_score(all_labels, np.argmax(all_probs, axis=1)) * 100
    ece = calculate_ece(all_probs, all_labels, num_bins=args.num_bins)

    # calculate aPE
    x_test_random = random_noise_data(args.dataset).to(device)
    with torch.no_grad():
        probs = mc_predict(model, x_test_random, threshold, args.mc_samples, args.slt)
        ape = np.mean(entropy(probs.detach().cpu().numpy(), axis=1))

    # calculate FLOPS
    slt_flag = args.slt
    if slt_flag:
        pruning_rate = args.pruning_rate[0]
        temp_args = type("Args", (), vars(args))()
        temp_args.slt = False
    else:
        temp_args = args

    # Re-Create model
    if args.model == "lenet":
        temp_model = LeNet(temp_args).to(device)
    elif args.model == "resnet18":
        temp_model = ResNet18(temp_args).to(device)

    temp_model.eval()

    if args.dataset.lower() == "mnist":
        input_shape = (1, 28, 28)
    elif args.dataset.lower() == "cifar10":
        input_shape = (3, 32, 32)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print("-" * 40)
    macs, _ = get_model_complexity_info(
        temp_model,
        input_shape,
        as_strings=False,
        print_per_layer_stat=True
    )
    flops = 2 * macs

    if slt_flag:
        flops = flops * (1 - pruning_rate)
    else:
        flops = flops

    print("-" * 40)
    print("Final Results")
    print("ECE(%), aPE(nats), Accuracy(%), FLOPS(10^6)")
    print("-" * 40)
    print(f"{float(ece)*100:.2f}\t{ape:.4f}\t{mean_acc:.2f}\t{flops/1e6:.2f}")
    print("-" * 40)

    # Delete the checkpoint file after calculating final results
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return mean_acc, float(ece) * 100, ape, flops / 1e6


def main():
    # Record start time
    start_time = time.time()
    start_datetime = datetime.datetime.now()

    args = get_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    # Load data based on dataset type
    if args.dataset == "mnist":
        train_loader, val_loader, test_loader = get_mnist_loaders(args)
    elif args.dataset == "cifar10":
        train_loader, val_loader, test_loader = get_cifar10_loaders(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Multiple training iterations
    all_results = []
    for i in range(args.num_repeats):
        print(f"\nIteration {i+1}/{args.num_repeats}")
        test_acc, ece, ape, flops = train_iteration(
            args, device, train_loader, val_loader, test_loader, i
        )
        all_results.append((test_acc, ece, ape, flops))

    # Calculate and print average results
    avg_acc = sum(r[0] for r in all_results) / len(all_results)
    avg_ece = sum(r[1] for r in all_results) / len(all_results)
    avg_ape = sum(r[2] for r in all_results) / len(all_results)
    avg_flops = sum(r[3] for r in all_results) / len(all_results)

    print("\nAverage Final Results")
    print("ECE(%), aPE(nats), Accuracy(%), FLOPS(10^6)")
    print("-" * 40)
    print(f"{avg_ece:.2f}\t{avg_ape:.4f}\t{avg_acc:.2f}\t{avg_flops:.2f}")
    print("-" * 40)

    # Record end time and calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print("\nExecution Summary:")
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Execution time: {execution_time:.2f} seconds")


if __name__ == "__main__":
    args = get_args()
    print(f"Arguments: {args}")
    main()
