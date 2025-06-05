import datetime
import os
import time
import uuid
import warnings
import numpy as np
import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
from utils.metrics import calculate_ece
from scipy.stats import entropy
from sklearn.metrics import accuracy_score
from args import get_args
from utils.datasets import get_mnist_loaders, get_cifar10_loaders, get_cora_loaders
from utils.slt_modules import get_threshold, initialize_params
from utils.optimizer import get_optimizer
from utils.scheduler import get_scheduler
from utils.data_utils import random_noise_data
from models import LeNet, ResNet18, GCN
from utils.set_kthvalue import set_kthvalue
from torch.nn import functional as F
from utils.flops_calc import estimate_gcn_flops


warnings.filterwarnings("ignore")


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for _, (data, target) in enumerate(train_loader):
        if args.slt:
            threshold = get_threshold(model, epoch, args)
        else:
            threshold = None

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
        if args.partial_frozen_slt:
            set_kthvalue(model, args.algo, device)

    return 100.0 * correct / total


def evaluate(
    args,
    model,
    device,
    data_loader
):
    model.eval()
    loss = 0
    correct = 0
    total = 0

    if args.slt:
        threshold = get_threshold(model, args.epochs, args)
    else:
        threshold = None

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


def train_iteration_cnn(args, device, train_loader, val_loader, test_loader, iteration):
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

    # Initialize parameters for partial frozen SLT
    if args.partial_frozen_slt:
        initialize_params(
            model=model,
            w_init_method=args.init_mode_weight,
            s_init_method=args.init_mode_score,
            m_init_method='epl',
            p_ratio=args.p_ratio,
            r_ratio=args.r_ratio,
            r_method='sparsity_distribution',
            nonlinearity='relu',
            algo=args.algo
        )

    # Print model architecture
    if iteration == 0:
        print("-" * 40)
        print("Model Architecture:")
        print(model)
        print("-" * 40)

    # Create optimizer
    optimizer = get_optimizer(
        optimizer_name     = args.optimizer_name,
        lr                 = args.lr,
        momentum           = args.momentum,
        weight_decay       = args.weight_decay,
        model              = model,
        filter_bias_and_bn = args.filter_bias_and_bn
        )

    scheduler = get_scheduler(
        scheduler_name = args.scheduler_name,
        optimizer      = optimizer,
        milestones     = args.milestones,
        gamma          = args.gamma,
        max_epoch      = args.epochs,
        min_lr         = args.min_lr,
        warmup_lr_init = args.warmup_lr_init,
        warmup_t       = args.warmup_t,
        warmup_prefix  = args.warmup_prefix
    )

    # print training results
    best_val_acc = 0
    print("-" * 40)
    print("Training Results:")
    print("Epoch, train_acc, val_acc, test_acc")
    print("-" * 40)

    # create checkpoint directory
    checkpoint_path = create_checkpoint_path(model_name=args.model, iteration=iteration)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        # train
        train_acc = train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()

        # evaluate
        val_acc = evaluate(
            args,
            model,
            device,
            val_loader,
        )
        test_acc = evaluate(
            args,
            model,
            device,
            test_loader,
        )

        print(f"{epoch}\t{train_acc:.2f}\t{val_acc:.2f}\t{test_acc:.2f}")

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    if args.partial_frozen_slt:
        set_kthvalue(model, args.algo, device)
    if args.slt:
        threshold = get_threshold(model, args.epochs, args)
    else:
        threshold = None

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
    ece = calculate_ece(all_probs, all_labels, num_bins=args.num_bins) * 100

    # calculate aPE
    x_test_random = random_noise_data(args.dataset).to(device)
    with torch.no_grad():
        probs = mc_predict(model, x_test_random, threshold, args.mc_samples, args.slt)
        ape = np.mean(entropy(probs.detach().cpu().numpy(), axis=1))

    # calculate FLOPS
    slt_flag = args.slt or args.partial_frozen_slt
    if slt_flag:
        pruning_rate = args.pruning_rate[0]
        temp_args = type("Args", (), vars(args))()
        temp_args.slt = False
        temp_args.partial_frozen_slt = False
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

    flops = flops / 1e6

    print("-" * 40)
    print("Final Results")
    print("ECE(%), aPE(nats), Accuracy(%), FLOPS(10^6)")
    print("-" * 40)
    print(f"{float(ece):.2f}\t{ape:.4f}\t{mean_acc:.2f}\t{flops:.2f}")
    print("-" * 40)

    # Delete the checkpoint file after calculating final results
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return mean_acc, ece, ape, flops


def train_iteration_gnn(args, device, data, train_mask, val_mask, test_mask, iteration):
    # print args
    if iteration == 0:
        print("Args:")
        print("-" * 40)
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")

    # Create model
    model = GCN(args).to(device)

    # Initialize parameters for partial frozen SLT
    if args.partial_frozen_slt:
        initialize_params(
            model=model,
            w_init_method=args.init_mode_weight,
            s_init_method=args.init_mode_score,
            m_init_method='epl',
            p_ratio=args.p_ratio,
            r_ratio=args.r_ratio,
            r_method='sparsity_distribution',
            nonlinearity='relu',
            algo=args.algo
        )

    # Print model architecture
    if iteration == 0:
        print("-" * 40)
        print("Model Architecture:")
        print(model)
        print("-" * 40)

    # Create optimizer
    optimizer = get_optimizer(
        optimizer_name=args.optimizer_name,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        model=model,
        filter_bias_and_bn=args.filter_bias_and_bn
    )

    scheduler = get_scheduler(
        scheduler_name=args.scheduler_name,
        optimizer=optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        max_epoch=args.epochs,
        min_lr=args.min_lr,
        warmup_lr_init=args.warmup_lr_init,
        warmup_t=args.warmup_t,
        warmup_prefix=args.warmup_prefix
    )

    # print training results
    best_val_acc = 0
    print("-" * 40)
    print("Training Results:")
    print("Epoch, train_acc, val_acc, test_acc")
    print("-" * 40)

    # create checkpoint directory
    checkpoint_path = create_checkpoint_path(model_name=args.model, iteration=iteration)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()

        # evaluate
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            train_acc = (out[train_mask].argmax(dim=1) == data.y[train_mask]).float().mean() * 100
            val_acc = (out[val_mask].argmax(dim=1) == data.y[val_mask]).float().mean() * 100
            test_acc = (out[test_mask].argmax(dim=1) == data.y[test_mask]).float().mean() * 100

        print(f"{epoch}\t{train_acc:.2f}\t{val_acc:.2f}\t{test_acc:.2f}")

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)

        if args.partial_frozen_slt:
            set_kthvalue(model, args.algo, device)

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    if args.partial_frozen_slt:
        set_kthvalue(model, args.algo, device)
    if args.slt:
        threshold = get_threshold(model, args.epochs, args)
    else:
        threshold = None

    # calculate ECE
    with torch.no_grad():
        out = model(data.x, data.edge_index, threshold)
        probs = torch.exp(out[test_mask])
        all_probs = probs.cpu().numpy()
        all_labels = data.y[test_mask].cpu().numpy()

    mean_acc = accuracy_score(all_labels, np.argmax(all_probs, axis=1)) * 100
    ece = calculate_ece(all_probs, all_labels, num_bins=args.num_bins) * 100

    # calculate aPE
    with torch.no_grad():
        probs = torch.exp(out[test_mask])
        ape = np.mean(entropy(probs.cpu().numpy(), axis=1))

    # calculate FLOPS
    slt_flag = args.slt or args.partial_frozen_slt
    if slt_flag:
        pruning_rate = args.pruning_rate[0]
        temp_args = type("Args", (), vars(args))()
        temp_args.slt = False
        temp_args.partial_frozen_slt = False
    else:
        temp_args = args

    # Re-Create model
    temp_model = GCN(temp_args).to(device)
    temp_model.eval()

    print("-" * 40)
    flops = estimate_gcn_flops(data, model)

    if slt_flag:
        flops = flops * (1 - pruning_rate)

    print("-" * 40)
    print("Final Results")
    print("ECE(%), aPE(nats), Accuracy(%), FLOPS(10^6)")
    print("-" * 40)
    print(f"{float(ece):.2f}\t{ape:.4f}\t{mean_acc:.2f}\t{flops:.2f}")
    print("-" * 40)

    # Delete the checkpoint file after calculating final results
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return mean_acc, ece, ape, flops


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
    elif args.dataset == "cora":
        data, train_mask, val_mask, test_mask = get_cora_loaders(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Multiple training iterations
    all_results = []
    for i in range(args.num_repeats):
        print(f"\nIteration {i+1}/{args.num_repeats}")
        if args.dataset in ["mnist", "cifar10"]:
            test_acc, ece, ape, flops = train_iteration_cnn(
                args, device, train_loader, val_loader, test_loader, i
            )
        elif args.dataset == "cora":
            test_acc, ece, ape, flops = train_iteration_gnn(
                args, device, data, train_mask, val_mask, test_mask, i
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
