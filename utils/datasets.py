import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


### Change the root to the path of the dataset
datasets_path = '/work/Shared/Datasets'


def get_mnist_loaders(args):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(
        root=datasets_path,   
        train=True,
        download=True,
        transform=transform
    )

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    test_dataset = datasets.MNIST(
        root=datasets_path,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def get_cifar10_loaders(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    full_train_dataset = datasets.CIFAR10(root=datasets_path, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=datasets_path, train=False, download=True, transform=transform_test)

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    val_dataset.dataset.transform = transform_test

    num_workers = getattr(args, 'num_workers', args.num_workers)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
