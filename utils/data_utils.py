import torch

SVHN_mean = tuple([x / 255 for x in [129.3, 124.1, 112.4]])
SVHN_std = tuple([x / 255 for x in [68.2, 65.4, 70.4]])
MNIST_mean = (0,)
MNIST_std = (1,)
CIFAR10_mean = (0.4914, 0.4822, 0.4465)
CIFAR10_std = (0.2023, 0.1994, 0.2010)

def random_noise_data(dataset):
    """
    Generate random noise dataset with the same statistics as the original dataset.
    
    Args:
        dataset (str): Name of the dataset ("mnist", "cifar10", or "svhn")
        
    Returns:
        torch.Tensor: Random noise data with the same shape and statistics as the original dataset
    """
    if dataset == "mnist":
        # Generate random noise test dataset with mean MNIST_mean and std MNIST_std
        x_test = torch.normal(mean=MNIST_mean[0], std=MNIST_std[0], size=(10000, 1, 28, 28))
        return x_test.float()
    elif dataset == "cifar10":
        # Generate random noise test dataset with mean CIFAR10_mean and std CIFAR10_std
        x_test = torch.normal(mean=CIFAR10_mean[0], std=CIFAR10_std[0], size=(10000, 3, 32, 32))
        return x_test.float()
    elif dataset == "svhn":
        # Generate random noise test dataset with mean SVHN_mean and std SVHN_std
        x_test = torch.normal(mean=SVHN_mean[0], std=SVHN_std[0], size=(10000, 3, 32, 32))
        return x_test.float()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

            
