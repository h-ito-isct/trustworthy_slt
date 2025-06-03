import torch
import numpy as np

def _compute_calibration_bin_statistics(num_bins, logits, labels_true, labels_predicted=None):
    """
    Compute calibration bin statistics as in TensorFlow's implementation.

    Args:
        num_bins: Number of probability bins
        logits: Logits tensor of shape (n_samples, n_classes)
        labels_true: True labels tensor of shape (n_samples,)
        labels_predicted: Predicted labels tensor of shape (n_samples,)

    Returns:
        event_bin_counts: Counts of correct/incorrect predictions in each bin
        pmean_observed: Mean predicted probability in each bin
    """
    probs = torch.softmax(logits, dim=1)
    if labels_predicted is None:
        labels_predicted = torch.argmax(probs, dim=1)

    # Get the predicted probabilities for the predicted classes
    pred_probs = probs[torch.arange(len(labels_predicted)), labels_predicted]

    # Create bins
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Initialize statistics
    event_bin_counts = torch.zeros((2, num_bins))  # [incorrect, correct] x [bins]
    pmean_observed = torch.zeros(num_bins)

    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Find samples in this bin
        in_bin = (pred_probs > bin_lower) & (pred_probs <= bin_upper)
        bin_size = in_bin.sum().float()

        if bin_size > 0:
            # Count correct and incorrect predictions
            correct = (labels_predicted[in_bin] == labels_true[in_bin])
            event_bin_counts[0, i] = (~correct).sum().float()  # incorrect
            event_bin_counts[1, i] = correct.sum().float()     # correct

            # Compute mean predicted probability in this bin
            pmean_observed[i] = pred_probs[in_bin].mean()

    return event_bin_counts, pmean_observed

def calculate_ece(probs, labels, num_bins=10):
    """
    Calculate Expected Calibration Error using PyTorch

    Args:
        probs (torch.Tensor or np.ndarray): Probability predictions of shape (n_samples, n_classes)
        labels (torch.Tensor or np.ndarray): True labels of shape (n_samples,)
        num_bins (int): Number of bins for calibration error calculation

    Returns:
        float: Expected Calibration Error
    """
    # Convert to torch tensors if they are numpy arrays
    if isinstance(probs, np.ndarray):
        probs = torch.from_numpy(probs)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    # Convert probabilities to logits
    logits = torch.log(probs / (1 - probs + 1e-15))

    # Convert labels to class indices if they are one-hot encoded
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = torch.argmax(labels, dim=1)

    # Compute calibration bin statistics
    event_bin_counts, pmean_observed = _compute_calibration_bin_statistics(
        num_bins, logits, labels)

    # Compute the marginal probability of observing a probability bin
    bin_n = event_bin_counts.sum(dim=0)
    pbins = bin_n / bin_n.sum()

    # Compute the marginal probability of making a correct decision given a bin
    tiny = np.finfo(np.float32).tiny
    pcorrect = event_bin_counts[1, :] / (bin_n + tiny)

    # Compute ECE as in the paper
    ece = torch.sum(pbins * torch.abs(pcorrect - pmean_observed))

    return ece.item()

def compute_ape(probs):
    log_probs = np.log(probs + 1e-12)
    entropy = -np.sum(probs * log_probs, axis=1)
    ape = np.mean(entropy)
    return ape
