import torch
import torch.nn.functional as F
from functools import partial
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
import numpy as np
from scipy.stats import entropy

def maximum_mean_discrepancy(
    source_samples, target_samples, kernel="gaussian", minimum=0.0, squared=True
):
    """
    This Maximum Mean Discrepancy (MMD) loss is calculated with a number of different Gaussian kernels.
    source_samples: samples from the true (true posterior samples when posterior is analytical for NPE, true samples from simulator without likelihood for NLE)
    target_samples: samples from the approximator
    """
    sigmas = [
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        100,
        1e3,
        1e4,
        1e5,
        1e6,
    ]

    if kernel == "gaussian":
        kernel = partial(_gaussian_kernel_matrix, sigmas=sigmas)
    else:
        print("Invalid kernel specified. Falling back to default Gaussian.")
        kernel = partial(_gaussian_kernel_matrix, sigmas=sigmas)

    loss_value = _mmd_kernel(source_samples, target_samples, kernel=kernel)
    loss_value = torch.maximum(torch.tensor(minimum), loss_value)

    if squared:
        return loss_value
    else:
        return torch.sqrt(loss_value)


def _gaussian_kernel_matrix(x, y, sigmas):
    """Computes a Gaussian Radial Basis Kernel between the samples of x and y."""
    
    norm = lambda v: torch.sum(v ** 2, dim=1)
    beta = 1.0 / (2.0 * torch.tensor(sigmas).unsqueeze(1))
    dist = torch.transpose(norm(x.unsqueeze(2) - y.transpose(0, 1)), 0, 1)
    s = torch.matmul(beta, dist.reshape((1, -1)))
    kernel = torch.sum(torch.exp(-s), dim=0).view(dist.shape)
    return kernel


def _mmd_kernel(x, y, kernel=None):
    """Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
    x      : torch.Tensor of shape (num_samples, num_features)
    y      : torch.Tensor of shape (num_samples, num_features)
    """

    loss = torch.mean(kernel(x, x))
    loss += torch.mean(kernel(y, y))
    loss -= 2 * torch.mean(kernel(x, y))
    return loss


def c2st(
    X: torch.Tensor,
    Y: torch.Tensor,
    n_folds: int = 5,
    scoring: str = "accuracy",
) -> torch.Tensor:
    """Classifier-based 2-sample test returning accuracy

    Trains classifiers with N-fold cross-validation [1]. Scikit learn MLPClassifier are
    used, with 2 hidden layers of 10x dim each, where dim is the dimensionality of the
    samples X and Y.

    Args:
        X: Sample 1
        Y: Sample 2
        seed: Seed for sklearn
        n_folds: Number of folds
        z_score: Z-scoring using X
        noise_scale: If passed, will add Gaussian noise with std noise_scale to samples

    References:
        [1]: https://scikit-learn.org/stable/modules/cross_validation.html
    """

    X = X.cpu().numpy()
    Y = Y.cpu().numpy()

    ndim = X.shape[1]

    clf = MLPClassifier(
        activation="relu",
        hidden_layer_sizes = (10 * ndim, 10 * ndim),
        max_iter = 10000,
        solver = "adam",
    )

    data = np.concatenate((X, Y))
    target = np.concatenate(
        (
            np.zeros((X.shape[0],)),
            np.ones((Y.shape[0],)),
        )
    )

    shuffle = KFold(n_splits=n_folds, shuffle=True)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring=scoring)

    return torch.tensor(scores.mean())

def MMD_heuristic(x, y, l_list):
    """ Approximates the squared MMD between samples x_i ~ P and y_i ~ Q
    """

    m = x.shape[0]
    n = y.shape[0]

    z = torch.cat((x, y), dim=0)

    K = kernel_matrix(z, z, l_list)

    kxx = K[0:m, 0:m]
    kyy = K[m:(m + n), m:(m + n)]
    kxy = K[0:m, m:(m + n)]

    return (1 / m ** 2) * torch.sum(kxx) - (2 / (m * n)) * torch.sum(kxy) + (1 / n ** 2) * torch.sum(kyy)


def median_heuristic(y):
    a = torch.cdist(y, y)**2
    return torch.sqrt(torch.median(a / 2))

def kernel_matrix(x, y, l_list):
    ds = torch.stack([torch.cdist(
        torch.atleast_2d(x[:, d]).reshape(-1, 1), 
        torch.atleast_2d(y[:, d].reshape(-1, 1))
        )**2 for d in range(x.shape[-1])])
    
    ls = torch.as_tensor(l_list)
    kernel = torch.exp(torch.sum(-(1 / (2 * ls.view(-1, 1, 1) ** 2)) * ds, dim = 0))
    # kernel = torch.sum(torch.exp(-(1 / (2 * ls.view(-1, 1, 1) ** 2)) * ds), dim=0)

    return kernel


# def kl_divergence(approximated_densities, exact_densities, forward = True):
#     jitter = 1e-20
#     approximated_densities = np.clip(approximated_densities, jitter, None)
#     exact_densities = np.clip(exact_densities, jitter, None)

#     if forward:
#         kl = entropy(exact_densities, approximated_densities) # forward KL divergence 

#     else:
#         kl = entropy(approximated_densities, exact_densities)

#     return kl


if __name__ == "__main__":
    x = torch.randn(100, 10)
    y = torch.randn(100, 10)

    mmd = maximum_mean_discrepancy(x, y)
    print(mmd)

    mmd = maximum_mean_discrepancy(x, x)
    print(mmd)

    x = torch.randn(10000, 2)
    y = torch.randn(10000, 2)

    c2st = c2st(x, y)



