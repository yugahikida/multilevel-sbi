
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import median_abs_deviation
from sklearn.metrics import r2_score
from torch import Tensor
import torch

def plot_recovery(
    post_samples,
    prior_samples,
    point_agg=np.median,
    uncertainty_agg=median_abs_deviation,
    param_names=None,
    fig_size=None,
    label_fontsize=16,
    title_fontsize=18,
    metric_fontsize=16,
    tick_fontsize=12,
    add_corr=True,
    add_r2=True,
    color="#8f2727",
    n_col=None,
    n_row=None,
    xlabel="Ground truth",
    ylabel="Estimated",
    **kwargs,
):
    """Creates and plots publication-ready recovery plot with true vs. point estimate + uncertainty.
    The point estimate can be controlled with the ``point_agg`` argument, and the uncertainty estimate
    can be controlled with the ``uncertainty_agg`` argument.

    This plot yields similar information as the "posterior z-score", but allows for generic
    point and uncertainty estimates:

    https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html

    Important: Posterior aggregates play no special role in Bayesian inference and should only
    be used heuristically. For instance, in the case of multi-modal posteriors, common point
    estimates, such as mean, (geometric) median, or maximum a posteriori (MAP) mean nothing.

    Parameters
    ----------
    post_samples      : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The posterior draws obtained from n_data_sets
    prior_samples     : np.ndarray of shape (n_data_sets, n_params)
        The prior draws (true parameters) obtained for generating the n_data_sets
    point_agg         : callable, optional, default: ``np.median``
        The function to apply to the posterior draws to get a point estimate for each marginal.
        The default computes the marginal median for each marginal posterior as a robust
        point estimate.
    uncertainty_agg   : callable or None, optional, default: scipy.stats.median_abs_deviation
        The function to apply to the posterior draws to get an uncertainty estimate.
        If ``None`` provided, a simple scatter using only ``point_agg`` will be plotted.
    param_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    fig_size          : tuple or None, optional, default : None
        The figure size passed to the matplotlib constructor. Inferred if None.
    label_fontsize    : int, optional, default: 16
        The font size of the y-label text
    title_fontsize    : int, optional, default: 18
        The font size of the title text
    metric_fontsize   : int, optional, default: 16
        The font size of the goodness-of-fit metric (if provided)
    tick_fontsize     : int, optional, default: 12
        The font size of the axis tick labels
    add_corr          : bool, optional, default: True
        A flag for adding correlation between true and estimates to the plot
    add_r2            : bool, optional, default: True
        A flag for adding R^2 between true and estimates to the plot
    color             : str, optional, default: '#8f2727'
        The color for the true vs. estimated scatter points and error bars
    n_row             : int, optional, default: None
        The number of rows for the subplots. Dynamically determined if None.
    n_col             : int, optional, default: None
        The number of columns for the subplots. Dynamically determined if None.
    xlabel            : str, optional, default: 'Ground truth'
        The label on the x-axis of the plot
    ylabel            : str, optional, default: 'Estimated'
        The label on the y-axis of the plot
    **kwargs          : optional
        Additional keyword arguments passed to ax.errorbar or ax.scatter.
        Example: `rasterized=True` to reduce PDF file size with many dots

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ShapeError
        If there is a deviation from the expected shapes of ``post_samples`` and ``prior_samples``.
    """

    # Compute point estimates and uncertainties
    est = point_agg(post_samples, axis=1)
    if uncertainty_agg is not None:
        u = uncertainty_agg(post_samples, axis=1)

    # Determine n params and param names if None given
    n_params = prior_samples.shape[-1]
    if param_names is None:
        param_names = [f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)]

    # Determine number of rows and columns for subplots based on inputs
    if n_row is None and n_col is None:
        n_row = int(np.ceil(n_params / 6))
        n_col = int(np.ceil(n_params / n_row))
    elif n_row is None and n_col is not None:
        n_row = int(np.ceil(n_params / n_col))
    elif n_row is not None and n_col is None:
        n_col = int(np.ceil(n_params / n_row))

    # Initialize figure
    if fig_size is None:
        fig_size = (int(4 * n_col), int(4 * n_row))
    f, axarr = plt.subplots(n_row, n_col, figsize=fig_size)
    
    # turn axarr into 1D list
    axarr = np.atleast_1d(axarr)
    if n_col > 1 or n_row > 1:
        axarr_it = axarr.flat
    else:
        axarr_it = axarr

    for i, ax in enumerate(axarr_it):
        if i >= n_params:
            break


        alpha = 0.3

        # Add scatter and error bars
        if uncertainty_agg is not None:
            _ = ax.errorbar(prior_samples[:, i], est[:, i], yerr=u[:, i], fmt="o", alpha=alpha, color=color, **kwargs)
        else:
            _ = ax.scatter(prior_samples[:, i], est[:, i], alpha=alpha, color=color, **kwargs)

        # Make plots quadratic to avoid visual illusions
        lower = min(prior_samples[:, i].min(), est[:, i].min())
        upper = max(prior_samples[:, i].max(), est[:, i].max())
        eps = (upper - lower) * 0.1
        ax.set_xlim([lower - eps, upper + eps])
        ax.set_ylim([lower - eps, upper + eps])
        ax.plot(
            [ax.get_xlim()[0], ax.get_xlim()[1]],
            [ax.get_ylim()[0], ax.get_ylim()[1]],
            color="black",
            alpha=0.9,
            linestyle="dashed",
        )

        # Add optional metrics and title
        if add_r2:
            r2 = r2_score(prior_samples[:, i], est[:, i])
            ax.text(
                0.1,
                0.9,
                # "$R^2$ = {:.3f}".format(r2),
                "$R^2$ = {:.2f}".format(r2),
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
                size=metric_fontsize,
            )
        if add_corr:
            corr = np.corrcoef(prior_samples[:, i], est[:, i])[0, 1]
            ax.text(
                0.1,
                0.8,
                # "$r$ = {:.3f}".format(corr),
                "$r$ = {:.2f}".format(corr),
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
                size=metric_fontsize,
            )
        ax.set_title(param_names[i], fontsize=title_fontsize)

        # Prettify
        sns.despine(ax=ax)
        # ax.grid(alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax.tick_params(axis="both", which="minor", labelsize=tick_fontsize)

    # Only add x-labels to the bottom row
    bottom_row = axarr if n_row == 1 else axarr[0] if n_col == 1 else axarr[n_row - 1, :]
    # for _ax in bottom_row:
    #     _ax.set_xlabel(xlabel, fontsize=label_fontsize)

    for i, _ax in enumerate(bottom_row):
        _ax.set_xlabel(xlabel[i], fontsize=label_fontsize)

    # Only add y-labels to right left-most row
    if n_row == 1:  # if there is only one row, the ax array is 1D
        axarr[0].set_ylabel(ylabel, fontsize=label_fontsize)
    # If there is more than one row, the ax array is 2D
    else:
        for _ax in axarr[:, 0]:
            _ax.set_ylabel(ylabel, fontsize=label_fontsize)

    # Remove unused axes entirely
    for _ax in axarr_it[n_params:]:
        _ax.remove()

    f.tight_layout()
    return f


def calculate_coverage(logprob_post: Tensor, logprob_true: Tensor, alpha: float, n_samples: int) -> Tensor:
    """
    Calculate the coverage of the posterior distributution.
    Args:
        logprob_post: Log-probability of the posterior samples. [n_samples, n_sim]
        logprob_true: Log-probability of the true parameter. [n_sim]
        alpha: Coverage level between 0 and 1.
        n_samples: Number of posterior samples.

    Returns:
        coverage: Mean coverage of the posterior distribution in 1 - alpha HPD credible interval over n_sim. [1]
    """
    if float(alpha) == 0:
        return torch.zeros(1).mean()
    elif float(alpha) == 1:
        return torch.ones(1).mean()
    else:
        beta = 1 - alpha
        cut_off = int(n_samples * beta)
        logq_samples_sorted, _ = torch.sort(logprob_post, dim = 0)
        alpha_logprob_min = logq_samples_sorted[cut_off:, :].min(0).values

        return (alpha_logprob_min < logprob_true).float().mean()
    
def prepare_data_for_coverage_plot(net, n_grid, n_samples, x: Tensor, theta):
    coverage = np.zeros(n_grid)
    confidence_level = np.linspace(0, 1, n_grid)
    logprob_true = net.log_prob_unstandardized(theta, x)
    post_samples = net.sample_unstandardized(num_samples = n_samples, condition = x)

    logprob_post = torch.stack([net.log_prob_unstandardized(post_samples[:, i, :], x) for i in range(n_samples)])

    for (j, alpha) in enumerate(confidence_level):
        coverage[j] += calculate_coverage(logprob_post, logprob_true, alpha, n_samples)

    return coverage, confidence_level

def plot_loss(loss_array, start:int = 0, end: int = None):
    """
    Input:
        loss_array: (epoch, 3) array. row1: loss, row2: log_prob_diff.mean, row3: penalty, row4: log_prob_1, row5: log_prob_0
    """

    l = loss_array[:, 0] # l = (h0 + h1) + penalty
    h1 = - loss_array[:, 1]
    penalty = loss_array[:, 2]
    E_f11 = - loss_array[:, 3]
    E_f10 = - loss_array[:, 4]
    h0 = l - h1 - penalty
    var_diff = loss_array[:, 5]

    title_list = ['Loss', r'$h_0$', r'$h_1$', r"$E(f_{11})$", r"$E(f_{10})$",  r'$h_0 - E(f_{10})$', 'penalty', r'$Var(h_0) - Var(h_1)$']

    if end is None:
        x_axis = range(start, len(l))
    else:
        x_axis = range(start, end)

    positive_indices = np.where(penalty > 0)[0]
    first_positive_index = positive_indices[0] if positive_indices.size > 0 else None

    fig, axes = plt.subplots(8, 1, figsize=(6, 10))
    axes[0].plot(x_axis, l[start:end], label='Loss', color='blue')
    axes[1].plot(x_axis, h0[start:end], label='First term', color='red')
    axes[2].plot(x_axis, h1[start:end], label = 'Second term', color='gray')
    axes[3].plot(x_axis, E_f11[start:end], label = '$E(f_1)$ (second term)', color='purple')
    axes[4].plot(x_axis, E_f10[start:end], label = '$E(f_0)$ (second term)', color='green')
    axes[5].plot(x_axis, h0[start:end] - E_f10[start:end], label = 'Second term', color='gray')
    axes[6].plot(x_axis, penalty[start:end], label= 'penalty', color='orange')
    axes[6].scatter(first_positive_index, penalty[first_positive_index], color='red', zorder=5)
    axes[7].plot(x_axis, var_diff[start:end], label= r'$Var(f_1 - f_0) - f_0)$', color='pink')

    for i in range(len(title_list)):
       axes[i].set_title(title_list[i])

       if i == 4:
          axes[i].set_xlabel('Epoch')
    fig.tight_layout()

    return fig