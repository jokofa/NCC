#
from typing import Union, Optional
import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch_kmeans.utils.utils import group_by_label_mean
from torch.utils.data import DataLoader


def calculate_inertia(x: Tensor, centers: Tensor, labels: Tensor) -> Tensor:
    """Compute sum of squared distances of samples
    to their closest cluster center."""
    n, d = x.size()
    assert len(labels) == n
    # select assigned center by label and calculate squared distance
    assigned_centers = centers.gather(
        index=labels[:, None].expand(-1, d),
        dim=0,
    )
    # squared distance to closest center
    d = torch.norm((x - assigned_centers), p=2, dim=-1) ** 2
    d[d == float("inf")] = 0
    return torch.sum(d, dim=-1)


def visualize_clustering(
        x: np.ndarray,
        y: np.ndarray,
        y_hat: np.ndarray,
        alpha: float = 0.5,
        plot_legend: bool = False,
        plot_centers: bool = True,
        plot_centroids: bool = False,
        compute_inertia: bool = True,
        remove_axis_ticks_and_labels: bool = True,
        fig_logger: Optional[SummaryWriter] = None,
        save_dir: str = None,
        save_format: str = "svg",
        plot_idx: Optional[int] = None,
        global_epoch: Optional[int] = None,
        **kwargs):

    if plot_legend:
        legend = "auto"
    else:
        legend = False

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
    x_data = pd.DataFrame(np.concatenate((x[:, :3], y[:, None], y_hat[:, None]), axis=-1),
                          columns=["x_coord", "y_coord", "weight", "org_label", "pred_label"])
    # convert dtype
    x_data.org_label = x_data.org_label.astype(int)
    x_data.pred_label = x_data.pred_label.astype(int)

    sns.scatterplot(x="x_coord", y="y_coord", hue="org_label",
                    size="weight", sizes=(10, 100),
                    alpha=alpha, palette="muted", legend=legend,
                    data=x_data, ax=axs[0])
    sns.scatterplot(x="x_coord", y="y_coord", hue="org_label",
                    alpha=alpha, palette="muted", legend=legend,
                    data=x_data, ax=axs[1])
    sns.scatterplot(x="x_coord", y="y_coord", hue="pred_label",
                    alpha=alpha, palette="muted", legend=legend,
                    data=x_data, ax=axs[2])

    if plot_centers or plot_centroids or compute_inertia:
        coords = x[:, :2]
        coords = torch.from_numpy(coords)
        lbl = np.unique(y)
        nc = len(lbl)
        y_ = torch.from_numpy(y)
        org_centers = group_by_label_mean(
            coords[None, :, :],
            y_[None, None, :],
            torch.arange(nc).unsqueeze(0)
        ).squeeze(0).squeeze(0)
        lbl = np.unique(y_hat)
        nc = len(lbl)
        y_hat_ = torch.from_numpy(y_hat)
        pred_centers = group_by_label_mean(
            coords[None, :, :],
            y_hat_[None, None, :],
            torch.arange(nc).unsqueeze(0)
        ).squeeze(0).squeeze(0)

    if compute_inertia:
        org_inertia = calculate_inertia(coords, org_centers, y_).item()
        pred_inertia = calculate_inertia(coords, pred_centers, y_hat_).item()

    if plot_centroids:
        n = len(coords)
        org_centroids = torch.argmin(
            torch.norm(
                (coords[None, None, :, :].expand(1, org_centers.size(0), -1, -1) -
                 org_centers[None, :, None, :].expand(1, -1, n, -1)),
                p=2, dim=-1
            ), dim=-1
        ).view(-1)
        org_centroids = coords.gather(index=org_centroids[:, None].expand(-1, 2), dim=0).cpu().numpy()
        pred_centroids = torch.argmin(
            torch.norm(
                (coords[None, None, :, :].expand(1, pred_centers.size(0), -1, -1) -
                 pred_centers[None, :, None, :].expand(1, -1, n, -1)),
                p=2, dim=-1
            ), dim=-1
        ).view(-1)
        pred_centroids = coords.gather(index=pred_centroids[:, None].expand(-1, 2), dim=0).cpu().numpy()
        #axs[0].scatter(x=org_centroids[:, 0], y=org_centroids[:, 1], marker="D", c="black", alpha=0.7)
        axs[1].scatter(x=org_centroids[:, 0], y=org_centroids[:, 1], marker="D", c="black", alpha=0.7)
        axs[2].scatter(x=pred_centroids[:, 0], y=pred_centroids[:, 1], marker="D", c="black", alpha=0.7)

    if plot_centers:
        org_centers = org_centers.cpu().numpy()
        #axs[0].scatter(x=org_centers[:, 0], y=org_centers[:, 1], marker="X", c="red", alpha=0.9)
        axs[1].scatter(x=org_centers[:, 0], y=org_centers[:, 1], marker="X", c="red", alpha=0.9)
        pred_centers = pred_centers.cpu().numpy()
        axs[2].scatter(x=pred_centers[:, 0], y=pred_centers[:, 1], marker="X", c="red", alpha=0.9)

    if remove_axis_ticks_and_labels:
        for ax in axs.flatten():
            ax.set(xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)

    if plot_legend:
        for ax in axs:
            sns.move_legend(ax, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

    oi_str = f"\ninertia={org_inertia: .5f}" if compute_inertia is not None else ''
    pi_str = f"\ninertia={pred_inertia: .5f}" if compute_inertia is not None else ''
    axs[0].set_title(f"opt label & weights")
    axs[1].set_title(f"opt label {oi_str}")
    axs[2].set_title(f"pred label {pi_str}")

    title = f"clustering{'_' + str(plot_idx) if plot_idx is not None else ''}"
    fig.suptitle(title)
    fig.tight_layout()

    if save_dir is None:
        plt.show()
    else:   # save figure
        save_pth = os.path.join(save_dir, f"plot_{title}.{save_format}")
        plt.savefig(fname=save_pth, format=save_format, bbox_inches='tight')

    if fig_logger is not None:
        fig_logger.add_figure(tag=title, figure=fig, close=True, global_step=global_epoch)

    # clear all current figures
    plt.clf()

