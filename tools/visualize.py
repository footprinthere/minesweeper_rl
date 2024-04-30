from typing import Optional

import torch
from torch import Tensor
import matplotlib.pyplot as plt


def visualize_2d_tensor(
    tensor: Tensor,
    lower_bound: Optional[float] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    if lower_bound is not None:
        tensor = torch.where(tensor < lower_bound, torch.nan, tensor)

    plt.figure(figsize=(8, 6))
    plt.imshow(tensor, cmap="coolwarm", interpolation="nearest")

    plt.colorbar()
    if title is not None:
        plt.title(title)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
