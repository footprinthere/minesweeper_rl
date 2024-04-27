from typing import Optional
from dataclasses import dataclass

from torch import Tensor

from game.open_result import OpenResult


@dataclass
class TrainStepResult:

    next_state: Optional[Tensor]
    # `None` if the episode has terminated

    loss: Optional[float]
    # `None` if the replay memory is not filled enough

    open_result: OpenResult
