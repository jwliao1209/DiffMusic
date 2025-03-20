from dataclasses import dataclass
from typing import Optional

import torch
from diffusers.utils import BaseOutput


@dataclass
class InverseProblemSchedulerOutput(BaseOutput):
    sample: Optional[torch.Tensor] = None
    prev_sample: torch.Tensor = None
    pred_original_sample: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    encoder_hidden_states: Optional[torch.Tensor] = None
    encoder_hidden_states_1: Optional[torch.Tensor] = None
    init_latents: Optional[torch.Tensor] = None
