from dataclasses import dataclass
from typing import Optional, Union, List, Tuple

import numpy as np
import torch
from diffusers.configuration_utils import register_to_config
from diffusers.models import AutoencoderKL
from diffusers.utils import BaseOutput
from transformers import SpeechT5HifiGan

from diffmusic.operators.operator import Operator
from diffusers.schedulers import DDIMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput

from ..torch_utils import randn_tensor


@dataclass
class DiffMusicSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


class DiffMusicScheduler(DDIMScheduler):

    @register_to_config
    def __init__(
        self,
        operator: Operator = None,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        rescale_betas_zero_snr: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas,
            clip_sample=clip_sample,
            set_alpha_to_one=set_alpha_to_one,
            steps_offset=steps_offset,
            prediction_type=prediction_type,
            thresholding=thresholding,
            dynamic_thresholding_ratio=dynamic_thresholding_ratio,
            clip_sample_range=clip_sample_range,
            sample_max_value=sample_max_value,
            timestep_spacing=timestep_spacing,
            rescale_betas_zero_snr=rescale_betas_zero_snr,
        )
        self.operator = operator

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator: Optional[torch.Generator] = None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        measurement: Optional[torch.Tensor] = None,  # ref_wav
        learning_rate: float = 1.0,
        vae: AutoencoderKL = None,
        vocoder: SpeechT5HifiGan = None,
        original_waveform_length: int = 0,
    ) -> Union[DDIMSchedulerOutput, Tuple]:

        pred_original_sample = super().step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            eta=eta,
            use_clipped_model_output=use_clipped_model_output,
            generator=generator,
            variance_noise=variance_noise,
            return_dict=return_dict,
        ).pred_original_sample

        pred_original_sample = pred_original_sample.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([pred_original_sample], lr=learning_rate)
        optimization_step = 1

        with torch.enable_grad():
            for _ in range(optimization_step):
                optimizer.zero_grad()

                pred_mel_spectrogram = vae.decode(
                    1 / vae.config.scaling_factor * pred_original_sample,
                    return_dict=False,
                )[0]

                pred_audio = self.operator.inverse_transform(pred_mel_spectrogram, vocoder)
                pred_audio = pred_audio[:, :original_waveform_length]

                pred_audio = self.operator.forward(pred_audio)  # pred_audio:  torch.Size([1, 80000])
                ref_mel = self.operator.transform(measurement)
                pred_mel = self.operator.transform(pred_audio)

                rec_loss = torch.linalg.norm(ref_mel - pred_mel)  # torch.nn.functional.mse_loss(ref_mel, pred_mel)

                rec_loss.backward()
                optimizer.step()

                distance = torch.linalg.norm(ref_mel - pred_mel)

        timesteps_prev = timestep - self.config.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        alpha_prod_t_prev = self.alphas_cumprod[timesteps_prev] if timesteps_prev >= 0 else self.final_alpha_cumprod

        variance = self._get_variance(timestep, timesteps_prev)
        std_dev_t = eta * variance ** 0.5

        noise_pred = (sample - (alpha_prod_t ** 0.5) * pred_original_sample) / (beta_prod_t ** 0.5)
        pred_sample_direction = ((1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5) * noise_pred
        prev_sample = (alpha_prod_t_prev ** 0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                )
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance

        return DiffMusicSchedulerOutput(
            prev_sample=prev_sample.detach(),
            pred_original_sample=pred_original_sample,
            loss=distance.detach(),
        )
