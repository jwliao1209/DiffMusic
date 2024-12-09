from typing import Optional, Union, List, Tuple

import numpy as np
import torch
from diffusers.configuration_utils import register_to_config
from diffusers.models import AutoencoderKL
from diffusers.schedulers import DDIMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from pydub import AudioSegment
from tqdm import tqdm


def gram_matrix(x):
    b, c, h, w = x.shape
    scale = (c * h * w) ** 0.5
    return torch.einsum("bchw,bdhw->bcd", x / scale, x / scale)


class MusicInpaintingScheduler(DDIMScheduler):

    @register_to_config
    def __init__(
        self,
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

    def compute_spectrogram_stats(self, mel_spectrogram, mask):
        mask = mask.to(torch.float32)
        constrained_region = mel_spectrogram * mask
        mean = constrained_region.flatten(-2).sum(dim=-1) / mask.flatten(-2).sum(dim=-1)
        std = ((constrained_region - mean.unsqueeze(-1)) ** 2).flatten(-2).sum(dim=-1) / mask.flatten(-2).sum(dim=-1)
        return mean, std

    def style_loss(self, mel_spectrogram, constrained_mask, unconstrained_mask):
        constrained_mean, constrained_std = self.compute_spectrogram_stats(mel_spectrogram, constrained_mask)
        unconstrained_mean, unconstrained_std = self.compute_spectrogram_stats(mel_spectrogram, unconstrained_mask)

        loss = torch.nn.functional.mse_loss(unconstrained_mean, constrained_mean) + \
               torch.nn.functional.mse_loss(unconstrained_std, constrained_std)
        return loss

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        # args for inverse problem
        start_sample: int = 1000,
        end_sample: int = 1500,
        measurement: Optional[torch.Tensor] = None,
        rec_weight: float = 1.,
        style_weight: float = 1.,
        style_weight2: float = 1.,
        learning_rate: float = 0.5,
        vae: AutoencoderKL = None,
        original_waveform_length: int = 0,
    ) -> Union[DDIMSchedulerOutput, Tuple]:

        with torch.enable_grad():
            sample = sample.clone().detach().requires_grad_(True)
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

            timesteps_prev = timestep - self.config.num_train_timesteps // self.num_inference_steps
            alpha_prod_t = self.alphas_cumprod[timestep]
            beta_prod_t = 1 - alpha_prod_t
            alpha_prod_t_prev = self.alphas_cumprod[timesteps_prev] if timesteps_prev >= 0 else self.final_alpha_cumprod
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            noise_pred = (sample - (alpha_prod_t ** 0.5) * pred_original_sample) / (beta_prod_t ** 0.5)
            prev_sample = (alpha_prod_t_prev ** 0.5) * pred_original_sample + (beta_prod_t_prev ** 0.5) * noise_pred

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

            # Guided diffusion posterior sampling using gradient-based methods

            # # # Supervise on mel_spectrogram # # #
            pred_original_sample = 1 / vae.config.scaling_factor * pred_original_sample
            pred_mel_spectrogram = vae.decode(pred_original_sample).sample

            difference = measurement - pred_mel_spectrogram
            difference[:, :, start_sample: end_sample, :] = 0.

            # style_loss (gram_matrix)
            gram_measurement = gram_matrix(measurement)
            gram_pred = gram_matrix(pred_mel_spectrogram)

            style_loss = torch.linalg.norm(gram_measurement - gram_pred)
            rec_loss = torch.linalg.norm(difference)

            # style_loss2 (calculate mean and std)
            constrained_mask = torch.ones_like(pred_mel_spectrogram)
            constrained_mask[:, :, start_sample: end_sample, :] = 0.
            unconstrained_mask = 1 - constrained_mask

            style_loss2 = self.style_loss(pred_mel_spectrogram, constrained_mask, unconstrained_mask)

            norm = rec_weight * rec_loss + style_weight * style_loss + style_weight2 * style_loss2
            tqdm.write(f"rec_loss: {rec_loss}, style_loss: {style_loss}, style_loss2: {style_loss2}")

            norm_grad = torch.autograd.grad(outputs=norm, inputs=sample)[0]
            prev_sample -= learning_rate * norm_grad

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        return DDIMSchedulerOutput(
            prev_sample=prev_sample.detach(),
            pred_original_sample=pred_original_sample,
        ), norm
