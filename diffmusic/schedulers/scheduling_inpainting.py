from typing import Optional, Union, List, Tuple

import torch
import numpy as np
from diffusers.configuration_utils import register_to_config
from diffusers.schedulers import DDIMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput

from pydub import AudioSegment
import random


class DDIMInpaintingScheduler(DDIMScheduler):

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

    def mel_spectrogram_to_waveform(self, vocoder, mel_spectrogram):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        waveform = vocoder(mel_spectrogram)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        waveform = waveform.float()
        return waveform

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

        measurement: Optional[torch.Tensor] = None,
        original_waveform_length: int = 0,
        vae=None,
        vocoder=None,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        with torch.enable_grad():
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

            # Supervise on mel_spectrogram
            pred_original_sample = 1 / vae.config.scaling_factor * pred_original_sample
            pred_mel_spectrogram = vae.decode(pred_original_sample).sample

            start_sample = 1000
            end_sample = 1500

            difference = measurement - pred_mel_spectrogram
            difference[:, :, start_sample: end_sample, :] = 0.
            norm = torch.linalg.norm(difference)

            norm_grad = torch.autograd.grad(outputs=norm, inputs=sample)[0]

            prev_sample -= norm_grad * 0.5

            # Supervise on waveform
            # pred_original_sample = 1 / vae.config.scaling_factor * pred_original_sample
            # pred_mel_spectrogram = vae.decode(pred_original_sample).sample
            #
            # pred_audio = self.mel_spectrogram_to_waveform(vocoder, pred_mel_spectrogram)
            # pred_audio = pred_audio[:, :original_waveform_length]
            #
            # # (1, 48000)
            # start_sample = 200000
            # end_sample = 300000
            #
            # # create mask
            # mask = torch.ones_like(pred_audio).to(pred_audio.device)
            # mask[:, start_sample: end_sample] = 0.
            #
            # pred_audio = pred_audio * mask
            # measurement = measurement * mask
            #
            # difference = measurement - pred_audio
            # norm = torch.linalg.norm(difference)
            #
            # norm_grad = torch.autograd.grad(outputs=norm, inputs=sample)[0]
            #
            # prev_sample -= norm_grad

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        return DDIMSchedulerOutput(
            prev_sample=prev_sample.detach(),
            pred_original_sample=pred_original_sample,
        ), norm
