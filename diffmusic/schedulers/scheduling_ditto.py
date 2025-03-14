from typing import Optional, Union, List, Tuple

import numpy as np
import torch
from diffusers.configuration_utils import register_to_config
from diffusers.models import AutoencoderKL
from diffusers.schedulers import DDIMScheduler

from transformers import SpeechT5HifiGan

from .utils import InverseProblemSchedulerOutput
from ..inverse_problem.operator import BaseOperator
from ..torch_utils import randn_tensor


class DITTOScheduler(DDIMScheduler):
    '''
    Diffusion Posterior Sampling for General Noisy Inverse Problems
    ref: https://arxiv.org/abs/2209.14687
    '''

    @register_to_config
    def __init__(
        self,
        operator: BaseOperator = None,
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

    def optim_prompt(
            self,
            model_output: torch.Tensor,
            timestep: int,
            sample: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_hidden_states_1: Optional[torch.Tensor] = None,
            eta: float = 0.0,
            use_clipped_model_output: bool = False,
            generator: Optional[torch.Generator] = None,
            variance_noise: Optional[torch.Tensor] = None,
            return_dict: bool = True,
            measurement: Optional[torch.Tensor] = None,  # ref_wav
            vae: AutoencoderKL = None,
            vocoder: SpeechT5HifiGan = None,
            original_waveform_length: int = 0,
            optim_prompt_learning_rate: float = 1e-4,
            supervised_space: str = "mel_spectrogram",
    ) -> Union[InverseProblemSchedulerOutput, Tuple]:
        """
        Update prompt embedding
        """
        if encoder_hidden_states is not None:
            optimizer = torch.optim.SGD([encoder_hidden_states, encoder_hidden_states_1], lr=optim_prompt_learning_rate)
        else:
            optimizer = torch.optim.SGD([encoder_hidden_states_1], lr=optim_prompt_learning_rate)

        with torch.enable_grad():
            for _ in range(1):
                optimizer.zero_grad()
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

                pred_mel_spectrogram = vae.decode(
                    1 / vae.config.scaling_factor * pred_original_sample
                ).sample

                pred_audio = self.operator.inverse_transform(pred_mel_spectrogram, vocoder)
                pred_audio = pred_audio[:, :original_waveform_length]
                pred_audio = self.operator.forward(pred_audio)

                if supervised_space == "wav_form":
                    difference = measurement - pred_audio
                elif supervised_space == "mel_spectrogram":
                    ref_mel = self.operator.transform(measurement)
                    pred_mel = self.operator.transform(pred_audio)
                    difference = ref_mel - pred_mel
                else:
                    raise ValueError("supervised_space should be either 'wav_form' or 'mel_spectrogram")

                rec_loss = torch.linalg.norm(difference)
                rec_loss.backward()
                optimizer.step()

        return InverseProblemSchedulerOutput(
            encoder_hidden_states=encoder_hidden_states.detach(),
            encoder_hidden_states_1=encoder_hidden_states_1.detach(),
        )

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
        vae: AutoencoderKL = None,
        vocoder: SpeechT5HifiGan = None,
        original_waveform_length: int = 0,
        supervised_space: str = "mel_spectrogram",
        ditto_optimizer: torch.optim.Optimizer = None,
        init_latents: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> Union[InverseProblemSchedulerOutput, Tuple]:

        timesteps_prev = timestep - self.config.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        alpha_prod_t_prev = self.alphas_cumprod[timesteps_prev] if timesteps_prev >= 0 else self.final_alpha_cumprod
        variance = self._get_variance(timestep, timesteps_prev)
        std_dev_t = eta * variance ** 0.5

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

        pred_mel_spectrogram = vae.decode(
            1 / vae.config.scaling_factor * prev_sample
        ).sample
        pred_audio = self.operator.inverse_transform(pred_mel_spectrogram, vocoder)
        pred_audio = pred_audio[:, :original_waveform_length]
        pred_audio = self.operator.forward(pred_audio)

        if supervised_space == "wav_form":
            difference = measurement - pred_audio
        elif supervised_space == "mel_spectrogram":
            ref_mel = self.operator.transform(measurement)
            pred_mel = self.operator.transform(pred_audio)
            difference = ref_mel - pred_mel
        else:
            raise ValueError("supervised_space should be either 'wav_form' or 'mel_spectrogram")

        rec_loss = torch.linalg.norm(difference)

        if timestep == 1:
            rec_loss.backward()
            torch.nn.utils.clip_grad_norm_([init_latents], max_norm=100.0)
            ditto_optimizer.step()

        return InverseProblemSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=pred_original_sample,
            loss=rec_loss.detach(),
        )
