from typing import Optional, Union, List, Tuple

import numpy as np
import torch
from diffusers.configuration_utils import register_to_config
from diffusers.models import AutoencoderKL
from transformers import SpeechT5HifiGan
from diffusers.schedulers import DDIMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from pydub import AudioSegment
from tqdm import tqdm
import torchaudio

from diffmusic.data.operator import MusicPhaseRetrievalOperator


def gram_matrix(x):
    b, c, h, w = x.shape
    scale = (c * h * w) ** 0.5
    return torch.einsum("bchw,bdhw->bcd", x / scale, x / scale)


class MusicPhaseRetrievalScheduler(DDIMScheduler):

    @register_to_config
    def __init__(
        self,
        operator: MusicPhaseRetrievalOperator = None,
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

        self.wav2mel = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=1024,
                hop_length=160,
                win_length=1024,
                n_mels=64,
                power=2.0,
            ),
            torchaudio.transforms.AmplitudeToDB(stype="power")
        ).to("cuda")

    def waveform_to_spectrogram(self, waveform, n_fft=1024, hop_length=160, win_length=1024):
        spectrogram = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            return_complex=True
        )
        magnitude, phase = torch.abs(spectrogram), torch.angle(spectrogram)
        return magnitude, phase

    def mel_spectrogram_to_waveform(self, mel_spectrogram, vocoder):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)
        waveform = vocoder(mel_spectrogram)
        return waveform

    def compute_spectrogram_stats(self, mel_spectrogram, mask):
        mask = mask.to(torch.float32)
        constrained_region = mel_spectrogram * mask
        mean = constrained_region.flatten(-2).sum(dim=-1) / mask.flatten(-2).sum(dim=-1)
        std = ((constrained_region - mean.unsqueeze(-1)) ** 2).flatten(-2).sum(dim=-1) / mask.flatten(-2).sum(dim=-1)
        return mean, std

    def style_loss(self, ref_mel_spectrogram, mel_spectrogram, constrained_mask, unconstrained_mask):
        constrained_mean, constrained_std = self.compute_spectrogram_stats(ref_mel_spectrogram, constrained_mask)
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
        start_sample_s: int = 2,
        end_sample_s: int = 3,
        measurement: Optional[torch.Tensor] = None,  # magnitude
        rec_weight: float = 1.,
        style_weight: float = 0.05,
        style_weight2: float = 1.,
        style_weight3: float = 0.005,
        learning_rate: float = 5e-4,
        vae: AutoencoderKL = None,
        vocoder: SpeechT5HifiGan = None,
        original_waveform_length: int = 0,
        audio_length_in_s: int = 0,
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

            pred_audio = self.mel_spectrogram_to_waveform(pred_mel_spectrogram, vocoder)
            pred_audio = pred_audio[:, :original_waveform_length]

            # Phase Retrieval
            pred_magnitude = self.operator.forward(pred_audio)

            difference_magnitude = measurement - pred_magnitude
            rec_loss = torch.linalg.norm(difference_magnitude)

            # style_loss (gram_matrix)
            # gram_measurement = gram_matrix(ref_mel_spectrogram)
            # gram_pred = gram_matrix(reproject_mel_spectrogram)
            #
            # style_loss = torch.linalg.norm(gram_measurement - gram_pred)

            # style_loss2 (calculate mean and std)
            # constrained_mask = torch.ones_like(pred_audio)
            # constrained_mask[:, start_sample_s * 16000: end_sample_s * 16000] = 0.
            # unconstrained_mask = 1 - constrained_mask
            #
            # style_loss2 = self.style_loss(pred_audio, constrained_mask, unconstrained_mask)

            # style_loss3 (calculate mean and std)
            # start_sample = 1000
            # end_sample = 1500
            # constrained_mask = torch.ones_like(reproject_mel_spectrogram)
            # constrained_mask[:, :, start_sample: end_sample, :] = 0.
            # unconstrained_mask = 1 - constrained_mask
            #
            # style_loss3 = self.style_loss(ref_mel_spectrogram, reproject_mel_spectrogram, constrained_mask, unconstrained_mask)

            norm = rec_weight * rec_loss  # + style_weight * style_loss
            tqdm.write(f"rec_loss: {rec_loss}")

            norm_grad = torch.autograd.grad(outputs=norm, inputs=sample)[0]
            prev_sample -= learning_rate * norm_grad

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        return DDIMSchedulerOutput(
            prev_sample=prev_sample.detach(),
            pred_original_sample=pred_original_sample,
        ), norm
