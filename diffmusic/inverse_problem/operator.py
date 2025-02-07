import torch
import torchaudio.transforms as T
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram, MelScale


class BaseOperator:
    def transform(self, data, *args, **kwargs):
        raise NotImplementedError

    def inverse_transform(self, data, *args, **kwargs):
        raise NotImplementedError

    def forward(self, data, *args, **kwargs):
        raise NotImplementedError


class IdentityOperator(BaseOperator):
    """
    This operator returns original music.
    """

    def __init__(self, sample_rate):
        self.wav2mel = torch.nn.Sequential(
            MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=1024,
                hop_length=160,
                win_length=1024,
                n_mels=64,
                power=2.0,
            ),
            AmplitudeToDB(stype="power"),
        ).to("cuda")

    def transform(self, audio):
        return torch.clamp(self.wav2mel(audio), min=-80, max=80)

    def inverse_transform(self, mel_spectrogram, vocoder):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)
        waveform = vocoder(mel_spectrogram)
        return waveform

    def forward(self, data, **kwargs):
        return data


class MusicInpaintingOperator(BaseOperator):
    """
    This operator get pre-defined mask and return masked music.
    """

    def __init__(self, audio_length_in_s, sample_rate, mask_type,
                 start_inpainting_s, end_inpainting_s,  mask_percentage, mask_duration_s, interval_s, noiser=None):
        self.audio_length_in_s = audio_length_in_s
        self.sample_rate = sample_rate
        self.mask_type = mask_type

        # for box
        self.start_inpainting_s = start_inpainting_s
        self.end_inpainting_s = end_inpainting_s

        # for random
        self.mask_percentage = mask_percentage

        # for periodic
        self.interval_s = interval_s
        self.mask_duration_s = mask_duration_s

        # generate mask for box inpainting
        self.mask = self.generate_mask()

        self.wav2mel = torch.nn.Sequential(
            MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=1024,
                hop_length=160,
                win_length=1024,
                n_mels=64,
                power=2.0,
            ),
            AmplitudeToDB(stype="power"),
        ).to("cuda")

        self.noiser = noiser

    def generate_mask(self):
        """
        Generate mask based on the masking type.

        Returns:
            torch.Tensor: The generated mask.
        """
        mask = torch.ones([1, self.audio_length_in_s * self.sample_rate])

        if self.mask_type == "box":
            # Box masking
            if self.start_inpainting_s is not None and self.end_inpainting_s is not None:
                mask[:, int(self.start_inpainting_s * self.sample_rate): int(self.end_inpainting_s * self.sample_rate)] = 0.

        elif self.mask_type == "random":
            # Random masking
            total_samples = self.audio_length_in_s * self.sample_rate
            mask_samples = int(self.mask_percentage * total_samples)

            # Ensure the random masks have the specified duration
            mask_count = max(1, mask_samples // int(self.mask_duration_s * self.sample_rate))
            for _ in range(mask_count):
                start = torch.randint(0, mask.shape[1] - int(self.mask_duration_s * self.sample_rate), (1,))
                end = start + int(self.mask_duration_s * self.sample_rate)
                mask[:, start:end] = 0.

        elif self.mask_type == "periodic":
            # Periodic masking
            interval_samples = int(self.interval_s * self.sample_rate)
            mask_duration_samples = int(self.mask_duration_s * self.sample_rate)
            for start in range(0, mask.shape[1], interval_samples):
                end = min(start + mask_duration_samples, mask.shape[1])
                mask[:, start:end] = 0.

        return mask

    def transform(self, audio):
        return self.wav2mel(audio)

    def inverse_transform(self, mel_spectrogram, vocoder):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)
        waveform = vocoder(mel_spectrogram)
        return waveform

    def forward(self, data, **kwargs):
        return self.noiser(data * self.mask.to(data.device))


class PhaseRetrievalOperator(BaseOperator):
    """
    This operator returns amplitude only.
    """

    def __init__(self, n_fft=1024, hop_length=160, win_length=1024, noiser=None):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.mag2mel = MelScale(
            n_mels=64,
            sample_rate=16000,
            n_stft=1024 // 2 + 1
        ).to("cuda")

        self.noiser = noiser

    def transform(self, magnitude):
        return torch.clamp(self.mag2mel(magnitude.float()), min=-80, max=80)

    def inverse_transform(self, mel_spectrogram, vocoder):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)
        waveform = vocoder(mel_spectrogram)
        return waveform

    def forward(self, data, **kwargs):
        spectrogram = torch.stft(
            data,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True
        )
        magnitude = torch.abs(spectrogram)
        return self.noiser(magnitude)


class SuperResolutionOperator(BaseOperator):
    """
    This operator returns downsample music.
    """

    def __init__(self, sample_rate, scale=10, noiser=None):
        self.resampler = T.Resample(orig_freq=sample_rate, new_freq=sample_rate//scale)
        self.wav2mel = torch.nn.Sequential(
            MelSpectrogram(
                sample_rate=16000,
                n_fft=1024,
                hop_length=160,
                win_length=1024,
                n_mels=64,
                power=2.0,
            ),
            AmplitudeToDB(stype="power"),
        ).to("cuda")
        self.noiser = noiser

    def transform(self, audio):
        return torch.clamp(self.wav2mel(audio), min=-80, max=80)

    def inverse_transform(self, mel_spectrogram, vocoder):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)
        waveform = vocoder(mel_spectrogram)
        return waveform

    def forward(self, data, **kwargs):
        self.resampler = self.resampler.to(data.device)
        return self.noiser(self.resampler(data.float()))


class MusicDereverberationOperator(BaseOperator):
    """
    This operator returns reverb music.
    """

    def __init__(self, ir_length=800, decay_factor=0.85, noiser=None):
        self.ir_length = ir_length
        self.decay_factor = decay_factor
        self.wav2mel = torch.nn.Sequential(
            MelSpectrogram(
                sample_rate=16000,
                n_fft=1024,
                hop_length=160,
                win_length=1024,
                n_mels=64,
                power=2.0,
            ),
            AmplitudeToDB(stype="power"),
        ).to("cuda")
        self.noiser = noiser

    def transform(self, audio):
        return torch.clamp(self.wav2mel(audio), min=-80, max=80)

    def inverse_transform(self, mel_spectrogram, vocoder):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)
        waveform = vocoder(mel_spectrogram)
        return waveform

    def generate_impulse_response(self, ir_length=800, decay_factor=0.85):
        ir = torch.randn(ir_length)
        ir = torch.cumsum(ir, dim=0) * decay_factor
        ir /= ir.abs().max()
        return ir.unsqueeze(0)

    def forward(self, data, **kwargs):
        # Generate impulse response of reverberation
        ir = self.generate_impulse_response(ir_length=self.ir_length, decay_factor=self.decay_factor).to(data.device)
        reverb_data = torch.nn.functional.conv1d(
            data.unsqueeze(1).float(), ir.unsqueeze(1), padding=ir.size(1) // 2
        ).squeeze(1)
        return self.noiser(reverb_data)


class StyleGuidanceOperator(BaseOperator):
    """
    This operator returns the outputs of CLAP.
    """

    def __init__(self, clap_model):
        self.clap_model = clap_model

    def transform(self, audio):
        return self.clap_model.get_gram_matrix(audio.float())

    def inverse_transform(self, mel_spectrogram, vocoder):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)
        waveform = vocoder(mel_spectrogram)
        return waveform

    def forward(self, data, **kwargs):
        return data
