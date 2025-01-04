import torch
import torchaudio.transforms as T
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram, MelScale


class Operator:
    def transform(self, data, *args, **kwargs):
        raise NotImplementedError
    
    def inverse_transform(self, data, *args, **kwargs):
        raise NotImplementedError

    def forward(self, data, *args, **kwargs):
        raise NotImplementedError
    
    def transpose(self, data, *args, **kwargs):
        raise NotImplementedError


class IdentityOperator(Operator):
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

    def transpose(self, data):
        return data


class MusicInpaintingOperator(Operator):
    """
    This operator get pre-defined mask and return masked music.
    """

    def __init__(self, audio_length_in_s, sample_rate, start_inpainting_s, end_inpainting_s):
        self.audio_length_in_s = audio_length_in_s
        self.sample_rate = sample_rate
        self.start_inpainting_s = start_inpainting_s
        self.end_inpainting_s = end_inpainting_s

        # generate mask for box inpainting
        self.mask = torch.ones([1, self.audio_length_in_s * self.sample_rate])
        self.mask[:, self.start_inpainting_s * self.sample_rate: self.end_inpainting_s * self.sample_rate] = 0.

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
        return self.wav2mel(audio)

    def inverse_transform(self, mel_spectrogram, vocoder):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)
        waveform = vocoder(mel_spectrogram)
        return waveform

    def forward(self, data, **kwargs):
        return data * self.mask.to(data.device)

    def transpose(self, data):
        return data


class PhaseRetrievalOperator(Operator):
    """
    This operator returns amplitude only.
    """

    def __init__(self, n_fft=1024, hop_length=160, win_length=1024):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.mag2mel = MelScale(
            n_mels=64,
            sample_rate=16000,
            n_stft=1024 // 2 + 1
        ).to("cuda")

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
        return magnitude

    def transpose(self, data):
        return data


class SuperResolutionOperator(Operator):
    """
    This operator returns downsample music.
    """

    def __init__(self, sample_rate, scale=10):
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

    def transform(self, audio):
        return torch.clamp(self.wav2mel(audio), min=-80, max=80)

    def inverse_transform(self, mel_spectrogram, vocoder):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)
        waveform = vocoder(mel_spectrogram)
        return waveform

    def forward(self, data, **kwargs):
        self.resampler = self.resampler.to(data.device)
        return self.resampler(data.float())

    def transpose(self, data):
        return data


class MusicDereverberationOperator(Operator):
    """
    This operator returns reverb music.
    """

    def __init__(self, ir_length=800, decay_factor=0.85):
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
        return reverb_data

    def transpose(self, data):
        return data


class StyleGuidanceOperator(Operator):
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

    def transpose(self, data):
        return data
