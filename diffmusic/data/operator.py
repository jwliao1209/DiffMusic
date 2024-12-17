import torch
import torchaudio.transforms as T


class MusicInpaintingOperator:
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

    def forward(self, data, **kwargs):
        return data * self.mask.to(data.device)

    def transpose(self, data):
        return data


class MusicPhaseRetrievalOperator:
    """
    This operator returns amplitude only.
    """

    def __init__(self, n_fft=1024, hop_length=160, win_length=1024):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

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


class MusicSuperResolutionOperator:
    """
    This operator returns downsample music.
    """

    def __init__(self, sample_rate, scale=10):
        self.resampler = T.Resample(orig_freq=sample_rate, new_freq=sample_rate//scale)

    def forward(self, data, **kwargs):
        self.resampler = self.resampler.to(data.device)
        return self.resampler(data.float())

    def transpose(self, data):
        return data


class MusicDereverberationOperator:
    """
    This operator returns reverb music.
    """

    def __init__(self, ir_length=800, decay_factor=0.85):
        self.ir_length = ir_length
        self.decay_factor = decay_factor

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


class MusicSourceSeparationOperator:
    """
    This operator returns a mix of music.
    """

    def __init__(self, num_mix=2):
        self.num_mix = num_mix

        self.weight = 1.0 / num_mix
        self.weight = [round(self.weight, 2) for _ in range(num_mix)]
        self.weight[-1] = 1.0 - sum(self.weight[:-1])
        self.weight[-1] = round(self.weight[-1], 2)

    def forward(self, data_list, **kwargs):
        mixed_music = torch.zeros_like(data_list[0]).float()
        for i in range(self.num_mix):
            mixed_music += self.weight[i] * data_list[i].float()
        return mixed_music

    def transpose(self, data):
        return data

