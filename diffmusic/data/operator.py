import torch


class MusicInpaintingOperator:
    """
    This operator get pre-defined mask and return masked music.
    """

    def __init__(self, audio_length_in_s, sample_rate, start_inpainting_s, end_inpainting_s):
        # generate mask for box inpainting
        self.mask = torch.ones([1, audio_length_in_s * sample_rate])
        self.mask[:, start_inpainting_s * sample_rate: end_inpainting_s * sample_rate] = 0.

    def forward(self, data):
        return data * self.mask.to(data.device)

    def transpose(self, data):
        return data


class MusicPhaseRetrievalOperator:
    """
    This operator return amplitude only.
    """

    def __init__(self, n_fft=1024, hop_length=160, win_length=1024):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, data):
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
