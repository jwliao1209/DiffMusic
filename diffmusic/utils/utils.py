import torch


def waveform_to_spectrogram(waveform, n_fft=1024, hop_length=160, win_length=1024):
    spectrogram = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        return_complex=True
    )
    magnitude, phase = torch.abs(spectrogram), torch.angle(spectrogram)
    return magnitude, phase
