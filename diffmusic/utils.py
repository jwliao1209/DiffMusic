import numpy as np
import resampy
import soundfile as sf
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


def load_audio_task(fname, sample_rate, channels, dtype="float32"):
    if dtype not in ['float64', 'float32', 'int32', 'int16']:
        raise ValueError(f"dtype not supported: {dtype}")

    wav_data, sr = sf.read(fname, dtype=dtype)
    # For integer type PCM input, convert to [-1.0, +1.0]
    if dtype == 'int16':
        wav_data = wav_data / 32768.0
    elif dtype == 'int32':
        wav_data = wav_data / float(2**31)

    # Convert to mono
    assert channels in [1, 2], "channels must be 1 or 2"
    if len(wav_data.shape) > channels:
        wav_data = np.mean(wav_data, axis=1)

    if sr != sample_rate:
        wav_data = resampy.resample(wav_data, sr, sample_rate)

    return wav_data