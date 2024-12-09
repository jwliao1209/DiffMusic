from glob import glob
from typing import Callable, Optional
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torch
import torchaudio.transforms as T
from pydub import AudioSegment
import numpy as np


__DATASET__ = {}


def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: Dataset,
                   batch_size: int,
                   num_workers: int,
                   train: bool):
    dataloader = DataLoader(dataset,
                            batch_size,
                            shuffle=train,
                            num_workers=num_workers,
                            drop_last=train)
    return dataloader


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


@register_dataset(name='wav')
class WAVDataset(Dataset):
    def __init__(self, root: str, sample_rate: int, transforms: Optional[Callable] = None):
        self.root = root
        self.transforms = transforms
        self.sample_rate = sample_rate
        self.fpaths = sorted(glob(root + '/**/*.wav', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]

        wave, sr = torchaudio.load(fpath)  # wave shape: [channels, time]
        duration = wave.shape[-1] / sr

        if wave.size(0) > 1:
            wave = wave.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            wave = resampler(wave)

        if self.transforms is not None:
            mel_spectrogram = self.transforms(wave)

        _, phase = waveform_to_spectrogram(wave)

        return wave, mel_spectrogram, phase, self.sample_rate, duration
