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

        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            wave = resampler(wave)

        if self.transforms is not None:
            wave = self.transforms(wave)

        return wave, self.sample_rate, duration
