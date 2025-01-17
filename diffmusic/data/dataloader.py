from glob import glob
from typing import Callable, Optional

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset
from pydub import AudioSegment
import os


__DATASET__ = {}


def register_dataset(name: str) -> Callable:
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs) -> Dataset:
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    train: bool,
):
    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle=train,
        num_workers=num_workers,
        drop_last=train
    )
    return dataloader


@register_dataset(name='wav')
class WAVDataset(Dataset):
    def __init__(
        self,
        root: str,
        sample_rate: int,
        audio_length_in_s: int,
        start_s: float = 0,
        end_s: float = 0,
        transforms: Optional[Callable] = None,
    ):
        self.root = root
        self.sample_rate = sample_rate
        self.audio_length_in_s = audio_length_in_s
        self.start_s = start_s
        self.end_s = end_s
        self.transforms = transforms

        self.fpaths = sorted(glob(root + '/**/*.wav', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]

        wave, sr = torchaudio.load(fpath)  # wave shape: [channels, time]

        if wave.size(0) > 1:
            wave = wave.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            wave = resampler(wave)

        if self.transforms is not None:
            wave = self.transforms(wave)
        
        wave = wave[0]
        gt_wave = wave[int(self.start_s * self.sample_rate): int(self.end_s * self.sample_rate)]

        return gt_wave, os.path.basename(fpath)


@register_dataset(name='mp3')
class MP3Dataset(Dataset):
    def __init__(
        self,
        root: str,
        sample_rate: int,
        audio_length_in_s: int,
        start_s: float = 0,
        end_s: float = 0,
        transforms: Optional[Callable] = None,
    ):
        self.root = root
        self.sample_rate = sample_rate
        self.audio_length_in_s = audio_length_in_s
        self.start_s = start_s
        self.end_s = end_s
        self.transforms = transforms

        self.fpaths = sorted(glob(root + '/**/*.mp3', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]

        # Load MP3 file using pydub
        audio = AudioSegment.from_file(fpath, format="mp3")

        # Resample to the target sample rate if necessary
        if audio.frame_rate != self.sample_rate:
            audio = audio.set_frame_rate(self.sample_rate)

        # Convert to mono if necessary
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Convert audio to raw samples (numpy array)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)

        # Normalize samples to [-1, 1]
        samples /= np.iinfo(audio.array_type).max

        # Extract the target segment based on start and end time
        start_idx = int(self.start_s * self.sample_rate)
        end_idx = int(self.end_s * self.sample_rate) if self.end_s > 0 else len(samples)
        gt_wave = samples[start_idx:end_idx]

        # Apply transformations if provided
        if self.transforms is not None:
            gt_wave = self.transforms(torch.tensor(gt_wave))

        return torch.tensor(gt_wave), os.path.basename(fpath)
