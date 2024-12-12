from glob import glob
from typing import Callable, Optional

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from pydub import AudioSegment
from torch.utils.data import DataLoader, Dataset


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
    train: bool = False,
    collate_fn: Optional[Callable] = None,
):
    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle=train,
        num_workers=num_workers,
        drop_last=train,
        collate_fn=collate_fn,
    )
    return dataloader


@register_dataset(name='wav')
class WAVDataset(Dataset):
    def __init__(
        self,
        root: str,
        sample_rate: int,
        audio_length_in_s: int,
        start_s: float,
        end_s: float,
        start_inpainting_s: float,
        end_inpainting_s: float,
        transforms: Optional[Callable] = None,
    ):
        self.root = root
        self.sample_rate = sample_rate
        self.audio_length_in_s = audio_length_in_s
        self.start_s = start_s
        self.end_s = end_s
        self.start_inpainting_s = start_inpainting_s
        self.end_inpainting_s = end_inpainting_s
        self.transforms = transforms

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

        wave = wave[0]
        ref_wave = wave.clone()
        ref_wave[ 
            int(self.start_inpainting_s * self.sample_rate) :
            int(self.end_inpainting_s * self.sample_rate)
        ] = 0.
        ref_wave = ref_wave[int(self.start_s * self.sample_rate) : int(self.end_s * self.sample_rate)]
        gt_wave = wave[int(self.start_s * self.sample_rate) : int(self.end_s * self.sample_rate)]
        
        data = {
            "gt_wave": gt_wave,
            "ref_wave": ref_wave,
            "gt_mel_spectrogram": gt_wave,
            "ref_mel_spectrogram": ref_wave,
            "start_inpainting_s": self.start_inpainting_s - self.start_s,
            "end_inpainting_s": self.end_inpainting_s - self.start_s,
            "duration": duration,
        }
        return self.transforms(data) if self.transforms is not None else data
