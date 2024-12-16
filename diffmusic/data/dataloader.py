from glob import glob
from typing import Callable, Optional

import torchaudio
import torchaudio.transforms as T
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

        return gt_wave


@register_dataset(name='source_separation')
class SourceSeparationDataset(Dataset):
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
        fpath1 = self.fpaths[index]
        fpath2 = self.fpaths[index+1] if index + 1 < len(self.fpaths) else self.fpaths[index-1]

        wave1, sr1 = torchaudio.load(fpath1)  # wave shape: [channels, time]
        wave2, sr2 = torchaudio.load(fpath2)  # wave shape: [channels, time]

        wave1 = wave1.mean(dim=0, keepdim=True) if wave1.size(0) > 1 else wave1
        wave2 = wave2.mean(dim=0, keepdim=True) if wave2.size(0) > 1 else wave2

        if sr1 != self.sample_rate:
            resampler = T.Resample(orig_freq=sr1, new_freq=self.sample_rate)
            wave1 = resampler(wave1)
        if sr2 != self.sample_rate:
            resampler = T.Resample(orig_freq=sr2, new_freq=self.sample_rate)
            wave2 = resampler(wave2)

        if self.transforms is not None:
            wave1 = self.transforms(wave1)
            wave2 = self.transforms(wave2)

        wave1 = wave1[0]
        wave2 = wave2[0]

        gt_wave1 = wave1[int(self.start_s * self.sample_rate): int(self.end_s * self.sample_rate)]
        gt_wave2 = wave2[int(self.start_s * self.sample_rate): int(self.end_s * self.sample_rate)]

        return [gt_wave1, gt_wave2]

