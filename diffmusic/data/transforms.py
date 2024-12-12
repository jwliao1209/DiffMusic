from typing import List

import torch
import numpy as np
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchvision.transforms import Compose


class BaseTransform:
    def __init__(self, keys, **kwargs):
        self.keys = keys
        self._parse_var(**kwargs)

    def __call__(self, data, **kwargs):
        for key in self.keys:
            if key in data:
                data[key] = self._process(data[key], **kwargs)
            else:
                raise KeyError(f"{key} is not a key in data.")
        return data

    def _parse_var(self, **kwargs):
        pass

    def _process(self, single_data, **kwargs):
        NotImplementedError

        
class Audio2MelSpectrogram(BaseTransform):
    def __init__(
        self,
        keys: List[str],
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        n_mels: int,
        power: int,
        audio_length_in_s: float,
    ):
        super().__init__(keys)
        self.transform = Compose(
            [
                MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mels=n_mels,
                    power=power
                ),
                AmplitudeToDB(stype="power"),
            ]
        )
        self.num_frames = int(sample_rate / hop_length * audio_length_in_s)

    def _process(self, single_data):
        return self.transform(single_data)[:, :self.num_frames].T
