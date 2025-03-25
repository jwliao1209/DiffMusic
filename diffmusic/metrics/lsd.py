import librosa
import numpy as np


class LogSpectralDistance:
    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        hop_length=160,
        eps=1e-10,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps

    def score(self, audio_background, audio_eval, output_mean=True):
        audio_background = np.array(audio_background)
        audio_eval = np.array(audio_eval)

        # Compute STFT
        audio_eval = np.nan_to_num(audio_eval, nan=0.0, posinf=1.0, neginf=-1.0)
        background_spectrogram = np.abs(librosa.stft(audio_background, n_fft=self.n_fft, hop_length=self.hop_length))
        eval_spectrogram = np.abs(librosa.stft(audio_eval, n_fft=self.n_fft, hop_length=self.hop_length))

        # Convert to log scale (add epsilon to avoid log of zero)
        log_background = np.log10(background_spectrogram + self.eps)
        log_eval = np.log10(eval_spectrogram + self.eps)

        # Compute squared differences
        squared_diff = (log_background - log_eval) ** 2

        # Compute average over frequency bins and time frames
        lsd_per_frame = np.sqrt(np.mean(squared_diff, axis=1))
        lsd_score = np.mean(lsd_per_frame, axis=1)

        if output_mean:
            return lsd_score.mean()
        else:
            return lsd_score
