import numpy as np


class MeanSquaredError:
    def __init__(self, reduction='mean'):
        assert reduction in ['mean', 'sum'], "reduction must be 'mean' or 'sum'"
        self.reduction = reduction

    def score(self, audio_background, audio_eval):
        audio_background = np.array(audio_background, dtype=np.float32)
        audio_eval = np.array(audio_eval, dtype=np.float32)

        # Sanitize input
        audio_eval = np.nan_to_num(audio_eval, nan=0.0, posinf=1.0, neginf=-1.0)
        audio_background = np.nan_to_num(audio_background, nan=0.0, posinf=1.0, neginf=-1.0)

        mse_scores = []
        for ref, est in zip(audio_background, audio_eval):
            min_len = min(len(ref), len(est))
            mse = np.mean((ref[:min_len] - est[:min_len]) ** 2)
            mse_scores.append(mse)

        mse_scores = np.array(mse_scores)

        if self.reduction == 'mean':
            return mse_scores.mean()
        elif self.reduction == 'sum':
            return mse_scores.sum()

