"""
Calculate Kullback-Leibler Divergence betweeen two audio directories.
"""
import os

import numpy as np
import resampy
import soundfile as sf
import torch
from scipy import linalg
from torch import nn
from tqdm import tqdm


class KullbackLeiblerDivergence:
    def __init__(
        self,
        ckpt_dir=None,
        sample_rate=16000,
        channels=1,
        use_pca=False,
        use_activation=False,
        verbose=False,
    ):
        """
        Initialize KL
        -- ckpt_dir: folder where the downloaded checkpoints are stored
        -- sample_rate: one between [8000, 16000, 32000, 48000]. depending on the model set the sample rate to use
        -- channels: number of channels in an audio track
        -- use_pca: whether to apply PCA to the vggish embeddings
        -- use_activation: whether to use the output activation in vggish
        """
        assert sample_rate == 16000, "sample_rate must be 16000"

        self.sample_rate = sample_rate
        self.channels = channels
        self.verbose = verbose
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if self.verbose:
            print("[Kullback-Leibler Divergence] Using device: {}".format(self.device))
        if ckpt_dir is not None:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.hub.set_dir(ckpt_dir)
            self.ckpt_dir = ckpt_dir
        else:
            # by default `ckpt_dir` is `torch.hub.get_dir()`
            self.ckpt_dir = torch.hub.get_dir()
        self.__get_model(use_pca=use_pca, use_activation=use_activation)

    def __get_model(self, use_pca=False, use_activation=False):
        """
        Get ckpt and set model for the specified model_name

        Params:
        -- use_pca: whether to apply PCA to the vggish embeddings
        -- use_activation: whether to use the output activation in vggish
        """
        # S. Hershey et al., "CNN Architectures for Large-Scale Audio Classification", ICASSP 2017
        self.model = torch.hub.load(repo_or_dir="harritaylor/torchvggish", model="vggish")
        if not use_pca:
            self.model.postprocess = False
        if not use_activation:
            self.model.embeddings = nn.Sequential(*list(self.model.embeddings.children())[:-1])
        self.model.device = self.device

        self.model.to(self.device)
        self.model.eval()

    def get_embeddings(self, x, sr):
        """
        Get embeddings using VGGish, PANN, CLAP or EnCodec models.
        Params:
        -- x    : a list of np.ndarray audio samples
        -- sr   : sampling rate.
        """
        embd_lst = []
        try:
            for audio in tqdm(x, disable=(not self.verbose)):
                embd = self.model.forward(audio, sr)

                if self.verbose:
                    print(
                        "[Kullback-Leibler Divergence] Embedding shape: {}".format(
                            embd.shape
                        )
                    )
                
                if embd.device != torch.device("cpu"):
                    embd = embd.cpu()
                
                if torch.is_tensor(embd):
                    embd = embd.detach().numpy()
                
                embd_lst.append(embd)
        except Exception as e:
            print("[Kullback-Leibler Divergence] get_embeddings throw an exception: {}".format(str(e)))

        return np.concatenate(embd_lst, axis=0)

    def calculate_kl(
        self,
        embds_eval,
        embds_background,
        eps=1e-6,
    ):
        p = torch.tensor(embds_eval, dtype=torch.float32).softmax(dim=-1)
        q = torch.tensor(embds_background, dtype=torch.float32).softmax(dim=-1)
        return torch.nn.functional.kl_div(
            (p + eps).log(), (q + eps), reduction="sum"
        ) / len(p)

    def score(
        self,
        audio_background,
        audio_eval,
        background_embds_path=None,
        eval_embds_path=None,
        dtype="float32"
    ):
        """
        Computes the Kullback-Leibler Divergence (KL) between two directories of audio files.

        Parameters:
        - background_dir (str): Path to the directory containing background audio files.
        - eval_dir (str): Path to the directory containing evaluation audio files.
        - background_embds_path (str, optional): Path to save/load background audio embeddings (e.g., /folder/bkg_embs.npy). If None, embeddings won"t be saved.
        - eval_embds_path (str, optional): Path to save/load evaluation audio embeddings (e.g., /folder/test_embs.npy). If None, embeddings won"t be saved.
        - dtype (str, optional): Data type for loading audio. Default is "float32".

        Returns:
        - float: The Frechet Audio Distance (FAD) score between the two directories of audio files.
        """
        # Load or compute background embeddings
        if background_embds_path is not None and os.path.exists(background_embds_path):
            if self.verbose:
                print(f"[Frechet Audio Distance] Loading embeddings from {background_embds_path}...")
            embds_background = np.load(background_embds_path)
        else:
            embds_background = self.get_embeddings(audio_background, sr=self.sample_rate)
            if background_embds_path:
                os.makedirs(os.path.dirname(background_embds_path), exist_ok=True)
                np.save(background_embds_path, embds_background)

        # Load or compute eval embeddings
        if eval_embds_path is not None and os.path.exists(eval_embds_path):
            if self.verbose:
                print(f"[Frechet Audio Distance] Loading embeddings from {eval_embds_path}...")
            embds_eval = np.load(eval_embds_path)
        else:
            embds_eval = self.get_embeddings(audio_eval, sr=self.sample_rate)
            if eval_embds_path:
                os.makedirs(os.path.dirname(eval_embds_path), exist_ok=True)
                np.save(eval_embds_path, embds_eval)

        # Check if embeddings are empty
        if len(embds_background) == 0:
            print("[Kullback-Leibler Divergence] background set dir is empty, exiting...")
            return -1
        if len(embds_eval) == 0:
            print("[Kullback-Leibler Divergence] eval set dir is empty, exiting...")
            return -1

        # Compute KL score
        kl_score = self.calculate_kl(
            embds_eval,
            embds_background,
        )

        return float(kl_score)
