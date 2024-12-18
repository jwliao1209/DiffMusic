from argparse import ArgumentParser, Namespace

from diffmusic.metrics.fad import FrechetAudioDistance
from diffmusic.metrics.kl import KullbackLeiblerDivergence
from diffmusic.metrics.lsd import LogSpectralDistance
from diffmusic.utils import load_audio_files


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-gt",
        "--ground_truth_dir",
        default="outputs/audioldm2/mpgd/music_inpainting/wav_label",
        type=str,
    )
    parser.add_argument(
        "-r",
        "--reconstruction_dir",
        default="outputs/audioldm2/mpgd/music_inpainting/wav_recon",
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    audio_background = load_audio_files(
        args.ground_truth_dir,
        sample_rate=16000,
    )
    audio_eval = load_audio_files(
        args.reconstruction_dir,
        sample_rate=16000,
    )

    frechet = FrechetAudioDistance(
        sample_rate=16000,
        use_pca=False, 
        use_activation=False,
        verbose=False
    )
    kl = KullbackLeiblerDivergence(
        sample_rate=16000,
        use_pca=False, 
        use_activation=False,
        verbose=False
    )
    lsd = LogSpectralDistance(
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        eps=1e-10,
    )

    fad_score = frechet.score(
        audio_background, 
        audio_eval,
    )
    kl_score = kl.score(
        audio_background, 
        audio_eval,
    )
    lsd_score = lsd.score(
        audio_background, 
        audio_eval,
    )
    print(f"lsd_score: {lsd_score:.4f}")
    print(f"fad_score: {fad_score:.4f}")
    print(f"kl_score: {kl_score:.4f}")
