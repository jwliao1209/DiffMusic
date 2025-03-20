from argparse import ArgumentParser, Namespace

from diffmusic.metrics.fad import FrechetAudioDistance
from diffmusic.metrics.kl import KullbackLeiblerDivergence
from diffmusic.metrics.lsd import LogSpectralDistance
from diffmusic.utils import load_audio_files

# For fadtk
import time
from fadtk.fad import FrechetAudioDistanceTK, log
from fadtk.model_loader import *
from fadtk.fad_batch import cache_embedding_files


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--inf', action='store_true', help="Use FAD-inf extrapolation")
    parser.add_argument('--indiv', action='store_true',
                        help="Calculate FAD for individual songs and store the results in the given file")
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

    """
    For FAD Toolkit.
    
    Models with reference data: [
    'clap-2023', 'clap-laion-audio', 'clap-laion-music', 'vggish', 
    'MERT-v1-95M-{1~11}', 'MERT-v1-95M', 
    'encodec-emb', 'encodec-emb-48k', 
    'w2v2-base-{1~11}', 'w2v2-base', 
    'w2v2-large-{1~23}', 'w2v2-large', 
    'hubert-base-{1~11}', 'hubert-base', 
    'hubert-large-{1~23}', 'hubert-large', 
    'wavlm-base-{1~11}', 'wavlm-base', 
    'wavlm-base-plus-{1~11}', 'wavlm-base-plus', 
    'wavlm-large-{1~23}', 'wavlm-large', 
    'whisper-{tiny/small/base/medium/large}']
    """

    fad_models = (
        "clap-2023",
        "clap-laion-audio",
        "clap-laion-music",
        "vggish",
        "encodec-emb",
        "w2v2-base",
        "hubert-base",
        "wavlm-base",
        "whisper-base"
    )
    fad_scores = {}
    models = {m.name: m for m in get_all_models()}

    baseline = args.ground_truth_dir
    eval = args.reconstruction_dir

    for fad_model in fad_models:
        model = models[fad_model]

        # 1. Calculate embedding files for each dataset
        for d in [baseline, eval]:
            if Path(d).is_dir():
                cache_embedding_files(d, model, workers=1)

        # 2. Calculate FAD
        fad = FrechetAudioDistanceTK(model, audio_load_worker=1, load_model=False)
        if args.inf:
            assert Path(eval).is_dir(), "FAD-inf requires a directory as the evaluation dataset"
            score = fad.score_inf(baseline, list(Path(eval).glob('*.*')))
            print("FAD-inf Information:", score)
            score, inf_r2 = score.score, score.r2
        elif args.indiv:
            assert Path(eval).is_dir(), "Individual FAD requires a directory as the evaluation dataset"
            csv_path = Path(args.csv or 'fad-individual-results.csv')
            fad.score_individual(baseline, eval, csv_path)
            log.info(f"Individual FAD scores saved to {csv_path}")
            exit(0)
        else:
            score = fad.score(baseline, eval)
            inf_r2 = None

        # 3. Save results
        fad_scores[fad_model] = score

    """
    For FAD, LSD, KL score.
    """
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

    # Print results
    key_width = max(len(key) for key in fad_scores.keys())

    # Print results
    print("")
    print("===> For FAD Toolkit.")
    for fad_model, score in fad_scores.items():
        print(f"{fad_model.ljust(key_width)} : {score:.4f}")

    print("")
    print("===> For FAD, LSD, KL score.")
    print(f"{'lsd_score'.ljust(key_width)} : {lsd_score:.4f}")
    print(f"{'fad_score'.ljust(key_width)} : {fad_score:.4f}")
    print(f"{'kl_score'.ljust(key_width)} : {kl_score:.4f}")


