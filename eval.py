from argparse import ArgumentParser, Namespace

from diffmusic.metrics.fad import FrechetAudioDistance


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-gt",
        "--ground_truth_dir",
        default="data",
        type=str,
    )
    parser.add_argument(
        "-r",
        "--reconstruction_dir",
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    frechet = FrechetAudioDistance(
        sample_rate=16000,
        use_pca=False, 
        use_activation=False,
        verbose=False
    )
    fad_score = frechet.score(
        args.ground_truth_dir, 
        args.reconstruction_dir,
        dtype="float32"
    )

    print(f"fad_score: {fad_score}")
