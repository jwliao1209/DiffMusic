import os
from argparse import ArgumentParser, Namespace

import scipy
import torch
import soundfile as sf
from diffmusic.plpeline_audioldm2 import AudioLDM2Pipeline
from diffmusic.pipeline_musicldm import MusicLDMPipeline
from diffmusic.pipeline_stable_audio import StableAudioPipeline
from omegaconf import OmegaConf


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/audioldm2.yaml",
        choices=[
            "configs/audioldm2.yaml",
            "configs/musicldm.yaml",
            "configs/stable_audio.yaml",
        ],
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Western music, chill out, folk instrument R & B beat.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config = OmegaConf.load(args.config_path)
    os.makedirs("outputs", exist_ok=True)

    match config.name:
        case "audioldm2":
            Pipeline = AudioLDM2Pipeline
        case "stable_audio":
            Pipeline = StableAudioPipeline
        case "musicldm":
            Pipeline = MusicLDMPipeline
        case _:
            raise ValueError(f"Unknown pipeline name: {config.name}")

    # prepare the pipeline
    pipe = Pipeline.from_pretrained(config.repo_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # set the seed for generator
    generator = torch.Generator("cuda").manual_seed(0)

    # run the generation
    audio = pipe(
        args.prompt,
        negative_prompt=args.negative_prompt,
        generator=generator,
        **config.pipe,
    ).audios

    # save the best audio sample (index 0) as a .wav file
    save_path = f"outputs/{config.name}_sample_music.wav"

    # TODO: refactor interface to save the music
    if config.name in ["audioldm2", "musicldm"]:
        scipy.io.wavfile.write(save_path, rate=16000, data=audio[0])
    elif config.name == "stable_audio":
        sf.write(
            save_path,
            audio[0].T.float().cpu().numpy(),
            pipe.vae.sampling_rate
        )
