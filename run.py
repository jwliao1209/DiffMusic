import os
from argparse import ArgumentParser, Namespace

import scipy
import torch
import soundfile as sf
from omegaconf import OmegaConf

from diffmusic.pipelines.plpeline_audioldm2 import AudioLDM2Pipeline
from diffmusic.pipelines.pipeline_musicldm import MusicLDMPipeline
from diffmusic.pipelines.pipeline_stable_audio import StableAudioPipeline
from diffmusic.schedulers.scheduling_inpainting import DDIMInpaintingScheduler

from data.dataloader import get_dataset, get_dataloader
import torchaudio


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
    pipe.scheduler = DDIMInpaintingScheduler(**config.scheduler)
    pipe = pipe.to("cuda")

    # set the seed for generator
    generator = torch.Generator("cuda").manual_seed(0)

    # load wav files
    data_config = config.data
    # transform = torchaudio.transforms.MelSpectrogram(
    #     sample_rate=16000,
    #     n_mels=64,
    #     hop_length=441
    # )
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=64
    )
    # transform = None
    dataset = get_dataset(**data_config, sample_rate=16000, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    print('Number of samples: ', len(loader))

    # run the generation
    for i, (ref_wave, sr, duration) in enumerate(loader):
        config.pipe.audio_length_in_s = duration.item()
        print('sr: ', sr)
        print('ref_wave: ', ref_wave.shape)
        print('duration: ', duration)

        audio = pipe(
            # latents=latent,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            generator=generator,
            measurement=ref_wave,
            **config.pipe,
        ).audios

        # save the best audio sample (index 0) as a .wav file
        save_path = f"outputs/{config.name}_sample_music_{i}.wav"

        # TODO: refactor interface to save the music
        if config.name in ["audioldm2", "musicldm"]:
            scipy.io.wavfile.write(save_path, rate=16000, data=audio[0])
        elif config.name == "stable_audio":
            sf.write(
                save_path,
                audio[0].T.float().cpu().numpy(),
                pipe.vae.sampling_rate
            )
