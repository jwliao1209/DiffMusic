import os
from argparse import ArgumentParser, Namespace

import scipy
import torch
import torchaudio
import soundfile as sf
from omegaconf import OmegaConf

from diffmusic.data.dataloader import get_dataset, get_dataloader
from diffmusic.pipelines.plpeline_audioldm2 import AudioLDM2Pipeline
from diffmusic.pipelines.pipeline_musicldm import MusicLDMPipeline
from diffmusic.pipelines.pipeline_stable_audio import StableAudioPipeline
from diffmusic.schedulers.scheduling_inpainting import MusicInpaintingScheduler


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="music_inpainting",
        choices=[
            "music_inpainting",
            "phase_retrieval",
            "super_resolution",
        ],
    )
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

    match args.task:
        case "music_inpainting":
            Scheduler = MusicInpaintingScheduler
        # TODO: implement the following tasks
        case "phase_retrieval":
            Scheduler = None
        case "super_resolution":
            Scheduler = None
        case _:
            raise ValueError(f"Unknown task: {args.task}")

    # prepare the pipeline
    pipe = Pipeline.from_pretrained(config.repo_id, torch_dtype=torch.float16)
    pipe.scheduler = Scheduler(**config.scheduler)
    pipe = pipe.to("cuda")

    # set the seed for generator
    generator = torch.Generator("cuda").manual_seed(0)

    # load wav files
    transform = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=160,
            win_length=1024,
            n_mels=64,
            power=2.0,
        ),
        torchaudio.transforms.AmplitudeToDB(stype="power")
    )

    # transform = None
    dataset = get_dataset(**config.data, sample_rate=16000, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    print('Number of samples: ', len(loader))

    # run the generation
    for i, (ref_wave, ref_mel_spectrogram, ref_phase, sr, duration) in enumerate(loader):
        config.pipe.audio_length_in_s = duration.item()
        ref_wave = ref_wave[:, 0].to("cuda")
        ref_mel_spectrogram = ref_mel_spectrogram[:, :, :, :int(duration.item()*100)].permute(0, 1, 3, 2).to("cuda")
        ref_phase = ref_phase[:, :, :, : int(duration.item() * 100)].to("cuda")

        # initialize the latents
        latents = pipe.vae.encode(ref_mel_spectrogram.half()).latent_dist.sample(generator)
        latents = pipe.vae.config.scaling_factor * latents

        audio = pipe(
            latents=latents,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            generator=generator,
            measurement=ref_mel_spectrogram,
            ref_phase=ref_phase,
            **config.pipe,
        ).audios

        # save the best audio sample (index 0) as a .wav file
        save_path = f"outputs/{config.name}_sample_music_{i + 1}.wav"

        # TODO: refactor interface to save the music
        if config.name in ["audioldm2", "musicldm"]:
            # save outputs
            scipy.io.wavfile.write(save_path, rate=16000, data=audio[0])

            # save inputs
            reconstructed_waveform_with_phase = pipe.mel_spectrogram_to_waveform_with_phase(
                ref_mel_spectrogram.cpu(), ref_phase.cpu()
            )
            scipy.io.wavfile.write(
                f"outputs/{config.name}_inputs_music_{i + 1}.wav",
                rate=16000,
                data=reconstructed_waveform_with_phase.cpu().detach().numpy()[0],
            )

            # save degraded inputs (inpainting)
            # simulated degradation
            # (1, 1, 3000, 64)
            start_sample = 1000
            end_sample = 1500

            # create mask
            mask = torch.ones_like(ref_mel_spectrogram).to(ref_mel_spectrogram.device)
            mask[:, :, start_sample: end_sample, :] = 0.

            ref_mel_spectrogram[mask == 0] = -80.

            reconstructed_degraded_waveform_with_phase = pipe.mel_spectrogram_to_waveform_with_phase(
                ref_mel_spectrogram.cpu(), ref_phase.cpu())

            scipy.io.wavfile.write(
                f"outputs/{config.name}_degraded_inputs_music_{i + 1}.wav",
                rate=16000,
                data=reconstructed_degraded_waveform_with_phase.cpu().detach().numpy()[0],
            )

        elif config.name == "stable_audio":
            sf.write(
                save_path,
                audio[0].T.float().cpu().numpy(),
                pipe.vae.sampling_rate
            )

        break
