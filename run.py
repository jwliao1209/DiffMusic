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
from diffmusic.utils.utils import waveform_to_spectrogram


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
        # default="Western music, chill out, folk instrument R & B beat.",
        default="",
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
    os.makedirs("results", exist_ok=True)

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
    wav2mel = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=config.data.sample_rate,
            n_fft=config.data.n_fft,
            hop_length=config.data.hop_length,
            win_length=config.data.win_length,
            n_mels=config.data.n_mels,
            power=config.data.power,
        ),
        torchaudio.transforms.AmplitudeToDB(stype="power")
    )

    # transform = None
    dataset = get_dataset(
        name=config.data.name,
        root=config.data.root,
        sample_rate=config.data.sample_rate,
        audio_length_in_s=config.pipe.audio_length_in_s,
        start_s=config.data.start_s,
        end_s=config.data.end_s,
        start_inpainting_s=config.data.start_inpainting_s,
        end_inpainting_s=config.data.end_inpainting_s,
        transforms=None,
    )

    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    print('Number of samples: ', len(loader))

    # run the generation
    for i, data in enumerate(loader, start=1):
        gt_wave = data["gt_wave"]
        ref_wave = data["ref_wave"]
        duration = data["duration"]

        gt_mel_spectrogram = wav2mel(gt_wave)
        gt_mel_spectrogram = gt_mel_spectrogram[:, :, :int(config.pipe.audio_length_in_s * 100)].permute(0, 2, 1).unsqueeze(0)
        pipe.save_mel_spectrogram(gt_mel_spectrogram, f"results/gt_mel_spectrogram_{i}.png")

        ref_mel_spectrogram = wav2mel(ref_wave)
        ref_mel_spectrogram = ref_mel_spectrogram[:, :, :int(config.pipe.audio_length_in_s * 100)].permute(0, 2, 1)

        ref_wave = ref_wave.to("cuda")
        ref_mel_spectrogram = ref_mel_spectrogram.to("cuda")

        _, ref_phase = waveform_to_spectrogram(waveform=ref_wave)
        ref_phase = ref_phase[:, :, : int(config.pipe.audio_length_in_s * 100)].to("cuda")

        ref_wave = ref_wave.unsqueeze(0)
        ref_mel_spectrogram = ref_mel_spectrogram.unsqueeze(0)
        pipe.save_mel_spectrogram(ref_mel_spectrogram, f"results/input_mel_spectrogram_{i}.png")

        # initialize the latents
        # latents = pipe.vae.encode(ref_mel_spectrogram.half()).latent_dist.sample(generator)
        # latents = pipe.vae.config.scaling_factor * latents

        audio = pipe(
            latents=None,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            generator=generator,
            ref_wave=ref_wave,
            ref_mel_spectrogram=ref_mel_spectrogram,
            ref_phase=ref_phase,
            start_inpainting_s=data["start_inpainting_s"],
            end_inpainting_s=data["end_inpainting_s"],
            **config.pipe,
        ).audios

        # save inputs
        scipy.io.wavfile.write(
            f"outputs/{config.name}_gt_music_{i}.wav",
            rate=config.data.sample_rate,
            data=gt_wave.cpu().detach().numpy()[0],
        )

        # save degraded inputs (inpainting)
        reconstructed_degraded_waveform_with_phase = pipe.mel_spectrogram_to_waveform_with_phase(
            ref_mel_spectrogram.cpu(), ref_phase.cpu())

        scipy.io.wavfile.write(
            f"outputs/{config.name}_input_music_{i}.wav",
            rate=config.data.sample_rate,
            data=reconstructed_degraded_waveform_with_phase.cpu().detach().numpy()[0],
        )

        # save the predicted mel spectrogram
        pred_mel_spectrogram = wav2mel(torch.tensor(audio))
        pred_mel_spectrogram = pred_mel_spectrogram[:, :, :int(config.pipe.audio_length_in_s * 100)].permute(0, 2, 1)
        pipe.save_mel_spectrogram(pred_mel_spectrogram, f"results/pred_mel_spectrogram_{i}.png")

        # save the best audio sample (index 0) as a .wav file
        # TODO: refactor interface to save the music
        save_path = f"outputs/{config.name}_sample_music_{i}.wav"
        if config.name in ["audioldm2", "musicldm"]:
            # save outputs
            scipy.io.wavfile.write(
                save_path,
                rate=config.data.sample_rate,
                data=audio[0],
            )

        elif config.name == "stable_audio":
            sf.write(
                save_path,
                audio[0].T.float().cpu().numpy(),
                pipe.vae.sampling_rate
            )
