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
from diffmusic.operators.operator import (
    MusicInpaintingOperator,
    MusicPhaseRetrievalOperator,
    MusicSuperResolutionOperator,
    MusicDereverberationOperator,
    MusicSourceSeparationOperator
)
from diffmusic.schedulers.scheduling_dps import DPSScheduler
from diffmusic.schedulers.scheduling_mpgd import MPGDScheduler
from diffmusic.schedulers.scheduling_dsg import DSGScheduler
from diffmusic.utils.utils import waveform_to_spectrogram


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default="music_inpainting",
        choices=[
            "music_inpainting",
            "phase_retrieval",
            "super_resolution",
            "dereverberation",
            "source_separation"
        ],
    )
    parser.add_argument(
        "-s",
        "--scheduler",
        type=str,
        default="dps",
        choices=[
            "dps",
            "mpgd",
            "dsg",
        ],
    )
    parser.add_argument(
        "-c",
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
        "-p",
        "--prompt",
        type=str,
        default="",
    )
    parser.add_argument(
        "-np",
        "--negative_prompt",
        type=str,
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    args = parse_arguments()
    config = OmegaConf.load(args.config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            start_inpainting_s = config.data.start_inpainting_s - config.data.start_s
            end_inpainting_s = config.data.end_inpainting_s - config.data.start_s
            downsample_scale = 1
            Operator = MusicInpaintingOperator(
                audio_length_in_s=config.pipe.audio_length_in_s,
                sample_rate=config.data.sample_rate,
                start_inpainting_s=start_inpainting_s,
                end_inpainting_s=end_inpainting_s,
            )
        case "phase_retrieval":
            start_inpainting_s = None
            end_inpainting_s = None
            downsample_scale = 1
            Operator = MusicPhaseRetrievalOperator(
                n_fft=config.data.n_fft,
                hop_length=config.data.hop_length,
                win_length=config.data.win_length,
            )
        case "super_resolution":
            start_inpainting_s = None
            end_inpainting_s = None
            downsample_scale = 5
            Operator = MusicSuperResolutionOperator(
                sample_rate=config.data.sample_rate,
                scale=downsample_scale,
            )
        case "dereverberation":
            start_inpainting_s = None
            end_inpainting_s = None
            downsample_scale = 1
            Operator = MusicDereverberationOperator(
                ir_length=5000,
                decay_factor=0.99,
            )
        case "source_separation":
            start_inpainting_s = None
            end_inpainting_s = None
            downsample_scale = 1
            config.pipe.num_waveforms_per_prompt = 2
            Operator = MusicSourceSeparationOperator(
                num_mix=2,
            )
        case _:
            raise ValueError(f"Unknown task: {args.task}")
        
    match args.scheduler:
        case "dps":
            Scheduler = DPSScheduler
            eta = 0.0
        case "mpgd":
            Scheduler = MPGDScheduler
            eta = 0.0
        case "dsg":
            Scheduler = DSGScheduler
            eta = 1.0
        case _:
            raise ValueError(f"Unknown scheduler: {args.scheduler}")

    # prepare the pipeline
    pipe = Pipeline.from_pretrained(config.repo_id, torch_dtype=torch.float16)
    pipe.scheduler = Scheduler(operator=Operator, **config.scheduler)
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

    if args.task == "source_separation":
        dataset = get_dataset(
            name="source_separation",
            root=config.data.root,
            sample_rate=config.data.sample_rate,
            audio_length_in_s=config.pipe.audio_length_in_s,
            start_s=config.data.start_s,
            end_s=config.data.end_s,
            transforms=None,
        )
    else:
        dataset = get_dataset(
            name=config.data.name,
            root=config.data.root,
            sample_rate=config.data.sample_rate,
            audio_length_in_s=config.pipe.audio_length_in_s,
            start_s=config.data.start_s,
            end_s=config.data.end_s,
            transforms=None,
        )

    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    print('Number of samples: ', len(loader))

    # run the generation
    for i, data in enumerate(loader, start=1):
        if args.task == "source_separation":
            gt_wave, other_wave = data
        else:
            gt_wave, other_wave = data, None

        gt_mel_spectrogram = wav2mel(gt_wave)
        gt_mel_spectrogram = gt_mel_spectrogram[:, :, :int(config.pipe.audio_length_in_s * 100)].permute(0, 2, 1).unsqueeze(0)
        pipe.save_mel_spectrogram(gt_mel_spectrogram, f"results/{config.name}_gt_mel_spectrogram_{i}.png")

        if args.task != "phase_retrieval":
            ref_wave = Operator.forward(data)

            # TODO: move mel spectrogram to dataloader
            ref_mel_spectrogram = wav2mel(ref_wave)
            ref_mel_spectrogram = ref_mel_spectrogram[:, :, :int(config.pipe.audio_length_in_s * 100)].permute(0, 2, 1)

            pipe.save_mel_spectrogram(ref_mel_spectrogram.unsqueeze(0),
                                      f"results/{config.name}_input_mel_spectrogram_{i}.png",
                                      sample_rate=config.data.sample_rate // downsample_scale,
                                      gt_mel_spectrogram=gt_mel_spectrogram,
                                      gt_sample_rate=config.data.sample_rate)

            ref_wave = ref_wave.to("cuda")
            ref_mel_spectrogram = ref_mel_spectrogram.to("cuda")

            _, ref_phase = waveform_to_spectrogram(waveform=ref_wave)
            ref_phase = ref_phase[:, :, : int(config.pipe.audio_length_in_s * 100)].to("cuda")

            ref_mel_spectrogram = ref_mel_spectrogram.unsqueeze(0)

            measurement = ref_wave.clone()
        elif args.task == "phase_retrieval":
            magnitude = Operator.forward(data).to("cuda")
            measurement = magnitude.clone()
        else:
            raise ValueError(f"Unknown task: {args.task}")

        # initialize the latents
        # latents = pipe.vae.encode(ref_mel_spectrogram.half()).latent_dist.sample(generator)
        # latents = pipe.vae.config.scaling_factor * latents

        audio = pipe(
            latents=None,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            measurement=measurement,
            eta=eta,
            generator=generator,
            **config.pipe,
        ).audios

        # save inputs
        scipy.io.wavfile.write(
            f"outputs/{config.name}_gt_music_{i}.wav",
            rate=config.data.sample_rate,
            data=gt_wave.cpu().detach().numpy()[0],
        )

        # save degraded inputs
        if args.task != "phase_retrieval":
            reconstructed_degraded_waveform_with_phase = pipe.mel_spectrogram_to_waveform_with_phase(
                ref_mel_spectrogram.cpu(), ref_phase.cpu())

            scipy.io.wavfile.write(
                f"outputs/{config.name}_input_music_{i}.wav",
                rate=config.data.sample_rate // downsample_scale,
                data=reconstructed_degraded_waveform_with_phase.cpu().detach().numpy()[0],
            )

        # save the predicted mel spectrogram
        pred_mel_spectrogram = wav2mel(torch.tensor(audio))
        pred_mel_spectrogram = pred_mel_spectrogram[:, :, :int(config.pipe.audio_length_in_s * 100)].permute(0, 2, 1)
        pipe.save_mel_spectrogram(pred_mel_spectrogram, f"results/{config.name}_sample_mel_spectrogram_{i}.png")

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
