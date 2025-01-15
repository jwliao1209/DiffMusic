import os
from pathlib import Path
from argparse import ArgumentParser, Namespace

import scipy
import torch
import torchaudio
import soundfile as sf
from omegaconf import OmegaConf

from diffmusic.data.dataloader import get_dataset, get_dataloader
from diffmusic.pipelines import get_pipeline
from diffmusic.operators.operator import (
    IdentityOperator,
    MusicInpaintingOperator,
    PhaseRetrievalOperator,
    SuperResolutionOperator,
    MusicDereverberationOperator,
    StyleGuidanceOperator,
    GaussianNoise,
    PoissonNoise,
)

from diffmusic.schedulers.scheduling_ddim import DDIMScheduler
from diffmusic.schedulers.scheduling_dps import DPSScheduler
from diffmusic.schedulers.scheduling_mpgd import MPGDScheduler
from diffmusic.schedulers.scheduling_dsg import DSGScheduler
from diffmusic.schedulers.scheduling_diffmusic import DiffMusicScheduler

from diffmusic.utils import waveform_to_spectrogram
from diffmusic.constants import (
    AUDIOLDM2, MUSICLDM,
    MUSIC_GENERATION, MUSIC_INPAINTING, SUPER_RESOLUTION,
    PHASE_RETREVAL, MUSIC_DEREVERBERATION, STYLE_GUIDANCE,
    DDIM, DPS, MPGD, DSG, DIFFMUSIC,
)

from diffmusic.msclap import CLAP


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--instrument",
        type=str,
        default="bass",
        choices=["bass", "bowed_strings", "drums", "guitar", "percussion", "piano", "wind"],
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default="music_inpainting",
        choices=[
            MUSIC_GENERATION,
            MUSIC_INPAINTING,
            SUPER_RESOLUTION,
            PHASE_RETREVAL,
            MUSIC_DEREVERBERATION,
            STYLE_GUIDANCE,
        ],
    )
    parser.add_argument(
        "-m",
        "--mask_type",
        type=str,
        default="box",
        choices=[
            "box",
            "random",
            "periodic",
        ],
    )
    parser.add_argument(
        "-s",
        "--scheduler",
        type=str,
        default=DPS,
        choices=[
            DDIM,
            DPS,
            MPGD,
            DSG,
            DIFFMUSIC,
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
            # "configs/stable_audio.yaml",
        ],
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-np",
        "--negative_prompt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--transcription",
        type=str,
        required=False,
        default="",
        help="Transcription for Text-to-Speech",
    )
    parser.add_argument(
        "--noise",
        type=str,
        required=False,
        default="gaussian",
        choices=["gaussian", "poisson"],
    )
    parser.add_argument(
        "--sigma",
        type=float,
        required=False,
        default=0.05,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config = OmegaConf.load(args.config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path("outputs", config.name, args.scheduler, args.task)
    for d in ["wav_input", "wav_recon", "wav_label", "mel_input", "mel_recon", "mel_label"]:
        os.makedirs(Path(output_dir, d), exist_ok=True)

    match args.noise:
        case "gaussian":
            Noiser = GaussianNoise(args.sigma)
        case "poisson":
            Noiser = PoissonNoise(args.sigma)
        case _:
            raise ValueError(f"Unknown noise: {args.noise}")

    match args.task:
        case "music_generation":
            start_inpainting_s = None
            end_inpainting_s = None
            downsample_scale = 1
            Operator = IdentityOperator(
                sample_rate=config.data.sample_rate,
            )
        case "music_inpainting":
            start_inpainting_s = config.data.start_inpainting_s - config.data.start_s
            end_inpainting_s = config.data.end_inpainting_s - config.data.start_s
            downsample_scale = 1
            Operator = MusicInpaintingOperator(
                audio_length_in_s=config.pipe.audio_length_in_s,
                sample_rate=config.data.sample_rate,
                mask_type=args.mask_type,
                # for box
                start_inpainting_s=start_inpainting_s,
                end_inpainting_s=end_inpainting_s,
                # for random
                mask_percentage=0.3,
                # for periodic
                interval_s=1,
                mask_duration_s=0.1,
                noiser=Noiser,
            )
        case "super_resolution":
            start_inpainting_s = None
            end_inpainting_s = None
            downsample_scale = 2
            Operator = SuperResolutionOperator(
                sample_rate=config.data.sample_rate,
                scale=downsample_scale,
                noiser=Noiser,
            )
        case "phase_retrieval":
            start_inpainting_s = None
            end_inpainting_s = None
            downsample_scale = 1
            Operator = PhaseRetrievalOperator(
                n_fft=config.data.n_fft,
                hop_length=config.data.hop_length,
                win_length=config.data.win_length,
                noiser=Noiser,
            )
        case "music_dereverberation":
            start_inpainting_s = None
            end_inpainting_s = None
            downsample_scale = 1
            Operator = MusicDereverberationOperator(
                ir_length=5000,
                decay_factor=0.99,
                noiser=Noiser,
            )
        case "style_guidance":
            start_inpainting_s = None
            end_inpainting_s = None
            downsample_scale = 1
            # "CLAP_weights_2022.pth", "CLAP_weights_2023.pth"
            clap_model = CLAP("CLAP_weights/CLAP_weights_2023.pth", version='2023', use_cuda=True)
            Operator = StyleGuidanceOperator(
                clap_model=clap_model,
            )
        case _:
            raise ValueError(f"Unknown task: {args.task}")

    match args.scheduler:
        case "ddim":
            Scheduler = DDIMScheduler
            eta = 1.0
        case "dps":
            Scheduler = DPSScheduler
            eta = 1.0
        case "mpgd":
            Scheduler = MPGDScheduler
            eta = 1.0
        case "dsg":
            Scheduler = DSGScheduler
            eta = 1.0
        case "diffmusic":
            Scheduler = DiffMusicScheduler
            eta = 1.0
        case _:
            raise ValueError(f"Unknown scheduler: {args.scheduler}")

    # prepare the pipeline
    pipe = get_pipeline(config=config).from_pretrained(config.repo_id, torch_dtype=torch.float16)
    pipe.scheduler = Scheduler(operator=Operator, **config.scheduler)
    pipe = pipe.to(device)

    # set the seed for generator
    generator = torch.Generator(device).manual_seed(0)

    # prepare the waveform to mel spectrogram transformation
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

    dataset = get_dataset(
        name=config.data.name,
        root=os.path.join(config.data.root, args.instrument),
        sample_rate=config.data.sample_rate,
        audio_length_in_s=config.pipe.audio_length_in_s,
        start_s=config.data.start_s,
        end_s=config.data.end_s,
        transforms=None,
    )

    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    print('Number of samples: ', len(loader))

    # run the generation
    for i, (data, file_name) in enumerate(loader, start=1):
        file_name = file_name[0]

        # Check if file already exists
        recon_path = Path(output_dir, 'wav_recon', file_name)
        if os.path.exists(recon_path):
            print("File {} already exists. Skipping.".format(file_name))
            continue

        gt_wave = data

        gt_mel_spectrogram = wav2mel(gt_wave)
        gt_mel_spectrogram = gt_mel_spectrogram[:, :, :int(config.pipe.audio_length_in_s * 100)].permute(0, 2, 1).unsqueeze(0)
        pipe.save_mel_spectrogram(
            gt_mel_spectrogram,
            Path(output_dir, 'mel_label', file_name).with_suffix('.png'),
        )

        if args.task != PHASE_RETREVAL:
            ref_wave = Operator.forward(data)

            # TODO: move mel spectrogram to dataloader
            ref_mel_spectrogram = wav2mel(ref_wave)
            ref_mel_spectrogram = ref_mel_spectrogram[:, :, :int(config.pipe.audio_length_in_s * 100)].permute(0, 2, 1)

            pipe.save_mel_spectrogram(
                ref_mel_spectrogram.unsqueeze(0),
                Path(output_dir, 'mel_input', file_name).with_suffix('.png'),
                sample_rate=config.data.sample_rate // downsample_scale,
                gt_mel_spectrogram=gt_mel_spectrogram,
                gt_sample_rate=config.data.sample_rate,
            )

            ref_wave = ref_wave.to(device)
            ref_mel_spectrogram = ref_mel_spectrogram.to(device)

            _, ref_phase = waveform_to_spectrogram(waveform=ref_wave)
            ref_phase = ref_phase[:, :, : int(config.pipe.audio_length_in_s * 100)].to(device)

            ref_mel_spectrogram = ref_mel_spectrogram.unsqueeze(0)

            measurement = ref_wave.clone()
        elif args.task == PHASE_RETREVAL:
            magnitude = Operator.forward(data).to(device)
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
            Path(output_dir, 'wav_label', file_name),
            rate=config.data.sample_rate,
            data=gt_wave.cpu().detach().numpy()[0],
        )

        # save degraded inputs
        if args.task != PHASE_RETREVAL:
            # reconstructed_degraded_waveform_with_phase = pipe.mel_spectrogram_to_waveform_with_phase(
            #     ref_mel_spectrogram.cpu(), ref_phase.cpu(),
            # )

            scipy.io.wavfile.write(
                Path(output_dir, 'wav_input', file_name),
                rate=config.data.sample_rate // downsample_scale,
                data=ref_wave.cpu().detach().numpy()[0],
            )

        # save the predicted mel spectrogram
        pred_mel_spectrogram = wav2mel(torch.tensor(audio))
        pred_mel_spectrogram = pred_mel_spectrogram[:, :, :int(config.pipe.audio_length_in_s * 100)].permute(0, 2, 1)
        pipe.save_mel_spectrogram(
            pred_mel_spectrogram,
            Path(output_dir, 'mel_recon', file_name).with_suffix('.png'),
        )

        # save the best audio sample (index 0) as a .wav file
        # TODO: refactor interface to save the music
        if config.name in [AUDIOLDM2, MUSICLDM]:
            # save outputs
            scipy.io.wavfile.write(
                Path(output_dir, 'wav_recon', file_name),
                rate=config.data.sample_rate,
                data=audio[0],
            )

        # elif config.name == "stable_audio":
        #     sf.write(
        #         Path(output_dir, 'wav_recon', file_name),
        #         audio[0].T.float().cpu().numpy(),
        #         pipe.vae.sampling_rate,
        #     )