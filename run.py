import os
from pathlib import Path
from argparse import ArgumentParser, Namespace

import scipy
import torch
import torchaudio
import soundfile as sf
from hydra import initialize, compose

from diffmusic.constants import CONFIG_PATH
from diffmusic.data.dataloader import get_dataset, get_dataloader
from diffmusic.pipelines import get_pipeline
from diffmusic.inverse_problem import get_noiser
from diffmusic.inverse_problem.operator import (
    IdentityOperator,
    MusicInpaintingOperator,
    PhaseRetrievalOperator,
    SuperResolutionOperator,
    MusicDereverberationOperator,
    StyleGuidanceOperator,
)
from diffmusic.schedulers import get_scheduler
from diffmusic.utils import waveform_to_spectrogram
from diffmusic.constants import (
    MOISES, MUSICCAPS, AUDIOLDM2, MUSICLDM,
    MUSIC_GENERATION, MUSIC_INPAINTING, SUPER_RESOLUTION,
    PHASE_RETREVAL, MUSIC_DEREVERBERATION, STYLE_GUIDANCE,
    DDIM, DPS, MPGD, DSG, DIFFMUSIC, DITTO,
    # for ablation studies
    NULL_TEXT, TAG, CLAP,
    WAV_FORM, MEL_SPECTROGRAM,
)
import warnings

warnings.filterwarnings("ignore")


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_name",
        type=str,
        default=DIFFMUSIC,
        choices=[
            DDIM,
            DPS,
            MPGD,
            DSG,
            DITTO,
            DIFFMUSIC,
        ],
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
        "-d",
        "--datasets",
        type=str,
        default=MOISES,
        choices=[
            MOISES,
            MUSICCAPS,
        ],
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=AUDIOLDM2,
        choices=[
            AUDIOLDM2,
            MUSICLDM,
        ],
    )
    parser.add_argument(
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
        "--supervised_space",
        type=str,
        default=MEL_SPECTROGRAM,
        choices=[
            WAV_FORM,
            MEL_SPECTROGRAM,
        ],
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default=NULL_TEXT,
        choices=[
            NULL_TEXT,
            TAG,
            CLAP
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
    parser.add_argument(
        "--transcription",
        type=str,
        required=False,
        default="",
        help="Transcription for Text-to-Speech",
    )
    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="Show Progress",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    with initialize(config_path=CONFIG_PATH, version_base="1.1"):
        config = compose(config_name=args.config_name, overrides=[
            "data={}".format(args.datasets),
            "model={}".format(args.model)
        ])

    output_dir = Path("outputs", config.model.name, config.data.name, args.config_name, args.task)
    for d in ["wav_input", "wav_recon", "wav_label", "mel_input", "mel_recon", "mel_label"]:
        os.makedirs(Path(output_dir, d), exist_ok=True)

    Noiser = get_noiser(**config.inverse_problem.noise)

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
                audio_length_in_s=config.model.pipe.audio_length_in_s,
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
        case _:
            raise ValueError(f"Unknown task: {args.task}")

    # prepare the pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = get_pipeline(pip_name=config.model.name).from_pretrained(config.model.repo_id, torch_dtype=torch.float16)
    pipe.scheduler = get_scheduler(scheduler_name=config.name)(operator=Operator, **config.model.scheduler)
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
        type=config.data.type,
        root=config.data.root,
        sample_rate=config.data.sample_rate,
        audio_length_in_s=config.model.pipe.audio_length_in_s,
        start_s=config.data.start_s,
        end_s=config.data.end_s,
        transforms=None,
    )

    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    print("==================================================")
    print("| Model             : {}".format(config.model.name))
    print("| Data              : {}".format(config.data.name))
    print("| Task              : {}".format(args.task))
    print("| Scheduler         : {}".format(args.config_name))
    print("| Supervised Space  : {}".format(args.supervised_space))
    print("| Prompt Type       : {}".format(args.prompt_type))
    print("| Prompt            : '{}'".format(args.prompt))
    print("| Show Progress     : {}".format(args.show_progress))
    print('| Number of Samples : {}'.format(len(loader)))
    print("==================================================")

    # run the generation
    for i, (data, file_name) in enumerate(loader, start=1):
        print(f"=====> Inference for audio {i}")

        file_name = file_name[0]

        # Check if file already exists
        recon_path = Path(output_dir, 'wav_recon', file_name)
        if os.path.exists(recon_path):
            print("File {} already exists. Skipping.".format(file_name))
            continue

        gt_wave = data
        gt_mel_spectrogram = wav2mel(gt_wave)
        gt_mel_spectrogram = gt_mel_spectrogram[:, :, :int(config.model.pipe.audio_length_in_s * 100)].permute(0, 2,
                                                                                                               1).unsqueeze(
            0)
        pipe.save_mel_spectrogram(
            gt_mel_spectrogram,
            Path(output_dir, 'mel_label', file_name).with_suffix('.png'),
        )

        if args.task != PHASE_RETREVAL:
            ref_wave = Operator.forward(data)

            # TODO: move mel spectrogram to dataloader
            ref_mel_spectrogram = wav2mel(ref_wave)
            ref_mel_spectrogram = ref_mel_spectrogram[:, :, :int(config.model.pipe.audio_length_in_s * 100)].permute(0,
                                                                                                                     2,
                                                                                                                     1)

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
            ref_phase = ref_phase[:, :, : int(config.model.pipe.audio_length_in_s * 100)].to(device)

            ref_mel_spectrogram = ref_mel_spectrogram.unsqueeze(0)

            measurement = ref_wave.clone()
        elif args.task == PHASE_RETREVAL:
            magnitude = Operator.forward(data).to(device)
            measurement = magnitude.clone()
        else:
            raise ValueError(f"Unknown task: {args.task}")

        audio = pipe(
            latents=None,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            measurement=measurement,
            eta=config.scheduler.eta,
            ip_guidance_rate=config.scheduler.ip_guidance_rate,
            optim_prompt_learning_rate=config.scheduler.optim_prompt_learning_rate,
            generator=generator,
            optim_prompt=config.scheduler.optim_prompt,
            optim_outer_loop=config.scheduler.optim_outer_loop,
            show_progress=args.show_progress,
            prompt_type=args.prompt_type,
            supervised_space=args.supervised_space,
            **config.model.pipe,
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
        pred_mel_spectrogram = pred_mel_spectrogram[:, :, :int(config.model.pipe.audio_length_in_s * 100)].permute(0, 2,
                                                                                                                   1)
        pipe.save_mel_spectrogram(
            pred_mel_spectrogram,
            Path(output_dir, 'mel_recon', file_name).with_suffix('.png'),
        )

        # save the best audio sample (index 0) as a .wav file
        # TODO: refactor interface to save the music
        if config.model.name in [AUDIOLDM2, MUSICLDM]:
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


if __name__ == "__main__":
    main()
