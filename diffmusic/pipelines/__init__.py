from .plpeline_audioldm2 import AudioLDM2Pipeline
from .pipeline_musicldm import MusicLDMPipeline


def get_pipeline(config):
    match config.name:
        case "audioldm2":
            return AudioLDM2Pipeline
        case "musicldm":
            return MusicLDMPipeline
        # case "stable_audio":
        #     Pipeline = StableAudioPipeline
        case _:
            raise ValueError(f"Unknown pipeline name: {config.name}")
