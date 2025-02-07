from .plpeline_audioldm2 import AudioLDM2Pipeline
from .pipeline_musicldm import MusicLDMPipeline


def get_pipeline(pip_name):
    match pip_name:
        case "audioldm2":
            return AudioLDM2Pipeline
        case "musicldm":
            return MusicLDMPipeline
        # TODO: Implement Stable Audio
        # case "stable_audio":
        #     Pipeline = StableAudioPipeline
        case _:
            raise ValueError(f"Unknown pipeline name: {pip_name}")
