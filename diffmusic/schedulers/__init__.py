from .scheduling_ddim import DDIMScheduler
from .scheduling_dps import DPSScheduler
from .scheduling_mpgd import MPGDScheduler
from .scheduling_dsg import DSGScheduler
from .scheduling_ditto import DITTOScheduler
from .scheduling_diffmusic import DiffMusicScheduler


def get_scheduler(scheduler_name):
    match scheduler_name:
        case "ddim":
            return DDIMScheduler
        case "dps":
            return DPSScheduler
        case "mpgd":
            return MPGDScheduler
        case "dsg":
            return DSGScheduler
        case "ditto":
            return DITTOScheduler
        case "diffmusic":
            return DiffMusicScheduler
        case _:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
