from pytracking.utils import TrackerParams
from pytracking.evaluation.environment import env_settings
import os.path


def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False
    params.use_gpu = True

    env = env_settings()
    params.model_network_path = os.path.join(env.network_path, "siamfc_alexnet_e50.pth")

    params.frame_skipping = None
    params.update_probability = 1.0  # Model update probability (for testing purposes)

    return params
