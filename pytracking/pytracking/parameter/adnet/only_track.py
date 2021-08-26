from .default import parameters as default_params

def parameters():
    params = default_params()

    # Don't forget to set appropriate model path
    # params.model_path = Path(env.network_path) / "adnet.pth"

    params.initial_fine_tuning = False  # Perform initial fine tuning  
    params.stop_unconfident = False  # Stop tracking procedure if below threshold
    params.redetection = False  # Perform redetection 
    params.fine_tuning = False  # Perform online fine tuning 

    return params