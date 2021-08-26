from pytracking.parameter.eco import default

def parameters():
    params = default.parameters()

    params.update_probability = 0  # Model update probability (for testing purposes)

    return params
