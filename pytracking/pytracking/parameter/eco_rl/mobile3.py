from pytracking.parameter.eco import mobile3

def parameters():
    params = mobile3.parameters()

    params.update_probability = None  # Model update probability (for testing purposes)

    return params
