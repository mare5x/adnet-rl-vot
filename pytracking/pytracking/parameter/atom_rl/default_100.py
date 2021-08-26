from pytracking.parameter.atom import default as atom_default


def parameters():
    params = atom_default.parameters()

    params.update_probability = 1.0  # Model update probability (for testing purposes)

    return params

