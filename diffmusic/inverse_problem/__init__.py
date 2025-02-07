from .noise import GaussianNoise, PoissonNoise


def get_noiser(name, sigma):
    match name:
        case "gaussian":
            return GaussianNoise(sigma)
        case "poisson":
            return PoissonNoise(sigma)
        case _:
            raise ValueError(f"Unknown noise: {name}")
