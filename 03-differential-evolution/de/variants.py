"""The three compared DE versions required by the assignment."""

from .config import DEConfig

# Recommended settings from the course materials.
VARIANTS: dict[str, DEConfig] = {
    "rand1": DEConfig(strategy="rand1", f=0.5, cr=0.8),
    "best1": DEConfig(strategy="best1", f=0.5, cr=0.8),
    "jde": DEConfig(strategy="rand1", f=0.5, cr=0.9, adaptive=True, tau1=0.1, tau2=0.1, f_low=0.1, f_high=0.9),
}

VARIANT_TITLES: dict[str, str] = {
    "rand1": "DE/rand/1/bin",
    "best1": "DE/best/1/bin",
    "jde": "jDE (samoadaptivní)",
}


def variant_config(name: str, np_size: int | None = None) -> DEConfig:
    from dataclasses import replace

    try:
        config = VARIANTS[name]
    except KeyError:
        raise ValueError(f"unknown variant {name!r}, choose from {list(VARIANTS)}") from None
    return config if np_size is None else replace(config, np_size=np_size)
