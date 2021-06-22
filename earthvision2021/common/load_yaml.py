import yaml


def load(config_file: str) -> dict:
    with open(config_file) as f:
        config = yaml.safe_load(f)
    return config


