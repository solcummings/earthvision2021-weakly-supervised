from models.siamese_unet import UNet


def build(model_name: str, model_args: dict, **kwargs):
    model_dict = {
            'unet': UNet,
    }
    return model_dict[model_name.lower()](**(model_args or {}))


