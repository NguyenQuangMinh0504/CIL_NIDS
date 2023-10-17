from models.memo import MEMO


def get_model(model_name: str, args: dict):
    name = model_name.lower()
    if name == "memo":
        return MEMO(args)
