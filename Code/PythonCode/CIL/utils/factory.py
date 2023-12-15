from models.memo import MEMO
from models.memo_kdd import MEMO_KDD
from models.der import DER


def get_model(model_name: str, args: dict):
    name = model_name.lower()
    if name == "memo":
        return MEMO(args)
    elif name == "memo_kdd":
        return MEMO_KDD(args=args)
    elif name == "der":
        return DER(args)
