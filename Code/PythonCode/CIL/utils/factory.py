from models.memo import MEMO
from models.memo_kdd import MEMO_KDD
from models.der import DER
from models.foster import FOSTER
from models.finetune import FineTune
from models.lwf import LwF


def get_model(model_name: str, args: dict):
    name = model_name.lower()
    if name == "memo":
        return MEMO(args)
    elif name == "memo_kdd":
        return MEMO_KDD(args)
    elif name == "der":
        return DER(args)
    elif name == "foster":
        return FOSTER(args)
    elif name == "finetune":
        return FineTune(args)
    elif name == "lwf":
        return LwF(args)
    else:
        raise NotImplementedError(f"Model '{model_name}' is not implement yet !!!")
