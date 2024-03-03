from models.memo import MEMO
from models.der import DER
from models.foster import FOSTER
from models.finetune import FineTune
from models.lwf import LwF
from models.icarl import iCaRL


def get_model(model_name: str, args: dict):
    name = model_name.lower()
    if name == "memo":
        return MEMO(args)
    elif name == "der":
        return DER(args)
    elif name == "foster":
        return FOSTER(args)
    elif name == "finetune":
        return FineTune(args)
    elif name == "lwf":
        return LwF(args)
    elif name == "icarl":
        return iCaRL(args)
    else:
        raise NotImplementedError(f"Model '{model_name}' is not implement yet !!!")
