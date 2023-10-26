import json
import numpy as np
from torch import nn
import os
import torch
import logging


class ConfigEncoder(json.JSONEncoder):
    pass


def count_parameters(model: nn.Module, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def accuracy(y_pred, y_true, nb_old, increment=10):

    logging.info(f"y pred size is: {y_pred.size()}")
    logging.info(f"y true size is: {y_true.size()}")
    logging.info(f"y pred type is: {type(y_pred)}")
    logging.info(f"y true type is: {type(y_true)}")

    assert len(y_pred) == len(y_true), "Data length error"
    all_acc = {}
    all_acc["total"] = np.around((y_pred == y_true).sum() * 100 / len(y_true), decimals=2)

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = "{}-{}".format(str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0"))
        all_acc[label] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)
    )

    # new accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc


def save_fc(args, model):
    _path = os.path.join(args['logfilename'], "fc.pt")
    if len(args['device']) > 1:
        fc_weight = model._network.fc.weight.data
    else:
        fc_weight = model._network.fc.weight.data.cpu()
    torch.save(fc_weight, _path)

    _save_dir = os.path.join(f"./results/fc_weights/{args['prefix']}")
    os.makedirs(_save_dir, exist_ok=True)
    _save_path = os.path.join(_save_dir, f"{args['csv_name']}.csv")
    with open(_save_path, "a+") as f:
        f.write(f"{args['time_str']},{args['model_name']},{_path} \n")


def save_model(args, model):
    # used in PODNet
    _path = os.path.join(args['logfilename'], "model.pt")
    if len(args['device']) > 1:
        weight = model._network
    else:
        weight = model._network.cpu()
    torch.save(weight, _path)
