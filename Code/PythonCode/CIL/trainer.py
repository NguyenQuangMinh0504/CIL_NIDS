import copy
import datetime
import os
import json
import logging

import torch
from utils.data_manager import DataManager


def train(args: dict):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args=args)


def _train(args: dict):
    time_str = datetime.datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]
    args["time_str"] = time_str
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    exp_name = "{}_{}_{}_{}_B{}_Inc{}".format(
        args["time_str"],
        args["dataset"],
        args["convnet_type"],
        args["seed"],
        init_cls,
        args["increment"])
    args["exp_name"] = exp_name
    if args["debug"]:
        logfilename = "logs/debug/{}/{}/{}/{}".format(
            args["prefix"],
            args["dataset"],
            args["model_name"],
            args["exp_name"],
        )
    else:
        logfilename = "logs/{}/{}/{}/{}".format(
            args["prefix"],
            args["dataset"],
            args["model_name"],
            args["exp_name"],
        )
    args["logfilename"] = logfilename
    csv_name = "{}_{}_{}_B{}_Inc{}".format(
        args["dataset"],
        args["seed"],
        args["convnet_type"],
        init_cls,
        args["increment"]
    )
    args["csv_name"] = csv_name
    os.makedirs(logfilename, exist_ok=True)
    log_path = os.path.join(args["logfilename"], "main.log")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(filename)s] => %(message)s",
                        handlers=[logging.FileHandler(filename=log_path),
                                  logging.StreamHandler()]
                        )
    logging.info(f"Time Str >>> {args['time_str']}")
    # save config
    config_filepath = os.path.join(args["logfilename"], "configs.json")
    with open(config_filepath, "w") as fd:
        json.dump(args, fd, indent=2, sort_keys=True)

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(dataset_name=args["dataset"],
                               shuffle=args["shuffle"],
                               seed=args["seed"],
                               init_cls=args["init_cls"],
                               increment=args["increment"])


def _set_device(args: dict):
    device_type = args["device"]
    gpus = []
    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))
        gpus.append(device)
    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args: dict):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
