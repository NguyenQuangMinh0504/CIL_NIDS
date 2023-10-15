import copy
import datetime


def train(args: dict):
    print("Training argument are:")
    print(args)
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args=args)


def _train(args: dict):
    pass
    time_str = datetime.datetime.now().strftime("%m%d%")