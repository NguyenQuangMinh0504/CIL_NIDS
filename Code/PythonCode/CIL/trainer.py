import copy
import datetime
import os
import json
import logging
import time

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters, save_fc, save_model
from utils.notify import send_telegram_notification


def train(args: dict):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args=args)


def _train(args: dict):
    send_telegram_notification(text="Start training")
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
    model = factory.get_model(model_name=args["model_name"], args=args)

    logging.info(f"Type of model is: {type(model)}")
    logging.info(f"Type of model network is: {type(model._network)}")

    cnn_curve, nme_curve, no_nme = {"top1": [], "top5": []}, {"top1": [], "top5": []}, True
    start_time = time.time()
    logging.info(f"Start time:{start_time}")

    count_parameters(model=model._network)

    logging.info(f"Data manager.nb_tasks is: {data_manager.nb_tasks}")

    for task in range(data_manager.nb_tasks):

        logging.info(f"current task is: {task}")
        logging.info("All params: {}".format(count_parameters(model=model._network, trainable=True)))
        logging.info("Trainable params: {}".format(count_parameters(model=model._network, trainable=True)))

        model.incremental_training(data_manager=data_manager)

        if task == data_manager.nb_tasks - 1:
            cnn_accy, nme_accy = model.eval_task(save_conf=True)
            no_nme = True if exp_name is None else False
        else:
            cnn_accy, nme_accy = model.eval_task(save_conf=False)

        model.after_task()

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            try:
                cnn_curve["top1"].append(cnn_accy["top1"])
                cnn_curve["top5"].append(cnn_accy["top5"])
                cnn_curve["top2"].append(cnn_accy["top2"])
            except KeyError:
                logging.info("Can not get key from cnn_curve !!!")

            try:
                nme_curve["top1"].append(nme_accy["top1"])
                nme_curve["top2"].append(nme_accy["top2"])
                nme_curve["top5"].append(nme_accy["top5"])
            except KeyError:
                logging.info("Can not get key from nme_curve !!!")

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            writer = SummaryWriter(log_dir="runs/{}/{}/{}_{}/Accuracy_curve".format(
                args["dataset"],
                args["model_name"],
                args["convnet_type"],
                args["batch_size"],)
            )
            for i, accy in enumerate(cnn_curve["top1"]):
                writer.add_scalar("Accuracy_Curve", accy, i)
            writer.close()

        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            try:
                cnn_curve["top1"].append(cnn_accy["top1"])
                cnn_curve["top5"].append(cnn_accy["top5"])
                cnn_curve["top2"].append(cnn_accy["top2"])
            except KeyError:
                logging.info("Can not get key from cnn_curve !!!")

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            writer = SummaryWriter(log_dir="runs/{}/{}/{}_{}/Accuracy_curve".format(
                args["dataset"],
                args["model_name"],
                args["convnet_type"],
                args["batch_size"],)
            )

            for i, accy in enumerate(cnn_curve["top1"]):
                writer.add_scalar("Accuracy_Curve", accy, i)

            writer.close()

    send_telegram_notification(text="Finish training")

    end_time = time.time()
    cost_time = end_time - start_time
    logging.info(f"End Time: {end_time}")
    save_time(args, cost_time)
    save_results(args, cnn_curve, nme_curve, no_nme)
    cost_time = end_time - start_time
    if args['model_name'] not in ["podnet", "coil"]:
        save_fc(args, model)
    else:
        save_model(args, model)


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
    logging.info("Calling function print args ....")
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))


def save_time(args, cost_time):
    _log_dir = os.path.join("./results/", "times", f"{args['prefix']}")
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    with open(_log_path, "a+") as f:
        f.write(f"{args['time_str']},{args['model_name']}, {cost_time} \n")


def save_results(args, cnn_curve, nme_curve, no_nme=False):
    cnn_top1, cnn_top5 = cnn_curve["top1"], cnn_curve['top5']
    nme_top1, nme_top5 = nme_curve["top1"], nme_curve['top5']

    # -------CNN TOP1----------
    _log_dir = os.path.join("./results/", f"{args['prefix']}", "cnn_top1")
    os.makedirs(_log_dir, exist_ok=True)

    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    if args['prefix'] == 'benchmark':
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},")
            for _acc in cnn_top1[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top1[-1]} \n")
    else:
        assert args['prefix'] in ['fair', 'auc']
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
            for _acc in cnn_top1[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top1[-1]} \n")

    # -------CNN TOP5----------
    _log_dir = os.path.join("./results/", f"{args['prefix']}", "cnn_top5")
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    if args['prefix'] == 'benchmark':
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},")
            for _acc in cnn_top5[:-1]:
                f.write(f"{_acc},")
            # f.write(f"{cnn_top5[-1]} \n")
    else:
        assert args['prefix'] in ['auc', 'fair']
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
            for _acc in cnn_top5[:-1]:
                f.write(f"{_acc},")
            # f.write(f"{cnn_top5[-1]} \n")

    # -------NME TOP1----------
    if no_nme is False:
        _log_dir = os.path.join("./results/", f"{args['prefix']}", "nme_top1")
        os.makedirs(_log_dir, exist_ok=True)
        _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
        if args['prefix'] == 'benchmark':
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},")
                for _acc in nme_top1[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top1[-1]} \n")
        else:
            assert args['prefix'] in ['fair', 'auc']
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
                for _acc in nme_top1[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top1[-1]} \n")

        # -------NME TOP5----------
        _log_dir = os.path.join("./results/", f"{args['prefix']}", "nme_top5")
        os.makedirs(_log_dir, exist_ok=True)
        _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
        if args['prefix'] == 'benchmark':
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},")
                for _acc in nme_top5[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top5[-1]} \n")
        else:
            assert args['prefix'] in ['auc', 'fair']
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
                for _acc in nme_top5[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top5[-1]} \n")
