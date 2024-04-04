import logging
import socket
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy
from utils.notify import send_telegram_notification
from utils.prog_bar import prog_bar
from utils.data_manager import DataManager

EPSILON = 1e-8

num_workers = 4
T = 2


class iCaRL(BaseLearner):
    """
    Implementation of paper iCaRL: Incremental Classifier and Representation Learning
    https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html
    """
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args["convnet_type"], False)
        self._old_network = None

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_training(self, data_manager: DataManager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
        )

        if self.args['skip'] and self._cur_task == 0:
            load_acc = self._network.load_checkpoint(self.args)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        if self._cur_task == 0:
            if self.args['skip']:
                self._network.to(self._device)
                cur_test_acc = self._compute_accuracy(self._network, self.test_loader)
                logging.info(f"Loaded_Test_Acc:{load_acc} Cur_Test_Acc:{cur_test_acc}")
            else:
                self._train(self.train_loader, self.test_loader)
                self._compute_accuracy(self._network, self.test_loader)
        else:
            self._train(self.train_loader, self.test_loader)

        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=self.args["momentum"],
                lr=self.args["init_lr"],
                weight_decay=self.args["weight_decay"],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.args["milestones"],
                gamma=self.args["lrate_decay"]
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.args["lrate"],
                momentum=self.args["momentum"],
                weight_decay=self.args["weight_decay"],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lrate_decay"]
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):

        writer = SummaryWriter(log_dir="runs/{}/{}/{}_{}/Task{}".format(
            self.args["dataset"],
            self.args["model_name"],
            self.args["convnet_type"],
            self.args["batch_size"],
            self._cur_task)
            )

        message = ""
        message += f"Instance: {socket.gethostname()} \n"
        message += f"Dataset: {self.args['dataset']} \n"
        message += f"Convnet type: {self.args['convnet_type']} \n"
        message += f"Model: {self.args['model_name']} \n"
        message += f"Current task: {self._cur_task} \n"
        send_telegram_notification(text=message)

        for _, epoch in enumerate(prog_bar(self.args["init_epoch"])):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            # Log to tensorboard
            writer.add_scalar("Loss/train", losses / len(train_loader), epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)

            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
                writer.add_scalar("Accuracy/Test", test_acc, epoch)
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                )

            logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):

        message = ""
        message += f"Instance: {socket.gethostname()} \n"
        message += f"Dataset: {self.args['dataset']} \n"
        message += f"Convnet type: {self.args['convnet_type']} \n"
        message += f"Model: {self.args['model_name']} \n"
        message += f"Current task: {self._cur_task} \n"
        send_telegram_notification(text=message)

        writer = SummaryWriter(log_dir="runs/{}/{}/{}_{}/Task{}".format(
            self.args["dataset"],
            self.args["model_name"],
            self.args["convnet_type"],
            self.args["batch_size"],
            self._cur_task)
            )

        for _, epoch in enumerate(prog_bar(self.args["init_epoch"])):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss_clf = F.cross_entropy(logits, targets)
                loss_kd = _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network(inputs)["logits"],
                    T,
                )

                loss = loss_clf + loss_kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            writer.add_scalar("Loss/train", losses / len(train_loader), epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)

            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                writer.add_scalar("Accuracy/Test", test_acc, epoch)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            logging.info(info)


def _KD_loss(pred, soft, T):
    """Calculating knowledge distillation loss"""
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
