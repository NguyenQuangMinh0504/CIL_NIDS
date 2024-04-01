import logging

import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from models.base import BaseLearner

from utils.inc_net import IncrementalNet
from utils.data_manager import DataManager
from utils.toolkit import tensor2numpy
from utils.prog_bar import prog_bar

# init_epoch = 200
init_weight_decay = 0.0005

# epochs = 80
# epochs = 30
# batch_size = 128
lrate_decay = 0.1
milestones = [40, 70]
num_workers = 4
weight_decay = 2e-4


class FineTune(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = IncrementalNet(args["convnet_type"], False)

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_training(self, data_manager: DataManager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        if self.args.get("exemplar_using") is True:
            train_dataset = data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes), source="train", mode="train", appendent=self._get_memory())
        else:
            train_dataset = data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_loader = DataLoader(train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        if self.args.get("exemplar_using") is True:
            self.build_rehearsal_memory(data_manager=data_manager, per_class=self.samples_per_class)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.args["init_lr"], weight_decay=init_weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args["init_epoch"])
            if self.args["skip"]:
                if len(self._multiple_gpus) > 1:
                    self._network = self._network.module
                load_acc = self._network.load_checkpoint(self.args)
                self._network.to(self._device)
                cur_test_acc = self._compute_accuracy(self._network, self.test_loader)
                logging.info(f"Loaded_Test_Acc: {load_acc} Cur_Test_Acc:{cur_test_acc}")
                if len(self._multiple_gpus) > 1:
                    self._network = nn.DataParallel(self._network, self._multiple_gpus)
            else:
                self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(self._network.parameters(), self.args["lrate"], momentum=0.9, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer: optim.SGD, scheduler):

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
            writer.add_scalar("Loss/train", losses / len(train_loader), epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            if (epoch + 1) % 5 != 0:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc
                )
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                writer.add_scalar("Accuracy/Test", test_acc, epoch)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc
                )
            logging.info(info)
        writer.close()

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        writer = SummaryWriter(log_dir="runs/{}/{}/{}_{}/Task{}".format(
            self.args["dataset"],
            self.args["model_name"],
            self.args["convnet_type"],
            self.args["batch_size"],
            self._cur_task)
            )

        for _, epoch in enumerate(prog_bar(self.args["epochs"])):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits = self._network(inputs)["logits"]
                # Implementation of Loss function
                if self.args.get("exemplar_using") is True:
                    loss = F.cross_entropy(logits, targets)
                elif self.args.get("regular_loss") is True:
                    loss = F.cross_entropy(logits, targets)
                else:
                    fake_targets = targets - self._known_classes
                    loss_clf = F.cross_entropy(logits[:, self._known_classes:], fake_targets)
                    loss = loss_clf

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
                # if round(losses, ndigits=3) == 0:
                #     logging.info(f"{logits}")
                #     logging.info(f"Target is: {targets}")
                #     logging.info(f"loss is: {loss}")
                #     logging.info(f"Pred is: {torch.max(logits, dim=1)}")
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            logging.info(info)
