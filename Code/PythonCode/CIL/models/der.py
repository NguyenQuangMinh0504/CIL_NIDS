import logging
import numpy as np
from torch import nn
import torch
import socket
from torch.nn import functional as F
from torch import optim
from models.base import BaseLearner
from utils.inc_net import DERNet
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from utils.data_manager import DataManager
from utils.toolkit import count_parameters, tensor2numpy
from utils.notify import send_telegram_notification
from utils.prog_bar import prog_bar

num_workers = 4


class DER(BaseLearner):
    """Implementation of Paper DER: Dynamically Expandable Representation for Class Incremental Learning"""
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = DERNet(convnet_type=args["convnet_type"], pretrained=False)

    def after_task(self):
        logging.info("Calling function after task ...")
        if self._cur_task != 0:
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(self._total_classes - self._known_classes)
            else:
                self._network.weight_align(self._total_classes - self._known_classes)
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_training(self, data_manager: DataManager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in self._network.convnets[i].parameters():
                    p.requires_grad = False
        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info("Trainable params: {}".format(count_parameters(self._network, True)))
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train", appendent=self._get_memory())
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.args["batch_size"],
                                       shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=self.args["batch_size"],
                                      shuffle=False, num_workers=num_workers)
        if len(self._multiple_gpus) > 1:
            self._network == nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters()),
                                  momentum=self.args["momentum"],
                                  lr=self.args["init_lr"],
                                  weight_decay=self.args["weight_decay"])
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lrate_decay"]
            )
            if not self.args["skip"]:
                self._init_train(train_loader, test_loader, optimizer, scheduler)
            else:
                test_acc = self._network.load_checkpoint(self.args)
                cur_test_acc = self._compute_accuracy(self._network, test_loader)
                logging.info(f"Loaded Test Acc: {test_acc} Cur_Test_Acc: {cur_test_acc}")
        else:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.args["lrate"],
                momentum=self.args["momentum"],
                weight_decay=self.args["weight_decay"]
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lrate_decay"]
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def train(self):
        self._network.train()
        if len(self._multiple_gpus) > 1:
            self._network_module_ptr = self._network.module
        else:
            self._network_module_ptr = self._network
        self._network_module_ptr.convnets[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                self._network_module_ptr.convnets[i].eval()

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        logging.info("Calling function _init_train ...")

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
            self.train()
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

            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                writer.add_scalar("Accuracy/Test", test_acc, epoch)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                )

            logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer: optim.SGD, scheduler):

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

        for _, epoch in enumerate(prog_bar(self.args["epochs"])):
            self.train()
            losses = 0.0
            losses_clf = 0.0
            losses_aux = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs. targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits = outputs["logits"]
                aux_logits = outputs["aux_logits"]
                loss_clf = F.cross_entropy(logits, targets)
                aux_targets = targets.clone()
                aux_targets = torch.where(aux_targets - self._known_classes + 1 > 0,
                                          aux_targets - self._known_classes + 1, 0)
                loss_aux = F.cross_entropy(aux_logits, aux_targets)
                if self.args.get("regular_loss") is True:
                    loss = loss_clf
                else:
                    loss = loss_clf + loss_aux
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_aux += loss_aux.item()
                losses_clf += loss_clf.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            writer.add_scalar("Loss/train", losses / len(train_loader), epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)

            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_aux / len(train_loader),
                    train_acc,
                    test_acc
                )
                writer.add_scalar("Accuracy/Test", test_acc, epoch)
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_aux / len(train_loader),
                    train_acc,
                )
            logging.info(info)
