import logging
import socket

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from models.base import BaseLearner
from torch.utils.data import DataLoader
from utils.adaptive_net import AdaptiveNet
from utils.data_manager import DataManager
from utils.toolkit import count_parameters, tensor2numpy
from utils.notify import send_telegram_notification
from typing import Union
from utils.prog_bar import prog_bar

num_workers = 4


class MEMO(BaseLearner):
    _network: Union[AdaptiveNet, nn.DataParallel]

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        logging.info(f"Arguments is: {args}")
        self._network = AdaptiveNet(convnet_type=args["convnet_type"], pretrained=False)
        logging.info(
            f">>> train generalized blocks:{self.args['train_base']} train_adaptive: {self.args['train_adaptive']}")

    def after_task(self):
        """After task"""
        logging.info("Running after task....")
        if self._cur_task != 0:
            # Weight align
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(self._total_classes - self._known_classes)
            else:
                self._network.weight_align(self._total_classes - self._known_classes)
        self._known_classes = self._total_classes
        if self._cur_task == 0:
            if self.args["train_base"]:
                logging.info("Train Generalized Blocks...")
                self._network.TaskAgnosticExtractor.train()
                for param in self._network.TaskAgnosticExtractor.parameters():
                    param.requires_grad = True
            else:
                logging.info("Fix Generalized Blocks...")
                self._network.TaskAgnosticExtractor.eval()
                for param in self._network.TaskAgnosticExtractor.parameters():
                    param.requires_grad = False
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_training(self, data_manager: DataManager):
        """Training model for current task"""
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        logging.info(f"Current total task is: {self._total_classes}")

        self._network.update_fc(self._total_classes)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        #  Freeze previous Adaptive Extractor weight
        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in self._network.AdaptiveExtractors[i].parameters():
                    if self.args["train_adaptive"]:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False

        logging.info("All params: {}".format(count_parameters(self._network, trainable=False)))
        logging.info("Trainable params: {}".format(count_parameters(self._network, trainable=True)))

        train_dataset = data_manager.get_dataset(
            indices=np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory()
        )
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.args["batch_size"],
                                       shuffle=True, num_workers=num_workers)

        test_dataset = data_manager.get_dataset(
            indices=np.arange(0, self._total_classes),
            source="test",
            mode="test",
        )
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.args["batch_size"],
                                      shuffle=False, num_workers=num_workers)
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(module=self._network, device_ids=self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager=data_manager, per_class=self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def set_network(self):
        """Update train() and eval() state of the model."""
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._network.train()  # All status from eval to train
        if self.args["train_base"]:
            self._network.TaskAgnosticExtractor.train()
        else:
            self._network.TaskAgnosticExtractor.eval()

        # set adaptive extractor's status
        self._network.AdaptiveExtractors[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                if self.args["train_adaptive"]:
                    self._network.AdaptiveExtractors[i].train()
                else:
                    self._network.AdaptiveExtractors[i].eval()
        if len(self._multiple_gpus) > 1:
            self._network == nn.DataParallel(self._network, self._multiple_gpus)

    def _train(self, train_loader: DataLoader, test_loader: DataLoader):
        """Training model ..."""
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(params=filter(lambda p: p.requires_grad, self._network.parameters()),
                                  lr=self.args["lrate"],
                                  momentum=self.args["momentum"],
                                  weight_decay=self.args["weight_decay"]
                                  )

            # Steplr Reference: https://hasty.ai/docs/mp-wiki/scheduler/multisteplr

            # Set up scheduler
            if self.args["scheduler"] == "steplr":
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer,
                    milestones=self.args["milestones"],
                    gamma=self.args["lrate_decay"],
                )
            elif self.args["scheduler"] == "cosine":
                assert self.args["t_max"] is not None
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=self.args["t_max"]
                )
            else:
                raise NotImplementedError

            if not self.args["skip"]:
                self._init_train(train_loader, test_loader, optimizer, scheduler)
            else:
                if isinstance(self._network, nn.DataParallel):
                    self._network = self._network.module
                load_acc = self._network.load_checkpoint(self.args)
                self._network.to(self._device)

                if len(self._multiple_gpus) > 1:
                    self._network = nn.DataParallel(self._network, self._multiple_gpus)
                cur_test_acc = self._compute_accuracy(self._network, self.test_loader)
                logging.info(f"Loaded_Test_Acc:{load_acc} Cur_Test_Acc:{cur_test_acc}")
        else:
            optimizer = optim.SGD(
                params=filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.args["lrate"],
                momentum=self.args["momentum"],
                weight_decay=self.args["weight_decay"],
            )
            if self.args["scheduler"] == "steplr":
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer,
                    milestones=self.args["milestones"],
                    gamma=self.args["lrate_decay"],
                )
            elif self.args["scheduler"] == "cosine":
                assert self.args["t_max"] is not None
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=self.args["t_max"],
                )
            else:
                raise NotImplementedError
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader: DataLoader, test_loader: DataLoader, optimizer, scheduler):

        logging.info("Initialize training.........................")

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
            losses = 0
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

            # Generate info message
            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                writer.add_scalar("Accuracy/Test", test_acc, epoch)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader), train_acc, test_acc
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task, epoch + 1, self.args["init_epoch"], losses / len(train_loader), train_acc
                )
            logging.info(info)

    def _update_representation(self, train_loader: DataLoader,
                               test_loader: DataLoader, optimizer: optim.SGD, scheduler):

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
            self.set_network()
            losses = 0.
            losses_clf = 0.
            losses_aux = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits, aux_logits = outputs["logits"], outputs["aux_logits"]
                loss_clf = F.cross_entropy(logits, targets)
                aux_targets = targets.clone()
                aux_targets = torch.where(condition=aux_targets-self._known_classes + 1 > 0,
                                          input=aux_targets - self._known_classes + 1, other=0)
                loss_aux = F.cross_entropy(aux_logits, aux_targets)
                loss = loss_clf + self.args["alpha_aux"] * loss_aux
                logits = outputs["logits"]
                loss = F.cross_entropy(logits, targets)
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
                writer.add_scalar("Accuracy/Test", test_acc, epoch)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses/len(train_loader),
                    losses_clf/len(train_loader),
                    losses_aux/len(train_loader),
                    train_acc,
                    test_acc
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses/len(train_loader),
                    losses_clf/len(train_loader),
                    loss_aux/len(train_loader),
                    train_acc,
                )
            # prog_bar.set_description(info)
            logging.info(info)
