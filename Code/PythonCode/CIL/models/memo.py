import logging
import numpy as np
from models.base import BaseLearner
from utils.inc_net import AdaptiveNet
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


class MEMO(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = AdaptiveNet(convnet_type=args["convnet_type"], pretrained=False)
        logging.info(
            f">>> train generalized blocks:{self.args['train_base']} train_adaptive: {self.args['train_adaptive']}")

    def incremental_training(self, data_manager: DataManager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in self._network.AdaptiveExtractor[i].parameters():
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
        