import logging
from models.base import BaseLearner
from utils.inc_net import AdaptiveNet
from utils.data_manager import DataManager


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
