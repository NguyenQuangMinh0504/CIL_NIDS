import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
from utils.inc_net import AdaptiveNet
from utils.data_manager import DataManager
import os

EPSILON = 1e-8
batch_size = 64


class BaseLearner(object):
    _network: AdaptiveNet
    test_loader: DataLoader

    def __init__(self, args: dict):
        self.args = args
        self._cur_task: int = -1
        """Current trained task"""
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(self._targets_memory), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def eval_task(self, save_conf=False):
        """Evaluating result"""

        logging.info(f"Test loader is: {self.test_loader}")

        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        if save_conf:
            _pred = y_pred.T[0]
            _pred_path = os.path.join(self.args['logfilename'], "pred.npy")
            _target_path = os.path.join(self.args['logfilename'], "target.npy")
            np.save(_pred_path, _pred)
            np.save(_target_path, y_true)

            _save_dir = os.path.join(f"./results/conf_matrix/{self.args['prefix']}")
            os.makedirs(_save_dir, exist_ok=True)
            _save_path = os.path.join(_save_dir, f"{self.args['csv_name']}.csv")
            with open(_save_path, "a+") as f:
                f.write(f"{self.args['time_str']},{self.args['model_name']},{_pred_path},{_target_path} \n")
        return cnn_accy, nme_accy

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return self._data_memory, self._targets_memory

    def _compute_accuracy(self, model: nn.Module, loader: DataLoader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
                predicts = torch.max(outputs, dim=1)[1]
                correct += (predicts.cpu() == targets).sum()
                total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader: DataLoader):
        """Returning y prediction and y true"""
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            logging.info(f"Inputs is: {inputs}")

            with torch.no_grad():
                outputs = self._network(inputs)["logits"]

            logging.info(f"Outputs is: {outputs}")
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader=loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        dists = cdist(class_means, vectors, "sqeuclidean")
        scores = dists.T
        return np.argsort(scores, axis=1)[:, : self.topk], y_true

    def _extract_vectors(self, loader):
        """Passing input and return layer before fully connected layer"""
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(self._network.module.extract_vector(_inputs.to(self._device)))
            else:
                _vectors = tensor2numpy(self._network.extract_vector(_inputs.to(self._device)))
            vectors.append(_vectors)
            targets.append(_targets)
        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager: DataManager, m):
        logging.info(f"Reducing exemplars...({m} per classes)")
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(self._targets_memory)
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]

            self._data_memory = (
                np.concatenate((self._data_memory, dd)) if len(self._data_memory) != 0 else dd
            )

            self._targets_memory = (
                np.concatenate((self._targets_memory, dt)) if len(self._targets_memory) != 0 else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(indices=[], source="train", mode="test", appendent=(dd, dt))
            idx_loader = DataLoader(dataset=idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)
            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager: DataManager, m):
        logging.info(f"Constructing exemplars...({m} per classes)")
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                indices=np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True
            )

            logging.info(f"Data is: {data}")
            logging.info(f"Target is: {targets}")

            idx_loader = DataLoader(dataset=idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)

            # Normalize vector
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(np.array(data[i]))  # new object to avoid passing by inference
                exemplar_vectors.append(np.array(vectors[i]))  # new object to avoid passsing by inference
                vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
                data = np.delete(data, i, axis=0)  # Remove it to avoid duplicative selection
                if len(vectors) == 0:
                    break

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(selected_exemplars.shape[0], class_idx)

            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0 else selected_exemplars)

            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0 else exemplar_targets)

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                indices=[],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )

            idx_loader = DataLoader(dataset=idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager: DataManager, m):
        logging.info(f"Constructing exemplars for new classes...({m} per classes)")

        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (self._data_memory[mask], self._targets_memory[mask])
            class_dset = data_manager.get_dataset(
                indices=[], source="train", mode="test", appendent=(class_data, class_targets)
                )
            class_loader = DataLoader(dataset=class_dset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = np / np.linalg.norm(mean)
            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                indices=(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True
            )
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(np.array(data[i]))  # new object to avoid passing by inference
                exemplar_vectors.append(np.array(vectors[i]))  # new object to avoid passing by inference
                vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
                data = np.delete(data, i, axis=0)  # Remove it to avoid duplicative selection
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)

            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0 else selected_exemplars)
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0 else exemplar_targets
                )

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset(
                indices=[],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets)
            )
            exemplar_loader = DataLoader(
                dataset=exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean
        self._class_means = _class_means
