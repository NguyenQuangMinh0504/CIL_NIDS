import copy
import logging
import torch
from torch import nn


from convs.linears import SimpleLinear
from convs.cifar_resnet import resnet32
from convs.memo_cifar_resnet import get_resnet32_a2fc as get_memo_resnet32
from convs.memo_kdd_fc import get_kdd_fc, get_memo_ann, get_memo_dnn
from convs.ann import get_ann, get_dnn


def get_convnet(convnet_type: str, pretrained: bool = False) -> tuple[nn.Module, nn.Module]:
    """Return generalize block + specialize block"""
    name = convnet_type.lower()
    if name == "resnet32":
        return resnet32()
    elif name == "memo_resnet32":
        return get_memo_resnet32()
    elif name == "kdd_fc":
        return get_kdd_fc()
    elif name == "kdd_memo_dnn":
        return get_memo_dnn(input_dim=121)
    elif name == "kdd_ann":
        return get_ann()
    elif name == "kdd_dnn":
        return get_dnn()
    elif name == "cic_ids_ann":
        return get_ann(input_dim=68)
    elif name == "ton_iot_network_ann":
        return get_ann(input_dim=248)
    elif name == "cic_ids_memo_ann":
        return get_memo_ann(input_dim=68)
    elif name == "ton_iot_network_memo_ann":
        return get_memo_ann(input_dim=248)
    elif name == "ton_iot_network_memo_dnn":
        return get_memo_dnn(input_dim=248)
    elif name == "cic_ids_dnn":
        return get_dnn(input_dim=68)
    elif name == "ton_iot_network_dnn":
        return get_dnn(input_dim=248)
    elif name == "cic_ids_memo_dnn":
        return get_memo_dnn(input_dim=68)
    elif name == "unsw_nb15_ann":
        return get_ann(input_dim=194)
    elif name == "unsw_nb15_dnn":
        return get_dnn(input_dim=194)
    elif name == "unsw_nb15_memo_ann":
        return get_memo_ann(input_dim=194)
    elif name == "unsw_nb15_memo_dnn":
        return get_memo_dnn(input_dim=194)
    else:
        raise NotImplementedError(f"Convnet type : {name} has not been implemented yet!!!")


class BaseNet(nn.Module):
    def __init__(self, convnet_type, pretrained):
        super(BaseNet, self).__init__()
        self.convnet = get_convnet(convnet_type=convnet_type, pretrained=pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self


class IncrementalNet(BaseNet):
    def __init__(self, convnet_type, pretrained, gradcam=False):
        super(IncrementalNet).__init__(convnet_type, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias
        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("align weights gamma = ", gamma)
        self.fc.weight.data[:-increment, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        out.update(x)
        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations
        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(backward_hook)
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(forward_hook)


class FOSTERNet(nn.Module):
    def __init__(self, convnet_type, pretrained):
        super(FOSTERNet, self).__init__()
        self.convnet_type = convnet_type
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.task_sizes = []
        self.oldfc = None

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        out = self.fc(features)
        fe_logits = self.fe_fc(features[:, -self.out_dim:])["logits"]

        out.update({"fe_logits": fe_logits, "features": features})

        if self.oldfc is not None:
            old_logits = self.oldfc(features[:, : -self.out_dim])["logits"]
            out.update({"old_logits": old_logits})

        out.update({"eval_logits": out["logits"]})
        return out

    def update_fc(self, nb_classes):
        # logging.info(f"convnets is: {get_convnet(self.convnet_type)}")
        # logging.info(f"self.convnets is: {self.convnets}")
        logging.info("Old fc is: {}".format(self.fc))
        self.convnets.append(get_convnet(self.convnet_type))
        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, :self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        self.oldfc = self.fc
        self.fc = fc
        logging.info("New fc is: {}".format(self.fc))
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.fe_fc = self.generate_fc(self.out_dim, nb_classes)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * (value ** (old / increment))
        logging.info("align weights, gamma = {}".format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma

    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            plt_name = "{}_{}_{}_B{}_Inc{}".format(
                args["dataset"],
                args["seed"],
                args["convnet_type"],
                0,
                args["init_cls"]
            )
            checkpoint_name = f"checkpoints/finetune_{plt_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.convnets) == 1
        self.convnets[0].load_state_dict(model_infos["convnet"])
        self.fc.load_state_dict(model_infos["fc"])
        test_acc = model_infos["test_acc"]
        return test_acc


class DERNet(nn.Module):
    convnets: nn.ModuleList

    def __init__(self, convnet_type, pretrained):
        super(DERNet, self).__init__()
        self.convnet_type = convnet_type
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)

        out = self.fc(features)
        aux_logits = self.aux_fc(features[:, -self.out_dim:])["logits"]
        out.update({"aux_logits": aux_logits, "features": features})
        return out

    def update_fc(self, nb_classes):
        logging.info(f"Convnet type is: {self.convnet_type}")
        logging.info(f"Convnet net is: {get_convnet(self.convnet_type)}")
        if len(self.convnets) == 0:
            self.convnets.append(get_convnet(self.convnet_type))
        else:
            self.convnets.append(get_convnet(self.convnet_type))
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[: nb_output] = bias
        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("align weights, gamma = ", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def load_checkpoint(self, args):
        checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.convnets) == 1
        self.convnets[0].load_state_dict(model_infos["convnet"])
        self.fc.load_state_dict(model_infos["fc"])
        test_acc = model_infos["test_acc"]
        return test_acc
