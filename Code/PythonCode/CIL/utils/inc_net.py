import logging
import torch
from torch import nn


from convs.linears import SimpleLinear
from convs.memo_cifar_resnet import get_resnet32_a2fc as get_memo_resnet32


def get_convnet(convnet_type: str, pretrained: bool = False) -> (nn.Module, nn.Module):
    """Return generalize block + specialize block"""
    name = convnet_type.lower()
    if name == "resnet32":
        pass
    elif name == "memo_resnet32":
        return get_memo_resnet32()


class AdaptiveNet(nn.Module):
    TaskAgnosticExtractor: nn.Module
    """Generalized block"""
    AdaptiveExtractors: nn.ModuleList
    "Specialized block"
    fc: nn.Module
    "Fully connected block"

    def __init__(self, convnet_type: str, pretrained: bool):
        super(AdaptiveNet, self).__init__()
        self.convnet_type = convnet_type
        self.TaskAgnosticExtractor: nn.Module
        self.TaskAgnosticExtractor, _ = get_convnet(convnet_type=convnet_type, pretrained=pretrained)
        logging.info(f"Task Agnostic Extractor is: {type(self.TaskAgnosticExtractor)}")
        logging.info(f"Task agnostic extractor structure: {self.TaskAgnosticExtractor}")
        self.TaskAgnosticExtractor.train()
        self.AdaptiveExtractors = nn.ModuleList()
        self.output_dim = None

    @property
    def feature_dim(self):
        if self.output_dim is None:
            return 0
        return self.output_dim*len(self.AdaptiveExtractors)

    def update_fc(self, nb_classes):
        """Get specialize extractor -> """
        _, _new_extractor = get_convnet(self.convnet_type)
        if len(self.AdaptiveExtractors) == 0:
            self.AdaptiveExtractors.append(_new_extractor)
        else:
            self.AdaptiveExtractors.append(_new_extractor)
            self.AdaptiveExtractors[-1].load_state_dict(self.AdaptiveExtractors[-2].state_dict())
        if self.output_dim is None:
            # logging.info(self.AdaptiveExtractors[-1])
            self.output_dim = self.AdaptiveExtractors[-1].feature_dim
        self.generate_fc(in_dim=self.feature_dim, out_dim=nb_classes)

    def generate_fc(self, in_dim: int, out_dim: int):
        fc = SimpleLinear(in_features=in_dim, out_features=out_dim)
        return fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = (torch.norm(weights[-increment:, :], p=2, dim=1))
        oldnorm = (torch.norm(weights[-increment:, :], p=2, dim=1))
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold/meannew
        print('alignweights,gamma=', gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def forward(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        out = self.fc(features)  # {logits: self.fc(features)}

        aux_logits = self.aux_fc(features[:, -self.out_dim:])["logits"]

        out.update({"aux_logits": aux_logits, "features": features})
        out.update({"base_features": base_feature_map})
        return out

        '''
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        '''

    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format(args["dataset"],
                                                   args["seed"],
                                                   args["convnet_type"], 0, args["init_cls"])
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        checkpoint_name = checkpoint_name.replace("memo_", "")
        model_infos = torch.load(checkpoint_name)
        model_dict: dict = model_infos["convnet"]
        assert len(self.AdaptiveExtractors) == 1

        base_state_dict = self.TaskAgnosticExtractor.state_dict()
        adapt_base_dict = self.AdaptiveExtractors[0].state_dict()
        pretrained_base_dict = {
            k: v for k, v in model_dict.items() if k in base_state_dict
        }
        pretrained_adap_dict = {
            k: v for k, v in model_dict.items() if k in adapt_base_dict
        }
        base_state_dict.update(pretrained_base_dict)
        adapt_base_dict.update(pretrained_adap_dict)

        self.TaskAgnosticExtractor.load_state_dict(base_state_dict)
        self.AdaptiveExtractors.load_state_dict(adapt_base_dict)
        self.fc.load_state_dict(model_infos["fc"])
        test_acc = model_infos["test_acc"]
        return test_acc
