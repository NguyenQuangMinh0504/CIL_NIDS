import logging
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
    AdaptiveExtractor: nn.ModuleList
    "Specialized block"
    def __init__(self, convnet_type: str, pretrained: bool):
        super(AdaptiveNet, self).__init__()
        self.convnet_type = convnet_type
        self.TaskAgnosticExtractor: nn.Module
        self.TaskAgnosticExtractor, _ = get_convnet(convnet_type=convnet_type, pretrained=pretrained)
        self.TaskAgnosticExtractor.train()
        self.AdaptiveExtractor = nn.ModuleList()
        self.output_dim = None
    
    @property
    def feature_dim(self):
        if self.output_dim is None:
            return 0
        return self.output_dim*len(self.AdaptiveExtractor)

    def update_fc(self, nb_classes):
        _, _new_extractor = get_convnet(self.convnet_type)
        if len(self.AdaptiveExtractor) == 0:
            self.AdaptiveExtractor.append(_new_extractor)
        else:
            self.AdaptiveExtractor.append(_new_extractor)
            self.AdaptiveExtractor[-1].load_state_dict(self.AdaptiveExtractor[-2].state_dict())
        if self.output_dim is None:
            logging.info(self.AdaptiveExtractor[-1])
            self.output_dim = self.AdaptiveExtractor[-1].feature_dim
        self.generate_fc(in_dim=self.feature_dim, out_dim=nb_classes)

    def generate_fc(self, in_dim: int, out_dim: int):
        fc = SimpleLinear(in_features=in_dim, out_features=out_dim)
        return fc
