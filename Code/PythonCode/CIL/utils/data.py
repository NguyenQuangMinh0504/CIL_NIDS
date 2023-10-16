from torchvision import datasets


class iData(object):
    pass


class iCIFAR10(iData):
    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)

