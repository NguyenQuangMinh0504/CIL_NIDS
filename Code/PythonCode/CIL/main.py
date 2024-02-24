import json
import argparse
from trainer import train


# Handle too many open files ??
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    args = setup_parser().parse_args()
    args.config = f"./exps/{args.model_name}.json"
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    param.update(args)
    train(param)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')

    parser.add_argument('--dataset', type=str, default="cifar100")
    parser.add_argument('--memory_size', '-ms', type=int, default=2000)
    parser.add_argument('--init_cls', '-init', type=int, default=10)
    parser.add_argument('--increment', '-incre', type=int, default=10)
    parser.add_argument('--model_name', '-model', type=str, default=None, required=True)
    parser.add_argument('--convnet_type', '-net', type=str, default='resnet32')
    parser.add_argument('--prefix', '-p', type=str, help='exp type', default='benchmark', choices=['benchmark', 'fair', 'auc'])
    parser.add_argument('--device', '-d', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--skip', action="store_true")
    parser.add_argument("--init_epoch", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=170)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--pre_processing", type=str, default="min_max_scale")
    parser.add_argument("--lrate", type=float, default=0.001)
    parser.add_argument("--init_lr", type=float, default=0.001)

    return parser


if __name__ == '__main__':
    main()
