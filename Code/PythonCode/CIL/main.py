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
    parser.add_argument("--init_epoch", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pre_processing", type=str, default="min_max")

    parser.add_argument("--init_lr", type=float, default=0.1)
    parser.add_argument("--exemplar_using", action="store_true")  # Using in finetune model for using exemplar or not
    parser.add_argument("--regular_loss", action="store_true")  # Using in lwf for testing purpose
    parser.add_argument("--temperature", type=float, default=2,
                        help="Temperature parameter T, introduced in famous paper distilling the knowledge in a neural network")

    # Training parameters
    parser.add_argument("--lrate", type=float, default=0.1, help="Lrate of SGD")
    parser.add_argument("--momentum", type=float, default=0, help="Momentum of SGD")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay of SGD")
    parser.add_argument("--lrate_decay", type=float, default=0.1, help="Lrate decay of MultiStepLR")
    parser.add_argument("--milestones", nargs="+", type=int, default=[100, 200], help="Milestones of MultiStepLR")
    return parser


if __name__ == '__main__':
    main()
