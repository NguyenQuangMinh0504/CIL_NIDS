import argparse
import json
from trainer import train


def main():
    args: argparse.Namespace = setup_parser().parse_args()
    args.config = f"./exps/{args.model_name}.json"
    param: dict = load_json(args.config)
    args = vars(args)
    param.update(args)
    train(args=param)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce of multiple continual learning algorithms.")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--init_cls", "-init", type=int, default=10)
    parser.add_argument("--increment", "-incre", type=int, default=10)
    parser.add_argument("--model_name", "-model", type=str, default=None, required=True)
    parser.add_argument("--convnet_type", "-net", type=str, default="resnet32")
    parser.add_argument("--prefix", "-p", type=str, help="exp type", default="benchmark",
                        choices=["benchmark", "fair", "auc"])
    parser.add_argument("--device", "-d", nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument("--debug", action="store_true")

    # Added in MEMO
    parser.add_argument("--train_base", action="store_true")
    parser.add_argument("--train_adaptive", action="store_true")

    return parser


if __name__ == "__main__":
    main()
