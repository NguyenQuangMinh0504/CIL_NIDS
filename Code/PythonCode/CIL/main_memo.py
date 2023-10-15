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
    parser.add_argument("--model_name", "-model", type=str, default=None, required=True)
    parser.add_argument("--convnet_type", "-net", type=str, default="resnet32")
    parser.add_argument("--device", '-d', nargs='+', type=int, default=[0, 1, 2, 3])
    return parser


if __name__ == "__main__":
    main()
