# encoding: utf-8

import argparse
import sys

sys.path.append('.')
from tsf_baselines.utils.config_tools import setup_config
from tsf_baselines.engine.engine import Engine

def main():
    parser = argparse.ArgumentParser(description="PyTorch Time Series Baselines Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg = setup_config(args)

    eng = Engine(cfg)
    for _ in range(5):
        eng.train()
        eng.test()

if __name__ == '__main__':
    main()
