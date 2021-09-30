# encoding: utf-8

import argparse
import sys
import os
import json

sys.path.append('.')
from tsf_baselines.utils.config_tools import setup_config
from tsf_baselines.engine.engine import Engine
from tsf_baselines.utils.misc import average_dict

def main():
    parser = argparse.ArgumentParser(description="PyTorch Time Series Baselines Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg = setup_config(args)

    test_list = []
    for rep in range(cfg.REPEAT):
        eng = Engine(cfg, rep+1)
        eng.train()
        test_result = eng.test()
        test_list.append(test_result)

        eng.shutdown_logger()

    test_average = average_dict(test_list)
    test_avg_file = os.path.join(cfg.OUTPUT_DIR, 'results.json')
    with open(test_avg_file, "w") as dump_f:
        json.dump(test_average, dump_f)

if __name__ == '__main__':
    main()
