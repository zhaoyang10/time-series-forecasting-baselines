import torch
from tsf_baselines.config import cfg

def acquire_device(cfg):
    if cfg.MODEL.USE_GPU:
        device = torch.device('cuda:{}'.format(cfg.MODEL.DEVICE))
        print('Use GPU: cuda:{}'.format(cfg.MODEL.DEVICE))
    else:
        device = torch.device('cpu')
        print('Use CPU')
    return device


def setup_config(args):
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return cfg