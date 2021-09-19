import os

from tsf_baselines.utils.logger import setup_logger
from tensorboardX import SummaryWriter

class Engine_Basic(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.output_dir = self.cfg.OUTPUT_DIR
        self.tensorboard_dir = os.path.join(self.output_dir, 'tensorboard')
        if self.tensorboard_dir and not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.tensorboard_writer = SummaryWriter(self.tensorboard_dir)

        num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        log_name = '{}'.format(cfg.MODEL.NAME)
        self.logger = setup_logger(log_name, self.output_dir, 0)
        self.logger.info("Using {} GPUS".format(num_gpus))
        self.logger.info("Running with config:\n{}".format(cfg))

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
