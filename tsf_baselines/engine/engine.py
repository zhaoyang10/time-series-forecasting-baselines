import os
import time

import numpy as np
import torch

from tsf_baselines.engine.engine_basic import Engine_Basic
from tsf_baselines.utils.metrics import metric
from tsf_baselines.algorithm import build_algorithm
from tsf_baselines.data import get_data
from tsf_baselines.utils.tools import adjust_learning_rate
import warnings

warnings.filterwarnings('ignore')


class Engine(Engine_Basic):
    def __init__(self, cfg, reps='default'):
        super(Engine, self).__init__(cfg, reps)
        self.algorithm = build_algorithm(self.cfg)
        self.algorithm.to(self.algorithm.device)

        self.train_data, self.train_loader = get_data(self.cfg, flag='train')
        self.valid_data, self.valid_loader = get_data(self.cfg, flag='val')
        self.test_data, self.test_loader = get_data(self.cfg, flag='test')

    def train(self):
        iterations = 0
        for epoch in range(self.cfg.SOLVER.MAX_EPOCHS):
            self.algorithm.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):

                iterations += 1
                train_loss_dict = self.algorithm.update(self.train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)

                if iterations % self.cfg.SOLVER.LOG_PERIOD  == 0:
                    self.logger.info('train loss dict : {}'.format(train_loss_dict))
                    for key, value in train_loss_dict.items():
                        self.tensorboard_writer.add_scalar(key, value, iterations)

                if iterations % self.cfg.SOLVER.EVAL_PERIOD == 0:
                    valid_loss_dict = self.validate()
                    self.logger.info('valid loss dict : {}'.format(valid_loss_dict))
                    for key, value in valid_loss_dict.items():
                        self.tensorboard_writer.add_scalar(key, value, iterations)

            adjust_learning_rate(self.algorithm.optimizer, epoch + 1, self.cfg)

            if (epoch + 1) % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                test_loss_dict = self.test()
                self.logger.info('test loss dict : {}'.format(test_loss_dict))
                for key, value in test_loss_dict.items():
                    self.tensorboard_writer.add_scalar(key, value, iterations)

                # save checkpoint
                self.logger.info('Saving checkpoint')
                checkpoint_file = os.path.join(self.output_dir, 'checkpoint_epoch_{}.pth'.format(epoch + 1))
                torch.save(self.algorithm.state_dict(), checkpoint_file)
                pass
        return

    def validate(self):
        self.algorithm.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.valid_loader):
            pred, batch_y = self.algorithm.predict(self.valid_data, batch_x, batch_y, batch_x_mark, batch_y_mark)

            loss = self.algorithm.loss_mse(pred.detach().cpu(), batch_y.cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.algorithm.train()
        return {'loss_valid': total_loss}

    def test(self):
        self.algorithm.eval()
        total_loss = []
        preds = []
        trues = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.test_loader):
            pred, batch_y = self.algorithm.predict(self.valid_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())
            loss = self.algorithm.loss_mse(pred.detach().cpu(), batch_y.cpu())
            total_loss.append(loss)
        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        total_loss = np.average(total_loss)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        self.logger.info('mse:{}, mae:{}'.format(mse, mae))

        np.save(os.path.join(self.output_dir, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(self.output_dir, 'pred.npy'), preds)
        np.save(os.path.join(self.output_dir, 'true.npy'), trues)

        self.algorithm.train()
        return {'loss_test': total_loss,
                'mse_test':  mse,
                'mae_test': mae,
                'rmse_test': rmse,
                'mape_test': mape,
                'mspe_test': mspe}

