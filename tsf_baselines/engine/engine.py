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
    def __init__(self, cfg):
        super(Engine, self).__init__(cfg)
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
                'mape': mape,
                'mspe': mspe}

    def predict_orig(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

    def vali_orig(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train_orig(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test_orig(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict_orig(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

    def _process_one_batch_orig(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y
