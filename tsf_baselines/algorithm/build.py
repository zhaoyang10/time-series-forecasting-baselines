import torch

from tsf_baselines.modeling import build_network

ALGORITHMS = [
    'BasicTransformerEncoderDecoder'
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    print('algorithm_name = {}'.format(algorithm_name))
    return globals()[algorithm_name]

def build_algorithm(cfg):
    algorithm = get_algorithm_class(cfg.ALGORITHM.NAME)(cfg)
    return algorithm

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a time series forecasting algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, cfg):
        super(Algorithm, self).__init__()
        self.cfg = cfg
        self.device = self._acquire_device()

    def _acquire_device(self):
        print('self.cfg = {}'.format(self.cfg))
        if self.cfg.MODEL.USE_GPU:
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cfg.MODEL.DEVICE) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.cfg.MODEL.DEVICE))
            print('Use GPU: cuda:{}'.format(self.cfg.MODEL.DEVICE))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def update(self, minibatches):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class BasicTransformerEncDec(Algorithm):
    def __init__(self, cfg):
        super(BasicTransformerEncDec, self).__init__(cfg)
        self.cfg = cfg

        # Backbone
        self.model = build_network(cfg)

        # Loss function
        self.loss_mse = torch.nn.MSELoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.SOLVER.BASE_LR)

        # other declarations
        pass

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.cfg.DATASETS.PADDING == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.cfg.MODEL.PRED_LEN, batch_y.shape[-1]]).float()
        elif self.DATASETS.PADDING == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.cfg.MODEL.PRED_LEN, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:, :self.cfg.MODEL.LABEL_LEN, :], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        if self.cfg.MODEL.OUTPUT_ATTENTION:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        if self.cfg.DATASETS.INVERSE:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.cfg.DATASETS.FEATURES == 'MS' else 0
        batch_y = batch_y[:, -self.cfg.MODEL.PRED_LEN:, f_dim:].to(self.device)

        return outputs, batch_y

    def update(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):

        outputs, batch_y = self._process_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)

        loss = self.loss_mse(outputs, batch_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):

        outputs, batch_y = self._process_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)

        return outputs, batch_y