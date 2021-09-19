# encoding: utf-8

from torch.utils.data import DataLoader

from tsf_baselines.data.datasets.ett import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred

def get_data(cfg, flag):

    data_dict = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute,
        'WTH': Dataset_Custom,
        'ECL': Dataset_Custom,
        'Solar': Dataset_Custom,
        'custom': Dataset_Custom,
    }
    Data = data_dict[cfg.DATASETS.NAME]
    timeenc = 0 if cfg.MODEL.EMBED != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = cfg.SOLVER.BATCH_SIZE
        freq = cfg.MODEL.FREQ
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = cfg.MODEL.FREQ
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = cfg.SOLVER.BATCH_SIZE
        freq = cfg.MODEL.FREQ
    data_set = Data(
        root_path=cfg.ROOT_PATH,
        data_path=cfg.DATASETS.DATA_PATH,
        flag=flag,
        # size=[args.seq_len, args.label_len, args.pred_len],
        size = [cfg.MODEL.SEQ_LEN, cfg.MODEL.LABEL_LEN, cfg.MODEL.PRED_LEN],
        features=cfg.DATASETS.FEATURES,
        target=cfg.DATASETS.TARGET,
        inverse=cfg.DATASETS.INVERSE,
        timeenc=timeenc,
        freq=freq,
        cols=cfg.DATASETS.COLUMNS
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=drop_last)

    return data_set, data_loader