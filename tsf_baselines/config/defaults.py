from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()


# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.NAME = 'Informer'
_C.MODEL.USE_GPU = True
_C.MODEL.DEVICE = "0"
# _C.MODEL.NUM_CLASSES = 10
_C.MODEL.ENC_IN = 7
_C.MODEL.DEC_IN = 7
_C.MODEL.C_OUT = 1
_C.MODEL.SEQ_LEN = 512
_C.MODEL.LABEL_LEN = 48
_C.MODEL.PRED_LEN = 24
_C.MODEL.FACTOR = 5
_C.MODEL.D_MODEL = 512
_C.MODEL.N_HEADS = 8
_C.MODEL.E_LAYERS = 2
_C.MODEL.D_LAYERS = 1
_C.MODEL.D_FF = 2048
_C.MODEL.DROP_OUT = 0.05
_C.MODEL.ATTN = 'prob' # ['prob', 'full']
_C.MODEL.EMBED = 'timeF' # ['timeF', 'fixed', 'learned']
_C.MODEL.FREQ = 'h' # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
_C.MODEL.ACTIVATION = 'gelu'
_C.MODEL.OUTPUT_ATTENTION = False
_C.MODEL.DISTIL = True # whether to use distilling in encoder, using this argument means not using distilling
_C.MODEL.MIX = True # use mix attention in generative decoder

# -----------------------------------------------------------------------------
# ALGORITHM
# -----------------------------------------------------------------------------
_C.ALGORITHM = CN()
_C.ALGORITHM.NAME = 'BasicTransformerEncDec'

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = 32
# Size of the image during test
_C.INPUT.SIZE_TEST = 32
# Minimum scale for the image during training
_C.INPUT.MIN_SCALE_TRAIN = 0.5
# Maximum scale for the image during test
_C.INPUT.MAX_SCALE_TRAIN = 1.2
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.1307, ]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.3081, ]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()

_C.DATASETS.NAME = 'ETTh1'
_C.DATASETS.DATA_PATH = 'datasets/ETT/ETTh1.csv'
_C.DATASETS.FEATURES = 'S'
_C.DATASETS.TARGET = 'OT'
_C.DATASETS.INVERSE = False
_C.DATASETS.COLUMNS = None
_C.DATASETS.PADDING = 0
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8



# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "ADAM"

_C.SOLVER.MAX_EPOCHS = 5

_C.SOLVER.BASE_LR = 1e-4
_C.SOLVER.LR_ADJUST = 'type1'

# _C.SOLVER.BASE_LR = 1e-4

_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.LOG_PERIOD = 100
_C.SOLVER.EVAL_PERIOD = 1000
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
# _C.SOLVER.IMS_PER_BATCH = 16
_C.SOLVER.BATCH_SIZE = 32

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
# _C.TEST.IMS_PER_BATCH = 8
_C.TEST.BATCH_SIZE = 32
_C.TEST.WEIGHT = ""

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "output/informer_v1"
_C.ROOT_PATH = "."
_C.REPEAT = 5
