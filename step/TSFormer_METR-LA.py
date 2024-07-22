import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from basicts.losses import masked_mae

from .step_arch import TSFormer
from .step_runner import TSFormerRunner
from .step_data import PretrainingDataset


CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "TSFormer(METR-LA) configuration"
CFG.RUNNER = TSFormerRunner          # Runner encapsulating the training, validation, and testing loops.
CFG.DATASET_CLS = PretrainingDataset # Pytorch's Dataset subclass
CFG.DATASET_NAME = "METR-LA"
CFG.DATASET_TYPE = "Traffic speed"
CFG.DATASET_INPUT_LEN = 288 * 7      # Length of the historical time-windows (2016 samples, frequency is 5 mins, so this equates to 1 week length)
CFG.DATASET_OUTPUT_LEN = 12          # Length of the windows to predict, i.e., 5 min * 12 = 1 hour
CFG.GPU_NUM = 1

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 0
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "TSFormer"
CFG.MODEL.ARCH = TSFormer           # Class of the model to train
CFG.MODEL.PARAM = {                 # Parameters to be passed to the model's class constructor
    "patch_size":12,
    "in_channel":1,
    "embed_dim":96,
    "num_heads":4,
    "mlp_ratio":4,
    "dropout":0.1,
    "num_token":288 * 7 / 12,
    "mask_ratio":0.75,
    "encoder_depth":4,
    "decoder_depth":1,
    "mode":"pre-train"
}
CFG.MODEL.FORWARD_FEATURES = [0] # It seems that the model is using just 1 feature in the input data...? TODO: to be checked.
CFG.MODEL.TARGET_FEATURES = [0] # It seems that the model is considering just 1 feature when generating output data...? TODO: to be checked.

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae    # Loss used during training is masked MAE
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":0.0005,
    "weight_decay":0,
    "eps":1.0e-8,
    "betas":(0.9, 0.95)
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM= {
    "milestones":[50],
    "gamma":0.5
}

# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0
}
CFG.TRAIN.NUM_EPOCHS = 100
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.NULL_VAL = 0.0
# read data
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE = 4
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 12
CFG.TRAIN.DATA.PIN_MEMORY = True

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 4
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 12
CFG.VAL.DATA.PIN_MEMORY = True

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# evluation
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 4
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 12
CFG.TEST.DATA.PIN_MEMORY = True
