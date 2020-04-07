from easydict import EasyDict as edict

config = edict()

# model parameters
config.MODEL = edict()
config.MODEL.NUM_CLASSES = 1000

# AnyNet parameters (REGNETX-8.0GF)
config.ANYNET = edict()
config.ANYNET.STEM_TYPE = 'resnet_stem_imagenet'
config.ANYNET.STEM_WIDTHS = 32
config.ANYNET.BLOCK_TYPE = 'residual_bottleneck_block'
config.ANYNET.DEPTHS = [2, 5, 15, 1]
config.ANYNET.WIDTHS = [80, 240, 720, 1920]
config.ANYNET.STRIDES = [2, 2, 2, 2]
config.ANYNET.BOTTLENECK_RATIOS = []
config.ANYNET.NUM_GROUPS = []

# batch normalization parameters
config.BN = edict()
config.BN.MOMENTUM = 0.1
config.BN.EPSILON = 1e-5

def _BuildConfig():
    return config
