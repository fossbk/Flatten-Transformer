# config.py (Version 4)
# Date: 2025-05-13
# Author: Your Name / Adapted from FLatten-Transformer
#
# Fix Notes & Changelog:
# - Ensured _C defines keys used in the provided flatten_swin_t.yaml to prevent KeyError.
# - Refined update_config to robustly handle command-line argument overrides for
#   epochs, resume, auto_resume, amp, data_path, batch_size, etc.
# - Added missing default keys to _C based on main.py usage (e.g., TRAIN.COOLDOWN_EPOCHS, TEST.SHUFFLE, RANK, WORLD_SIZE).
# - Ensured data types in _C (e.g., tuple for BETAS, None for CUTMIX_MINMAX) match yacs expectations.
# - Clarified how MODEL.RESUME and TRAIN.AUTO_RESUME interact and are updated.
# - Improved logic for constructing the final config.OUTPUT path.

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 128
_C.DATA.DATA_PATH = '' # Will be overridden by command line or specific YAML
_C.DATA.DATASET = 'imagenet'
_C.DATA.IMG_SIZE = 224
_C.DATA.INTERPOLATION = 'bicubic'
_C.DATA.ZIP_MODE = False
_C.DATA.CACHE_MODE = 'part' # choices: 'no', 'full', 'part'
_C.DATA.PIN_MEMORY = True
_C.DATA.NUM_WORKERS = 8
_C.DATA.CROP_PCT = 0.875 # Standard crop percentage for models like Swin, DeiT

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = 'swin' # Example: flatten_swin, will be overridden by YAML
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224' # Example, will be overridden by YAML
_C.MODEL.RESUME = '' # Path to checkpoint for resuming or evaluation
_C.MODEL.NUM_CLASSES = 1000 # Default for ImageNet, will be updated by build_loader
_C.MODEL.DROP_RATE = 0.0
_C.MODEL.DROP_PATH_RATE = 0.1
_C.MODEL.LABEL_SMOOTHING = 0.1

# Swin Transformer specific parameters (Used by FLatten-Swin)
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7 # Default for Swin-T. YAML can override this.
_C.MODEL.SWIN.MLP_RATIO = 4.0
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None # Python None
_C.MODEL.SWIN.APE = False # Absolute Position Embedding
_C.MODEL.SWIN.PATCH_NORM = True
# FLatten-Swin specific or other Swin variant params (ensure your model uses these if set in YAML)
_C.MODEL.SWIN.KA = [7, 7, 7, 7] # Kernel Attention sizes?
_C.MODEL.SWIN.DIM_REDUCTION = [4, 4, 4, 4] # Dimension reduction factors?
_C.MODEL.SWIN.STAGES = [True, True, True, True] # Which stages to build
_C.MODEL.SWIN.STAGES_NUM = [-1, -1, -1, -1] # Specific block indices for LA/SA?
_C.MODEL.SWIN.RPB = True # Relative Position Bias
_C.MODEL.SWIN.PADDING_MODE = 'zeros'
_C.MODEL.SWIN.SHARE_DWC_KERNEL = True
_C.MODEL.SWIN.SHARE_QKV = False
_C.MODEL.SWIN.LR_FACTOR = 2 # For PVD/LR-VIT
_C.MODEL.SWIN.DEPTHS_LR = [2, 2, 2, 2] # For PVD/LR-VIT
_C.MODEL.SWIN.FUSION_TYPE = 'add' # For PVD/LR-VIT
_C.MODEL.SWIN.STAGE_CFG = None # Python None, for custom stage configurations

# Placeholders for other Swin variants if used in more complex configs
_C.MODEL.SWIN_HR = CN(new_allowed=True)
_C.MODEL.SWIN_LRVIT = CN(new_allowed=True)
_C.MODEL.PVD = CN(new_allowed=True)

# Linear Attention (FLatten) specific settings
_C.MODEL.LA = CN()
_C.MODEL.LA.FOCUSING_FACTOR = 3
_C.MODEL.LA.KERNEL_SIZE = 5
_C.MODEL.LA.ATTN_TYPE = 'LLLL' # e.g., LLLL means all 4 stages use Linear Attention. Your YAML used 'LLSS'.
_C.MODEL.LA.PVT_LA_SR_RATIOS = 1111 # Integer, used for FLatten-PVT
_C.MODEL.LA.CSWIN_LA_SPLIT_SIZE = '56-28-14-7' # String, used for FLatten-CSwin

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0 # Will be updated by load_checkpoint if resuming
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.COOLDOWN_EPOCHS = 0
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
_C.TRAIN.CLIP_GRAD = 5.0
_C.TRAIN.AUTO_RESUME = True # Default is True
_C.TRAIN.USE_CHECKPOINT = False # Gradient checkpointing (model specific)

_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30 # For StepLR
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1  # For StepLR

_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999) # Tuple
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9     # For SGD

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
_C.AUG.COLOR_JITTER = 0.4
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
_C.AUG.REPROB = 0.25
_C.AUG.REMODE = 'pixel'
_C.AUG.RECOUNT = 1
_C.AUG.MIXUP = 0.8
_C.AUG.CUTMIX = 1.0
_C.AUG.CUTMIX_MINMAX = None # Python None
_C.AUG.MIXUP_PROB = 1.0
_C.AUG.MIXUP_SWITCH_PROB = 0.5
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.CROP = True
_C.TEST.SHUFFLE = False # For validation/test dataloader sampler

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.AMP = False # Automatic Mixed Precision, can be overridden by --amp or --no-amp
_C.OUTPUT = 'output' # Default base output dir
_C.TAG = 'default'   # Default experiment tag
_C.SAVE_FREQ = 1     # Save checkpoint every N epochs
_C.PRINT_FREQ = 100  # Print log every N iterations
_C.SEED = 0
_C.EVAL_MODE = False # Evaluation only mode
_C.THROUGHPUT_MODE = False # Throughput test only mode

# Distributed training params (will be updated by main.py)
_C.LOCAL_RANK = 0
_C.RANK = 0
_C.WORLD_SIZE = 1

# Placeholder for DDP specific configs if any model uses them
# _C.DISTRIBUTED = CN()
# _C.DISTRIBUTED.BROADCAST_BUFFERS = True


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg_base_file in yaml_cfg.setdefault('BASE', ['']):
        if cfg_base_file:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg_base_file)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    # config.freeze() # Freeze at the very end of update_config

def update_config(config, args):
    # Load from YAML file first
    if args.cfg:
        _update_config_from_file(config, args.cfg)

    config.defrost() # Allow modifications

    # Then, merge from command line --opts
    if args.opts:
        print(f"=> merge options from command line (--opts): {args.opts}")
        config.merge_from_list(args.opts)

    # Then, override with specific command line arguments (highest priority)
    if args.batch_size is not None:
        config.DATA.BATCH_SIZE = args.batch_size
        print(f"=> CMD_LINE: DATA.BATCH_SIZE overriden to {args.batch_size}")
    if args.data_path is not None:
        config.DATA.DATA_PATH = args.data_path
        print(f"=> CMD_LINE: DATA.DATA_PATH overriden to '{args.data_path}'")
    if args.zip: # This is an action, will be True if present
        config.DATA.ZIP_MODE = True
        print(f"=> CMD_LINE: DATA.ZIP_MODE set to True")
    if args.cache_mode is not None:
        config.DATA.CACHE_MODE = args.cache_mode
        print(f"=> CMD_LINE: DATA.CACHE_MODE overriden to '{args.cache_mode}'")

    if args.resume is not None: # Allows --resume '' to explicitly disable resume
        config.MODEL.RESUME = args.resume
        print(f"=> CMD_LINE: MODEL.RESUME overriden to '{args.resume}'")

    if args.use_checkpoint is not None: # default=None in parser, action='store_true'
        config.TRAIN.USE_CHECKPOINT = args.use_checkpoint # Will be True if --use-checkpoint is present
        print(f"=> CMD_LINE: TRAIN.USE_CHECKPOINT set to {args.use_checkpoint}")

    if args.amp is not None: # Handles --amp or --no-amp
        config.AMP = args.amp
        print(f"=> CMD_LINE: AMP set to {config.AMP}")

    if args.output is not None:
        config.OUTPUT = args.output
        # Full output path is constructed later after MODEL.NAME and TAG are finalized
    if args.tag is not None:
        config.TAG = args.tag
        print(f"=> CMD_LINE: TAG overriden to '{args.tag}'")

    if args.eval: # action='store_true'
        config.EVAL_MODE = True
        print(f"=> CMD_LINE: EVAL_MODE set to True")
    if args.throughput: # action='store_true'
        config.THROUGHPUT_MODE = True
        print(f"=> CMD_LINE: THROUGHPUT_MODE set to True")

    if hasattr(args, 'epochs') and args.epochs is not None:
        config.TRAIN.EPOCHS = args.epochs
        print(f"=> CMD_LINE: TRAIN.EPOCHS overriden to {args.epochs}")

    if hasattr(args, 'auto_resume') and args.auto_resume is not None: # Handles --auto-resume or --no-auto-resume
        config.TRAIN.AUTO_RESUME = args.auto_resume
        print(f"=> CMD_LINE: TRAIN.AUTO_RESUME set to {args.auto_resume}")
    
    # LOCAL_RANK is primarily handled by main.py after DDP setup.
    # However, if explicitly passed via args (e.g., by launch utility), store it.
    if hasattr(args, 'local_rank') and args.local_rank != -1:
        config.LOCAL_RANK = args.local_rank
        # print(f"=> CMD_LINE: Initial config.LOCAL_RANK set to {args.local_rank} (from args)")


    # Construct final output folder path
    # Ensure MODEL.NAME and TAG have values (from YAML, _C defaults, or args)
    model_name_for_path = config.MODEL.NAME if config.MODEL.NAME else "unknown_model"
    tag_for_path = config.TAG if config.TAG else "default" # Changed from default_tag
    
    # If config.OUTPUT was set by args.output, use that as base. Otherwise use _C.OUTPUT.
    base_output_dir = config.OUTPUT if (args.output is not None) else _C.OUTPUT

    config.OUTPUT = os.path.join(base_output_dir, model_name_for_path, tag_for_path)
    print(f"=> Effective OUTPUT directory will be: {config.OUTPUT}")

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with defaults, merged with file, and then with args."""
    config = _C.clone()
    update_config(config, args) # update_config handles merging from file and other args
    return config
