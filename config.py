{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww30040\viewh16180\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # File: /content/drive/MyDrive/00 AI/MultiView_SLR/FLatten-Transformer/config.py\
\
import os\
import yaml\
from yacs.config import CfgNode as CN\
\
_C = CN()\
\
# Base config files\
_C.BASE = [''] # Cho ph\'e9p k\uc0\u7871  th\u7915 a t\u7915  c\'e1c file config kh\'e1c\
\
# -----------------------------------------------------------------------------\
# Data settings\
# -----------------------------------------------------------------------------\
_C.DATA = CN()\
_C.DATA.BATCH_SIZE = 128    # Batch size PER GPU if DDP, total batch size if single GPU\
_C.DATA.DATA_PATH = ''      # S\uc0\u7869  \u273 \u432 \u7907 c ghi \u273 \'e8 b\u7903 i --data-path ho\u7863 c file YAML\
_C.DATA.DATASET = 'imagenet'# T\'ean dataset, v\'ed d\uc0\u7909 : imagenet, cifar10\
_C.DATA.IMG_SIZE = 224      # K\'edch th\uc0\u432 \u7899 c \u7843 nh \u273 \u7847 u v\'e0o\
_C.DATA.INTERPOLATION = 'bicubic' # Ph\uc0\u432 \u417 ng ph\'e1p n\u7897 i suy\
_C.DATA.ZIP_MODE = False    # S\uc0\u7917  d\u7909 ng dataset d\u7841 ng zip\
_C.DATA.CACHE_MODE = 'part' # Ch\uc0\u7871  \u273 \u7897  cache: no, full, part\
_C.DATA.PIN_MEMORY = True   # Pin memory cho DataLoader\
_C.DATA.NUM_WORKERS = 8     # S\uc0\u7889  worker cho DataLoader\
_C.DATA.CROP_PCT = 0.875    # Crop percentage cho validation (timm default)\
\
# -----------------------------------------------------------------------------\
# Model settings\
# -----------------------------------------------------------------------------\
_C.MODEL = CN()\
_C.MODEL.TYPE = 'swin'      # Lo\uc0\u7841 i model, v\'ed d\u7909 : flatten_swin, swin, pvt\
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224' # T\'ean model c\uc0\u7909  th\u7875 \
_C.MODEL.RESUME = ''        # \uc0\u272 \u432 \u7901 ng d\u7851 n \u273 \u7871 n checkpoint \u273 \u7875  resume ho\u7863 c eval\
_C.MODEL.NUM_CLASSES = 1000 # S\uc0\u7889  l\u7899 p (s\u7869  \u273 \u432 \u7907 c c\u7853 p nh\u7853 t t\u7915  build_loader)\
_C.MODEL.DROP_RATE = 0.0    # Dropout rate chung\
_C.MODEL.DROP_PATH_RATE = 0.1 # Stochastic depth rate\
_C.MODEL.LABEL_SMOOTHING = 0.1\
\
# Swin Transformer specific parameters (cho FLatten-Swin v\'e0 Swin g\uc0\u7889 c)\
_C.MODEL.SWIN = CN()\
_C.MODEL.SWIN.PATCH_SIZE = 4\
_C.MODEL.SWIN.IN_CHANS = 3\
_C.MODEL.SWIN.EMBED_DIM = 96\
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]\
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]\
_C.MODEL.SWIN.WINDOW_SIZE = 7 # M\uc0\u7863 c \u273 \u7883 nh cho Swin-T. FLatten-Swin c\'f3 th\u7875  c\'f3 logic ri\'eang.\
_C.MODEL.SWIN.MLP_RATIO = 4.0\
_C.MODEL.SWIN.QKV_BIAS = True\
_C.MODEL.SWIN.QK_SCALE = None # S\uc0\u7869  \u273 \u432 \u7907 c t\'ednh t\u7921  \u273 \u7897 ng n\u7871 u None\
_C.MODEL.SWIN.APE = False     # Absolute Position Embedding\
_C.MODEL.SWIN.PATCH_NORM = True\
# C\'e1c key sau c\'f3 th\uc0\u7875  \u273 \u7863 c th\'f9 cho c\u7845 u h\'ecnh FLatten-Transformer ho\u7863 c c\'e1c bi\u7871 n th\u7875  Swin t\'f9y ch\u7881 nh\
_C.MODEL.SWIN.KA = [7, 7, 7, 7]                 # Kernel Attention sizes?\
_C.MODEL.SWIN.DIM_REDUCTION = [4, 4, 4, 4]      # Dimensionality reduction factors?\
_C.MODEL.SWIN.STAGES = [True, True, True, True] # Which stages to build\
_C.MODEL.SWIN.STAGES_NUM = [-1, -1, -1, -1]     # Num blocks in stages if overriding DEPTHS for specific LA/SA config?\
_C.MODEL.SWIN.RPB = True                        # Relative Position Bias\
_C.MODEL.SWIN.PADDING_MODE = 'zeros'            # Padding mode for LA DWC?\
_C.MODEL.SWIN.SHARE_DWC_KERNEL = True         # For FLatten specific attention\
_C.MODEL.SWIN.SHARE_QKV = False               # For FLatten specific attention\
_C.MODEL.SWIN.LR_FACTOR = 2                   # Factor for LR in later stages?\
_C.MODEL.SWIN.DEPTHS_LR = [2, 2, 2, 2]        # Depths for LR variant?\
_C.MODEL.SWIN.FUSION_TYPE = 'add'             # Fusion type for LA and SA?\
_C.MODEL.SWIN.STAGE_CFG = None                # Cho ph\'e9p c\uc0\u7845 u h\'ecnh stage ph\u7913 c t\u7841 p (v\'ed d\u7909 , xen k\u7869  LA v\'e0 SA)\
\
\
# C\'e1c lo\uc0\u7841 i Swin kh\'e1c (new_allowed=True cho ph\'e9p th\'eam key con t\u7915  YAML n\u7871 u c\u7847 n)\
_C.MODEL.SWIN_HR = CN(new_allowed=True)\
_C.MODEL.SWIN_LRVIT = CN(new_allowed=True)\
_C.MODEL.PVD = CN(new_allowed=True) # Placeholder cho PVT-like model configs?\
\
# Linear Attention (FLatten) specific settings\
_C.MODEL.LA = CN()\
_C.MODEL.LA.FOCUSING_FACTOR = 3\
_C.MODEL.LA.KERNEL_SIZE = 5\
_C.MODEL.LA.ATTN_TYPE = 'LLLL' # V\'ed d\uc0\u7909 : 'LLSS' (2 stage \u273 \u7847 u Linear, 2 sau Standard Swin)\
_C.MODEL.LA.PVT_LA_SR_RATIOS = 1111    # Cho FLatten-PVT, l\'e0 s\uc0\u7889  nguy\'ean\
_C.MODEL.LA.CSWIN_LA_SPLIT_SIZE = '56-28-14-7' # Cho FLatten-CSwin, l\'e0 chu\uc0\u7895 i\
\
# -----------------------------------------------------------------------------\
# Training settings\
# -----------------------------------------------------------------------------\
_C.TRAIN = CN()\
_C.TRAIN.START_EPOCH = 0    # S\uc0\u7869  \u273 \u432 \u7907 c c\u7853 p nh\u7853 t t\u7915  checkpoint n\u7871 u resume\
_C.TRAIN.EPOCHS = 300       # T\uc0\u7893 ng s\u7889  epoch hu\u7845 n luy\u7879 n\
_C.TRAIN.WARMUP_EPOCHS = 20 # S\uc0\u7889  epoch warmup\
_C.TRAIN.COOLDOWN_EPOCHS = 0  # S\uc0\u7889  epoch cooldown \u7903  cu\u7889 i (th\u432 \u7901 ng l\'e0 0)\
_C.TRAIN.WEIGHT_DECAY = 0.05\
_C.TRAIN.BASE_LR = 5e-4     # LR c\uc0\u417  s\u7903  cho t\u7893 ng batch size 512\
_C.TRAIN.WARMUP_LR = 5e-7   # LR kh\uc0\u7903 i \u273 \u7847 u cho warmup\
_C.TRAIN.MIN_LR = 5e-6      # LR t\uc0\u7889 i thi\u7875 u cho cosine scheduler\
_C.TRAIN.CLIP_GRAD = 5.0    # Gi\'e1 tr\uc0\u7883  clip gradient norm, 0 ho\u7863 c None \u273 \u7875  t\u7855 t\
_C.TRAIN.AUTO_RESUME = True # T\uc0\u7921  \u273 \u7897 ng resume t\u7915  checkpoint m\u7899 i nh\u7845 t trong output dir\
_C.TRAIN.USE_CHECKPOINT = False # Gradient checkpointing (torch.utils.checkpoint)\
\
_C.TRAIN.LR_SCHEDULER = CN()\
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine' # Lo\uc0\u7841 i scheduler (cosine, step, etc.)\
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30 # Cho StepLRScheduler\
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1  # Cho StepLRScheduler\
\
_C.TRAIN.OPTIMIZER = CN()\
_C.TRAIN.OPTIMIZER.NAME = 'adamw' # Lo\uc0\u7841 i optimizer (adamw, sgd)\
_C.TRAIN.OPTIMIZER.EPS = 1e-8\
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999) # Tuple cho AdamW\
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9     # Cho SGD\
\
# -----------------------------------------------------------------------------\
# Augmentation settings\
# -----------------------------------------------------------------------------\
_C.AUG = CN()\
_C.AUG.COLOR_JITTER = 0.4\
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1' # Ch\'ednh s\'e1ch AutoAugment\
_C.AUG.REPROB = 0.25        # Random Erasing probability\
_C.AUG.REMODE = 'pixel'     # Random Erasing mode\
_C.AUG.RECOUNT = 1          # Random Erasing count\
_C.AUG.MIXUP = 0.8          # Mixup alpha\
_C.AUG.CUTMIX = 1.0         # Cutmix alpha\
_C.AUG.CUTMIX_MINMAX = None # Cutmix min/max ratio (None \uc0\u273 \u7875  d\'f9ng alpha)\
_C.AUG.MIXUP_PROB = 1.0     # X\'e1c su\uc0\u7845 t \'e1p d\u7909 ng Mixup/Cutmix\
_C.AUG.MIXUP_SWITCH_PROB = 0.5 # X\'e1c su\uc0\u7845 t chuy\u7875 n sang Cutmix (n\u7871 u c\u7843  hai \u273 \u7873 u b\u7853 t)\
_C.AUG.MIXUP_MODE = 'batch'   # C\'e1ch \'e1p d\uc0\u7909 ng Mixup/Cutmix\
\
# -----------------------------------------------------------------------------\
# Testing settings (Validation)\
# -----------------------------------------------------------------------------\
_C.TEST = CN()\
_C.TEST.CROP = True     # C\'f3 s\uc0\u7917  d\u7909 ng center crop khi test kh\'f4ng\
_C.TEST.SHUFFLE = False # Sampler cho t\uc0\u7853 p validation c\'f3 shuffle kh\'f4ng (th\u432 \u7901 ng l\'e0 False)\
\
# -----------------------------------------------------------------------------\
# Misc\
# -----------------------------------------------------------------------------\
_C.AMP = False # S\uc0\u7917  d\u7909 ng Automatic Mixed Precision kh\'f4ng\
_C.OUTPUT = '' # Th\uc0\u432  m\u7909 c g\u7889 c cho output (s\u7869  \u273 \u432 \u7907 c ghi \u273 \'e8)\
_C.TAG = 'default' # Tag cho th\uc0\u7917  nghi\u7879 m (s\u7869  \u273 \u432 \u7907 c ghi \u273 \'e8)\
_C.SAVE_FREQ = 1   # T\uc0\u7847 n su\u7845 t l\u432 u checkpoint (s\u7889  epoch)\
_C.PRINT_FREQ = 100 # T\uc0\u7847 n su\u7845 t in log (s\u7889  batch)\
_C.SEED = 0\
_C.EVAL_MODE = False # Ch\uc0\u7881  ch\u7841 y \u273 \'e1nh gi\'e1 kh\'f4ng (s\u7869  \u273 \u432 \u7907 c ghi \u273 \'e8)\
_C.THROUGHPUT_MODE = False # Ch\uc0\u7881  ch\u7841 y test throughput kh\'f4ng (s\u7869  \u273 \u432 \u7907 c ghi \u273 \'e8)\
\
# Distributed training parameters (s\uc0\u7869  \u273 \u432 \u7907 c main.py c\u7853 p nh\u7853 t)\
_C.LOCAL_RANK = 0 # S\uc0\u7869  l\'e0 device_id cho single GPU, ho\u7863 c local rank t\u7915  DDP\
_C.RANK = 0       # Global rank trong DDP\
_C.WORLD_SIZE = 1 # T\uc0\u7893 ng s\u7889  process trong DDP\
\
# _C.DISTRIBUTED = CN() # N\uc0\u7871 u c\'f3 c\'e1c config DDP c\u7909  th\u7875  kh\'e1c\
# _C.DISTRIBUTED.BROADCAST_BUFFERS = True\
\
\
def _update_config_from_file(config, cfg_file):\
    config.defrost()\
    with open(cfg_file, 'r') as f:\
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)\
\
    # Load base configs first (n\uc0\u7871 u c\'f3)\
    for base_cfg_file in yaml_cfg.setdefault('BASE', ['']): # \uc0\u272 \u7893 i t\'ean bi\u7871 n \u273 \u7875  r\'f5 r\'e0ng h\u417 n\
        if base_cfg_file:\
            _update_config_from_file(\
                config, os.path.join(os.path.dirname(cfg_file), base_cfg_file)\
            )\
    print('=> merge config from \{\}'.format(cfg_file))\
    config.merge_from_file(cfg_file)\
    # Kh\'f4ng freeze \uc0\u7903  \u273 \'e2y, s\u7869  freeze \u7903  cu\u7889 i update_config t\u7893 ng th\u7875 \
\
def update_config(config, args):\
    # 1. Load t\uc0\u7915  file config ch\'ednh (n\u7871 u \u273 \u432 \u7907 c cung c\u7845 p)\
    if args.cfg:\
        _update_config_from_file(config, args.cfg)\
\
    config.defrost() # Cho ph\'e9p thay \uc0\u273 \u7893 i config\
\
    # 2. Ghi \uc0\u273 \'e8 t\u7915  --opts (danh s\'e1ch c\'e1c key-value)\
    if args.opts:\
        print(f"=> merge options from command line opts: \{args.opts\}")\
        config.merge_from_list(args.opts)\
\
    # 3. Ghi \uc0\u273 \'e8 t\u7915  c\'e1c tham s\u7889  d\'f2ng l\u7879 nh c\u7909  th\u7875  (\u432 u ti\'ean cao nh\u7845 t)\
    if args.batch_size is not None:\
        config.DATA.BATCH_SIZE = args.batch_size\
        print(f"=> CMD_LINE: DATA.BATCH_SIZE overriden to \{config.DATA.BATCH_SIZE\}")\
    if args.data_path is not None:\
        config.DATA.DATA_PATH = args.data_path\
        print(f"=> CMD_LINE: DATA.DATA_PATH overriden to '\{config.DATA.DATA_PATH\}'")\
    if args.zip is not None and args.zip is True: # --zip l\'e0 store_true\
        config.DATA.ZIP_MODE = True\
        print(f"=> CMD_LINE: DATA.ZIP_MODE set to True")\
    if args.cache_mode is not None:\
        config.DATA.CACHE_MODE = args.cache_mode\
        print(f"=> CMD_LINE: DATA.CACHE_MODE overriden to '\{config.DATA.CACHE_MODE\}'")\
\
    if args.resume is not None: # Cho ph\'e9p truy\uc0\u7873 n --resume '' \u273 \u7875  t\u7855 t resume\
        config.MODEL.RESUME = args.resume\
        print(f"=> CMD_LINE: MODEL.RESUME set to '\{config.MODEL.RESUME\}'")\
\
    if args.use_checkpoint is not None: # N\uc0\u7871 u --use-checkpoint \u273 \u432 \u7907 c d\'f9ng, args.use_checkpoint s\u7869  l\'e0 True\
        config.TRAIN.USE_CHECKPOINT = args.use_checkpoint\
        print(f"=> CMD_LINE: TRAIN.USE_CHECKPOINT set to \{config.TRAIN.USE_CHECKPOINT\}")\
\
    if args.amp is not None: # --amp ho\uc0\u7863 c --no-amp \u273 \'e3 \u273 \u432 \u7907 c d\'f9ng\
        config.AMP = args.amp\
        print(f"=> CMD_LINE: AMP set to \{config.AMP\}")\
\
    if args.output is not None:\
        config.OUTPUT = args.output\
        # \uc0\u272 \u432 \u7901 ng d\u7851 n output \u273 \u7847 y \u273 \u7911  s\u7869  \u273 \u432 \u7907 c t\u7841 o sau khi MODEL.NAME v\'e0 TAG \u273 \u432 \u7907 c x\'e1c \u273 \u7883 nh\
    if args.tag is not None:\
        config.TAG = args.tag\
        print(f"=> CMD_LINE: TAG set to '\{config.TAG\}'")\
\
    if args.eval: # action='store_true'\
        config.EVAL_MODE = True\
    if args.throughput: # action='store_true'\
        config.THROUGHPUT_MODE = True\
\
    if hasattr(args, 'epochs') and args.epochs is not None:\
        config.TRAIN.EPOCHS = args.epochs\
        print(f"=> CMD_LINE: TRAIN.EPOCHS set to \{config.TRAIN.EPOCHS\}")\
\
    if hasattr(args, 'auto_resume') and args.auto_resume is not None:\
        # args.auto_resume s\uc0\u7869  l\'e0 True n\u7871 u --auto-resume, False n\u7871 u --no-auto-resume\
        config.TRAIN.AUTO_RESUME = args.auto_resume\
        print(f"=> CMD_LINE: TRAIN.AUTO_RESUME set to \{config.TRAIN.AUTO_RESUME\}")\
\
    # LOCAL_RANK \uc0\u273 \u432 \u7907 c parser l\u7845 y t\u7915  env var ho\u7863 c default -1.\
    # N\'f3 s\uc0\u7869  \u273 \u432 \u7907 c c\u7853 p nh\u7853 t l\u7841 i trong main() sau khi DDP init (n\u7871 u c\'f3)\
    # v\'e0 sau \uc0\u273 \'f3 config.LOCAL_RANK, config.RANK, config.WORLD_SIZE \u273 \u432 \u7907 c c\u7853 p nh\u7853 t.\
    # \uc0\u7902  \u273 \'e2y, ch\'fang ta ch\u7881  c\u7847 n \u273 \u7843 m b\u7843 o n\'f3 c\'f3 m\u7897 t gi\'e1 tr\u7883  ban \u273 \u7847 u.\
    if hasattr(args, 'local_rank'):}