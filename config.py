# config.py (Version 3)
# Ghi chú fix lỗi:
# - Đảm bảo các key trong _C khớp với file YAML mẫu (flatten_swin_t.yaml) để tránh KeyError.
# - Hoàn thiện update_config để xử lý ghi đè từ args (bao gồm epochs, resume, auto_resume, amp).
# - Thêm các key mặc định còn thiếu trong _C dựa trên file YAML và main.py.
# - Đảm bảo kiểu dữ liệu mặc định trong _C là đúng (ví dụ: tuple cho BETAS, None cho CUTMIX_MINMAX).

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
_C.DATA.DATA_PATH = '' # Sẽ được ghi đè
_C.DATA.DATASET = 'imagenet'
_C.DATA.IMG_SIZE = 224
_C.DATA.INTERPOLATION = 'bicubic'
_C.DATA.ZIP_MODE = False
_C.DATA.CACHE_MODE = 'part'
_C.DATA.PIN_MEMORY = True
_C.DATA.NUM_WORKERS = 8
_C.DATA.CROP_PCT = 0.875 # Thêm key này cho timm's create_transform

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = 'swin' # Ví dụ: flatten_swin
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224' # Ví dụ: flatten_swin_tiny_patch4_224
_C.MODEL.RESUME = '' # Đường dẫn đến checkpoint để resume hoặc eval
_C.MODEL.NUM_CLASSES = 1000 # Sẽ được cập nhật bởi build_loader
_C.MODEL.DROP_RATE = 0.0
_C.MODEL.DROP_PATH_RATE = 0.1
_C.MODEL.LABEL_SMOOTHING = 0.1

# Swin Transformer parameters (cho FLatten-Swin)
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7     # Mặc định cho Swin-T. YAML của bạn có thể là 56 cho các stage đầu của FLatten.
                                  # Code model cần xử lý nếu WINDOW_SIZE khác nhau cho các stage.
                                  # Hoặc bạn cần một key như WINDOW_SIZES (list) nếu model hỗ trợ.
                                  # Giữ 7 ở đây vì đây là config.py, YAML sẽ ghi đè.
_C.MODEL.SWIN.MLP_RATIO = 4.0
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None # None trong Python
_C.MODEL.SWIN.APE = False # Absolute Position Embedding
_C.MODEL.SWIN.PATCH_NORM = True
# Các key đặc thù cho FLatten-Swin (đảm bảo model của bạn sử dụng chúng)
_C.MODEL.SWIN.KA = [7, 7, 7, 7]
_C.MODEL.SWIN.DIM_REDUCTION = [4, 4, 4, 4]
_C.MODEL.SWIN.STAGES = [True, True, True, True]
_C.MODEL.SWIN.STAGES_NUM = [-1, -1, -1, -1]
_C.MODEL.SWIN.RPB = True
_C.MODEL.SWIN.PADDING_MODE = 'zeros'
_C.MODEL.SWIN.SHARE_DWC_KERNEL = True
_C.MODEL.SWIN.SHARE_QKV = False
_C.MODEL.SWIN.LR_FACTOR = 2
_C.MODEL.SWIN.DEPTHS_LR = [2, 2, 2, 2]
_C.MODEL.SWIN.FUSION_TYPE = 'add'
_C.MODEL.SWIN.STAGE_CFG = None # None trong Python

# Các loại Swin khác
_C.MODEL.SWIN_HR = CN(new_allowed=True)
_C.MODEL.SWIN_LRVIT = CN(new_allowed=True)
_C.MODEL.PVD = CN(new_allowed=True)

# Linear Attention (FLatten) settings
_C.MODEL.LA = CN()
_C.MODEL.LA.FOCUSING_FACTOR = 3
_C.MODEL.LA.KERNEL_SIZE = 5
_C.MODEL.LA.ATTN_TYPE = 'LLLL' # Ví dụ: 4 tầng LA. YAML của bạn là 'LLSS'
_C.MODEL.LA.PVT_LA_SR_RATIOS = 1111 # Số nguyên
_C.MODEL.LA.CSWIN_LA_SPLIT_SIZE = '56-28-14-7' # Chuỗi

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.COOLDOWN_EPOCHS = 0
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
_C.TRAIN.CLIP_GRAD = 5.0
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.USE_CHECKPOINT = False

_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999) # Tuple
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9 # Cho SGD

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
_C.AUG.CUTMIX_MINMAX = None # None trong Python
_C.AUG.MIXUP_PROB = 1.0
_C.AUG.MIXUP_SWITCH_PROB = 0.5
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.CROP = True
_C.TEST.SHUFFLE = False # Cho validation sampler

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.AMP = False # Automatic Mixed Precision
_C.OUTPUT = 'output' # Mặc định nếu không được ghi đè
_C.TAG = 'default' # Mặc định nếu không được ghi đè
_C.SAVE_FREQ = 1
_C.PRINT_FREQ = 100
_C.SEED = 0
_C.EVAL_MODE = False
_C.THROUGHPUT_MODE = False
_C.LOCAL_RANK = 0 # Sẽ được main.py cập nhật
_C.RANK = 0       # Global rank, sẽ được main.py cập nhật
_C.WORLD_SIZE = 1 # Sẽ được main.py cập nhật

# Thêm DISTRIBUTED block nếu DDP có dùng config từ đây
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
    # Không freeze ở đây, freeze ở cuối update_config

def update_config(config, args):
    if args.cfg:
        _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        print(f"=> merge options from command line: {args.opts}")
        config.merge_from_list(args.opts)

    # Ghi đè từ các tham số dòng lệnh cụ thể (có độ ưu tiên cao nhất)
    if args.batch_size is not None:
        config.DATA.BATCH_SIZE = args.batch_size
        print(f"=> CMD_LINE: DATA.BATCH_SIZE overriden to {args.batch_size}")
    if args.data_path is not None:
        config.DATA.DATA_PATH = args.data_path
        print(f"=> CMD_LINE: DATA.DATA_PATH overriden to '{args.data_path}'")
    if args.zip: # args.zip là True nếu cờ --zip được dùng
        config.DATA.ZIP_MODE = True
        print(f"=> CMD_LINE: DATA.ZIP_MODE set to True")
    if args.cache_mode is not None:
        config.DATA.CACHE_MODE = args.cache_mode
        print(f"=> CMD_LINE: DATA.CACHE_MODE overriden to '{args.cache_mode}'")

    if args.resume is not None: # Cho phép --resume '' để tắt resume
        config.MODEL.RESUME = args.resume
        print(f"=> CMD_LINE: MODEL.RESUME overriden to '{args.resume}'")

    if args.use_checkpoint is not None : # Nếu dùng action='store_true', default=False
        config.TRAIN.USE_CHECKPOINT = args.use_checkpoint # args.use_checkpoint sẽ là True/False
        print(f"=> CMD_LINE: TRAIN.USE_CHECKPOINT set to {args.use_checkpoint}")

    if args.amp is not None: # Xử lý từ --amp hoặc --no-amp
        config.AMP = args.amp
        print(f"=> CMD_LINE: AMP set to {config.AMP}")

    if args.output is not None:
        config.OUTPUT = args.output
        # Đường dẫn output đầy đủ sẽ được tạo sau, sau khi MODEL.NAME và TAG được xác định
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

    if hasattr(args, 'auto_resume') and args.auto_resume is not None:
        config.TRAIN.AUTO_RESUME = args.auto_resume
        print(f"=> CMD_LINE: TRAIN.AUTO_RESUME set to {args.auto_resume}")

    # LOCAL_RANK sẽ được main.py xử lý và cập nhật vào config sau khi DDP init (nếu có)
    # Tuy nhiên, nếu args.local_rank được truyền từ launch utility, ta có thể cập nhật sớm
    if hasattr(args, 'local_rank') and args.local_rank != -1: # -1 là giá trị mặc định nếu không phải DDP
        config.LOCAL_RANK = args.local_rank
        # print(f"=> CMD_LINE: config.LOCAL_RANK set to {args.local_rank} (from args)")

    # Tạo đường dẫn output đầy đủ
    # Đảm bảo MODEL.NAME và TAG có giá trị (từ YAML hoặc default của _C hoặc args)
    model_name_for_path = config.MODEL.NAME if config.MODEL.NAME else "unknown_model"
    tag_for_path = config.TAG if config.TAG else "default_tag"
    base_output_path = config.OUTPUT if config.OUTPUT else "output" # Lấy từ args hoặc default của _C

    config.OUTPUT = os.path.join(base_output_path, model_name_for_path, tag_for_path)
    print(f"=> Effective OUTPUT directory set to: {config.OUTPUT}")

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    # update_config sẽ được gọi trong main.py sau khi args được parse hoàn toàn
    # Ở đây chỉ trả về config mặc định, main.py sẽ gọi update_config(config, args)
    # Hoặc, nếu bạn muốn get_config trả về config đã update:
    update_config(config, args) # Gọi update_config ở đây
    return config