{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww30040\viewh16180\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # File: /content/drive/MyDrive/00 AI/MultiView_SLR/FLatten-Transformer/main.py\
\
import os\
import time\
import argparse\
import datetime\
import numpy as np\
\
import torch\
import torch.nn as nn\
import torch.backends.cudnn as cudnn\
import torch.distributed as dist\
from torch.cuda.amp import autocast, GradScaler\
\
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy\
from timm.utils import accuracy, AverageMeter\
\
from config import get_config\
from models import build_model\
from data import build_loader # build_loader c\uc0\u7847 n nh\u7853 n is_distributed\
from lr_scheduler import build_scheduler\
from optimizer import build_optimizer\
from logger import create_logger\
from utils import load_checkpoint, save_checkpoint_new, get_grad_norm, auto_resume_helper, reduce_tensor, load_pretrained\
\
import warnings\
# warnings.filterwarnings('ignore') # C\'f3 th\uc0\u7875  b\u7853 t l\u7841 i n\u7871 u mu\u7889 n xem c\'e1c warning c\u7911 a timm\
\
def parse_option():\
    parser = argparse.ArgumentParser('FLatten Transformer training and evaluation script', add_help=False)\
    parser.add_argument('--cfg', type=str, metavar="FILE", default=None,\
                        help='Path to config file (optional, values can be overridden by command line args)')\
    parser.add_argument(\
        "--opts",\
        help="Modify config options by adding 'KEY VALUE' pairs. Example: TRAIN.EPOCHS 10 DATA.BATCH_SIZE 32",\
        default=None,\
        nargs='+',\
    )\
\
    # Basic command line overrides for convenience\
    parser.add_argument('--batch-size', type=int, default=None, help="Batch size per GPU (overrides config value)")\
    parser.add_argument('--data-path', type=str, default=None, help='Path to dataset (overrides config value)')\
    parser.add_argument('--epochs', type=int, default=None, help="Total number of training epochs (overrides config value)")\
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint path (can be empty string for no resume, overrides config value)')\
    parser.add_argument('--output', default=None, type=str, metavar='PATH',\
                        help='Root of output folder (base path, model_name and tag will be appended, overrides config value)')\
    parser.add_argument('--tag', default=None, type=str, help='Tag of experiment (used in output path, overrides config value)')\
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained weights (not a full training checkpoint, e.g., for fine-tuning backbone).')\
\
    # Boolean flags\
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only (sets EVAL_MODE=True)')\
    parser.add_argument('--throughput', action='store_true', help='Test throughput only (sets THROUGHPUT_MODE=True)')\
    \
    # default=None cho ph\'e9p config.py quy\uc0\u7871 t \u273 \u7883 nh gi\'e1 tr\u7883  m\u7863 c \u273 \u7883 nh t\u7915  _C n\u7871 u kh\'f4ng c\'f3 c\u7901  n\'e0o \u273 \u432 \u7907 c d\'f9ng\
    zip_group = parser.add_mutually_exclusive_group()\
    zip_group.add_argument('--zip', action='store_true', dest='zip', default=None, help='Use zipped dataset (set DATA.ZIP_MODE=True)')\
    zip_group.add_argument('--no-zip', action='store_false', dest='zip', help='Do not use zipped dataset (set DATA.ZIP_MODE=False)')\
\
    use_ckpt_group = parser.add_mutually_exclusive_group()\
    use_ckpt_group.add_argument('--use-checkpoint', action='store_true', dest='use_checkpoint', default=None, help="Enable gradient checkpointing")\
    use_ckpt_group.add_argument('--no-use-checkpoint', action='store_false', dest='use_checkpoint', help="Disable gradient checkpointing")\
\
    amp_group = parser.add_mutually_exclusive_group()\
    amp_group.add_argument('--amp', action='store_true', dest='amp', default=None, help="Enable Automatic Mixed Precision")\
    amp_group.add_argument('--no-amp', action='store_false', dest='amp', help="Disable Automatic Mixed Precision")\
\
    auto_resume_group = parser.add_mutually_exclusive_group()\
    auto_resume_group.add_argument('--auto-resume', action='store_true', dest='auto_resume', default=None)\
    auto_resume_group.add_argument('--no-auto-resume', action='store_false', dest='auto_resume')\
\
    # Other specific args\
    parser.add_argument('--cache-mode', type=str, default=None, choices=['no', 'full', 'part'], help='Dataset cache mode (overrides config value)')\
    parser.add_argument('--find-unused-params', action='store_true', default=False, help='For DDP, to find unused parameters (debug).')\
\
    # Distributed training parameters (torch.distributed.launch/torchrun will set environment variables)\
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', -1)),\
                        help='local rank for distributed training. Default: -1 or from LOCAL_RANK env var.')\
\
    args = parser.parse_args()\
    config = get_config(args) # get_config calls update_config which itself uses args\
\
    return args, config\
\
def adjusted_reduce_tensor(tensor, world_size):\
    """ Reduces tensor across all processes if world_size > 1 and dist is initialized. """\
    if world_size > 1 and dist.is_initialized():\
        rt = tensor.clone() # Clone before all_reduce to avoid modifying the original tensor if it's needed elsewhere\
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)\
        rt /= world_size\
        return rt\
    return tensor\
\
def main():\
    args, config = parse_option() # config is now fully resolved here\
\
    # --- DISTRIBUTED/DEVICE SETUP ---\
    is_distributed = False\
    # Check if launched with a distributed runner (torchrun/torch.distributed.launch)\
    # These utilities set WORLD_SIZE and LOCAL_RANK (or RANK) environment variables.\
    if 'WORLD_SIZE' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1:\
        is_distributed = True\
        # LOCAL_RANK from args should have been populated from os.environ by argparse\
        if args.local_rank == -1: # Should not happen if WORLD_SIZE > 1 and launch utility is used\
            print("Warning: WORLD_SIZE > 1 but LOCAL_RANK is -1. Attempting to get LOCAL_RANK from env again.")\
            args.local_rank = int(os.environ.get('LOCAL_RANK', 0)) # Default to 0 if still not found\
    \
    # If user explicitly passes --local_rank >= 0 but WORLD_SIZE is not set (e.g. manual DDP setup)\
    # This is an advanced case and usually means user knows what they are doing.\
    # For simplicity, we rely on 'WORLD_SIZE' env var for DDP detection.\
\
    current_device_local_rank = args.local_rank # This is the rank on the current node\
\
    if is_distributed:\
        torch.cuda.set_device(current_device_local_rank)\
        try:\
            # init_method='env://' relies on MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE env vars\
            # These are set by torch.distributed.launch or torchrun.\
            dist.init_process_group(backend='nccl', init_method='env://')\
            global_rank = dist.get_rank()\
            world_size_actual = dist.get_world_size()\
            torch.distributed.barrier() # Synchronize all processes\
            print(f"DDP Initialized: GlobalRank=\{global_rank\}, LocalRankOnNode=\{current_device_local_rank\}, WorldSize=\{world_size_actual\}")\
        except Exception as e:\
            print(f"Failed to initialize distributed group with 'env://': \{e\}.")\
            print("Ensure MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE are set, or try another init_method.")\
            print("Falling back to non-distributed mode.")\
            is_distributed = False # Fallback\
            global_rank = 0\
            world_size_actual = 1\
            current_device_local_rank = 0 if torch.cuda.is_available() else -1\
    else:\
        global_rank = 0\
        world_size_actual = 1\
        current_device_local_rank = 0 if torch.cuda.is_available() else -1\
        if current_device_local_rank != -1:\
            torch.cuda.set_device(current_device_local_rank) # Set to GPU 0 if available\
\
    # Update config with the true DDP status and ranks\
    config.defrost()\
    config.LOCAL_RANK = current_device_local_rank # This is the device_id for the current process\
    config.RANK = global_rank\
    config.WORLD_SIZE = world_size_actual\
    config.freeze()\
\
    if config.LOCAL_RANK != -1: # On GPU\
        print(f"Process (Global Rank \{config.RANK\}) running on GPU: cuda:\{config.LOCAL_RANK\}")\
        if is_distributed and dist.is_initialized() and dist.get_backend() == 'nccl':\
            os.environ["NCCL_BLOCKING_WAIT"] = "1"\
    else: # On CPU\
        print(f"Process (Global Rank \{config.RANK\}) running on CPU")\
\
\
    seed = config.SEED + config.RANK # Use global rank for seed diversity\
    torch.manual_seed(seed)\
    np.random.seed(seed)\
    cudnn.enabled = True\
    cudnn.benchmark = True\
\
    # Scale learning rate\
    if config.WORLD_SIZE > 0: # Should always be true (at least 1)\
        # config.DATA.BATCH_SIZE is batch_size_per_gpu\
        total_batch_size = config.DATA.BATCH_SIZE * config.WORLD_SIZE\
        base_lr_from_config = config.TRAIN.BASE_LR # LR before scaling\
        warmup_lr_from_config = config.TRAIN.WARMUP_LR\
        min_lr_from_config = config.TRAIN.MIN_LR\
\
        config.defrost()\
        # Scale if the base LR in config was meant for a reference total batch size (e.g., 512)\
        config.TRAIN.BASE_LR = base_lr_from_config * total_batch_size / 512.0\
        config.TRAIN.WARMUP_LR = warmup_lr_from_config * total_batch_size / 512.0\
        config.TRAIN.MIN_LR = min_lr_from_config * total_batch_size / 512.0\
        config.freeze()\
    \
    os.makedirs(config.OUTPUT, exist_ok=True)\
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=config.RANK, name=f"\{config.MODEL.NAME\}")\
\
    if config.RANK == 0: # Only main process saves config\
        path = os.path.join(config.OUTPUT, "config_effective.json") # Save effective config\
        with open(path, "w") as f:\
            f.write(config.dump())\
        logger.info(f"Effective config saved to \{path\}")\
\
    logger.info(f"FULL CONFIGURATION:\\n\{config.dump()\}")\
    if is_distributed: logger.info(f"Distributed training active with \{config.WORLD_SIZE\} processes.")\
    logger.info(f"Batch size per process: \{config.DATA.BATCH_SIZE\}")\
    logger.info(f"Total effective batch size: \{config.DATA.BATCH_SIZE * config.WORLD_SIZE\}")\
    logger.info(f"Training for \{config.TRAIN.EPOCHS\} epochs (from START_EPOCH: \{config.TRAIN.START_EPOCH\})")\
    logger.info(f"Effective Base LR: \{config.TRAIN.BASE_LR:.2e\}, Warmup LR: \{config.TRAIN.WARMUP_LR:.2e\}, Min LR: \{config.TRAIN.MIN_LR:.2e\}")\
    logger.info(f"AMP: \{config.AMP\}, Auto Resume: \{config.TRAIN.AUTO_RESUME\}, Explicit Resume: '\{config.MODEL.RESUME\}'")\
\
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config, is_distributed)\
\
    logger.info(f"Creating model: \{config.MODEL.TYPE\}/\{config.MODEL.NAME\}")\
    model = build_model(config) # build_model might use config.MODEL.NUM_CLASSES\
    if config.LOCAL_RANK != -1: model.cuda(config.LOCAL_RANK)\
    # logger.info(str(model)) # Can be very long\
\
    optimizer = build_optimizer(config, model)\
\
    if is_distributed:\
        broadcast_buffers_cfg = getattr(getattr(config, 'DISTRIBUTED', CN()), 'BROADCAST_BUFFERS', True)\
        model = nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK],\
                                                    broadcast_buffers=broadcast_buffers_cfg,\
                                                    find_unused_parameters=args.find_unused_params)\
    model_without_ddp = model.module if is_distributed else model\
    \
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)\
    logger.info(f"Number of learnable parameters: \{n_parameters/1e6:.2f\} M")\
\
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))\
    # config.TRAIN.EPOCHS is the total epochs to run FOR THIS INVOCATION if not resuming,\
    # or the target total epochs if resuming.\
    # config.TRAIN.START_EPOCH is where we start from (0 or from checkpoint).\
    # total_epochs_for_loop should be the target total, e.g., 300.\
    # The loop will be for epoch in range(config.TRAIN.START_EPOCH, total_epochs_for_loop)\
    # Let's use config.TRAIN.EPOCHS as the target total number of epochs for the whole training process\
    target_total_epochs = config.TRAIN.EPOCHS + config.TRAIN.COOLDOWN_EPOCHS\
\
\
    if config.AUG.MIXUP > 0. or config.AUG.CUTMIX > 0.:\
        criterion = SoftTargetCrossEntropy()\
    elif config.MODEL.LABEL_SMOOTHING > 0.:\
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)\
    else:\
        criterion = nn.CrossEntropyLoss()\
\
    max_accuracy = 0.0\
\
    if hasattr(args, 'pretrained') and args.pretrained:\
        logger.info(f"Loading pretrained backbone weights (NOT a full checkpoint) from: \{args.pretrained\}")\
        # load_pretrained function might need config for num_classes if it adapts the head\
        load_pretrained(args.pretrained, model_without_ddp, logger, config=config)\
\
    # Resume logic:\
    # 1. `args.resume` (from cmd line) overrides `config.MODEL.RESUME` (from YAML). This is done in `update_config`.\
    # 2. If `config.TRAIN.AUTO_RESUME` is True AND `config.MODEL.RESUME` is empty, then try `auto_resume_helper`.\
    \
    actual_resume_path = config.MODEL.RESUME # This path is now the one from cmd line or YAML\
    if config.TRAIN.AUTO_RESUME and (not actual_resume_path or actual_resume_path == ''):\
        resume_file_auto = auto_resume_helper(config.OUTPUT) # Searches in config.OUTPUT\
        if resume_file_auto:\
            logger.info(f'AUTO_RESUME: Found checkpoint in output directory: \{resume_file_auto\}. Using it.')\
            actual_resume_path = resume_file_auto\
            # Update config.MODEL.RESUME so load_checkpoint uses it\
            config.defrost()\
            config.MODEL.RESUME = actual_resume_path\
            config.freeze()\
        else:\
            logger.info(f'AUTO_RESUME: No checkpoint found in \{config.OUTPUT\}. Will use explicit resume path if provided, or train from scratch.')\
    \
    if actual_resume_path and actual_resume_path != '':\
        logger.info(f"Attempting to load full training checkpoint from: \{actual_resume_path\}")\
        # load_checkpoint should update config.TRAIN.START_EPOCH\
        loaded_max_acc, loaded_start_epoch = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)\
        max_accuracy = max(max_accuracy, loaded_max_acc) # Use max_accuracy from checkpoint\
        # config.TRAIN.START_EPOCH is updated inside load_checkpoint\
        logger.info(f"Resumed. Training will continue from epoch \{config.TRAIN.START_EPOCH +1 \}.")\
        \
        # Optionally validate after resuming (if not in EVAL_MODE already)\
        if not config.EVAL_MODE:\
            logger.info("Validating model after resuming checkpoint...")\
            acc1_resumed, _, _ = validate(config, data_loader_val, model, logger, config.WORLD_SIZE)\
            logger.info(f"Accuracy of resumed model on val set: \{acc1_resumed:.2f\}%")\
            max_accuracy = max(max_accuracy, acc1_resumed)\
        \
        if config.LOCAL_RANK != -1 : torch.cuda.empty_cache()\
    else:\
        logger.info("No checkpoint specified or found. Training from scratch (epoch 0).")\
        config.defrost()\
        config.TRAIN.START_EPOCH = 0 # Ensure start epoch is 0 if no resume\
        config.freeze()\
\
    if config.EVAL_MODE:\
        if not config.MODEL.RESUME or config.MODEL.RESUME == '': # Re-check after auto-resume logic\
            logger.error("EVAL_MODE is True, but no checkpoint specified or found. Cannot evaluate.")\
            return\
        logger.info(f"--- Starting Evaluation on Validation Set (from checkpoint: \{config.MODEL.RESUME\}) ---")\
        acc1_eval, acc5_eval, loss_eval = validate(config, data_loader_val, model, logger, config.WORLD_SIZE)\
        logger.info(f"Evaluation Results: Acc@1 \{acc1_eval:.3f\}, Acc@5 \{acc5_eval:.3f\}, Loss \{loss_eval:.4f\}")\
        return\
\
    if config.THROUGHPUT_MODE:\
        logger.info("--- Starting Throughput Test ---")\
        throughput(data_loader_val, model, logger, config)\
        return\
\
    # target_total_epochs is the final epoch number (e.g., 300)\
    # The loop will run for (target_total_epochs - config.TRAIN.START_EPOCH) epochs.\
    logger.info(f"--- Starting Training from epoch \{config.TRAIN.START_EPOCH + 1\} up to total epoch \{target_total_epochs\} ---")\
    start_time = time.time()\
\
    for epoch in range(config.TRAIN.START_EPOCH, target_total_epochs):\
        current_epoch_display = epoch + 1 # For logging (1-based)\
        if is_distributed and hasattr(data_loader_train.sampler, 'set_epoch'):\
            data_loader_train.sampler.set_epoch(epoch) # Sampler needs 0-based epoch\
\
        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, logger, target_total_epochs)\
        \
        if config.LOCAL_RANK != -1 : torch.cuda.empty_cache()\
\
        if (current_epoch_display % config.SAVE_FREQ == 0) or (current_epoch_display == target_total_epochs):\
            acc1_val_epoch, _, _ = validate(config, data_loader_val, model, logger, config.WORLD_SIZE)\
            logger.info(f"Validation after epoch \{current_epoch_display\}: Acc@1 \{acc1_val_epoch:.2f\}%")\
\
            is_best = acc1_val_epoch > max_accuracy\
            max_accuracy = max(max_accuracy, acc1_val_epoch)\
            logger.info(f'Max accuracy so far: \{max_accuracy:.2f\}%')\
\
            if config.RANK == 0: # Only main process saves checkpoints\
                save_checkpoint_new(config, current_epoch_display, model_without_ddp, acc1_val_epoch, optimizer, lr_scheduler, logger, name=f'ckpt_epoch_\{current_epoch_display\}')\
                if is_best:\
                    save_checkpoint_new(config, current_epoch_display, model_without_ddp, acc1_val_epoch, optimizer, lr_scheduler, logger, name='max_acc')\
        \
        # Step LR scheduler (some step per epoch, some per batch/update)\
        if lr_scheduler is not None:\
            # If scheduler has step_epoch attribute and it's True, or if it's ReduceLROnPlateau\
            if hasattr(lr_scheduler, 'step_epoch') and lr_scheduler.step_epoch:\
                lr_scheduler.step(current_epoch_display)\
            elif isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):\
                if 'acc1_val_epoch' in locals(): # Check if validation was run this epoch\
                    lr_scheduler.step(acc1_val_epoch)\
                else: # Fallback if no validation metric (e.g. SAVE_FREQ > 1)\
                    # For ReduceLROnPlateau, this might not be ideal without a metric\
                    pass\
            # For schedulers like CosineAnnealingWarmRestarts that step per batch,\
            # they are handled by lr_scheduler.step_update() in train_one_epoch.\
            # For standard CosineAnnealingLR or StepLR, they usually step per epoch.\
            elif not hasattr(lr_scheduler, 'step_update'): # If not a per-batch_update scheduler\
                lr_scheduler.step() # Most epoch-wise schedulers take no args or epoch number\
\
    total_time_spent = time.time() - start_time\
    total_time_str = str(datetime.timedelta(seconds=int(total_time_spent)))\
    logger.info('Total training run time for this session: \{\}'.format(total_time_str))\
    logger.info(f'Final Max accuracy achieved: \{max_accuracy:.2f\}%')\
\
\
# --- C\'e1c h\'e0m train_one_epoch, validate, throughput (Gi\uc0\u7919  nguy\'ean ho\u7863 c \u273 \u7843 m b\u7843 o ch\'fang s\u7917  d\u7909 ng `config` \u273 \'fang) ---\
# (D\'e1n l\uc0\u7841 i c\'e1c h\'e0m \u273 \'e3 s\u7917 a t\u7915  c\'e2u tr\u7843  l\u7901 i tr\u432 \u7899 c, \u273 \u7843 m b\u7843 o ch\'fang nh\u7853 n `config` v\'e0 d\'f9ng `config.LOCAL_RANK`, `config.WORLD_SIZE` etc.)\
\
def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, logger, target_total_epochs):\
    model.train()\
    num_steps = len(data_loader)\
    batch_time = AverageMeter()\
    loss_meter = AverageMeter()\
    norm_meter = AverageMeter()\
    start = time.time()\
    end = time.time()\
\
    use_amp_scaler = config.AMP and torch.cuda.is_available() and config.LOCAL_RANK != -1\
    scaler = GradScaler() if use_amp_scaler else None\
\
    for idx, (samples, targets) in enumerate(data_loader):\
        optimizer.zero_grad()\
        \
        device_to_use = torch.device(config.LOCAL_RANK) if config.LOCAL_RANK != -1 else torch.device('cpu')\
        samples = samples.to(device_to_use, non_blocking=True if config.LOCAL_RANK != -1 else False)\
        targets = targets.to(device_to_use, non_blocking=True if config.LOCAL_RANK != -1 else False)\
\
\
        if mixup_fn is not None:\
            samples, targets = mixup_fn(samples, targets)\
\
        grad_norm_val = None # Kh\uc0\u7903 i t\u7841 o \u273 \u7875  tr\'e1nh l\u7895 i n\u7871 u kh\'f4ng \u273 \u432 \u7907 c g\'e1n\
        if use_amp_scaler:\
            with autocast():\
                outputs = model(samples)\
                loss = criterion(outputs, targets)\
            scaler.scale(loss).backward()\
            if config.TRAIN.CLIP_GRAD and config.TRAIN.CLIP_GRAD > 0:\
                scaler.unscale_(optimizer) \
                grad_norm_val = nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)\
            scaler.step(optimizer)\
            scaler.update()\
        else:\
            outputs = model(samples)\
            loss = criterion(outputs, targets)\
            loss.backward()\
            if config.TRAIN.CLIP_GRAD and config.TRAIN.CLIP_GRAD > 0:\
                grad_norm_val = nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)\
            else:\
                try: # get_grad_norm c\'f3 th\uc0\u7875  l\u7895 i n\u7871 u kh\'f4ng c\'f3 grad\
                    grad_norm_val = get_grad_norm(model.parameters())\
                except Exception:\
                    grad_norm_val = torch.tensor(0.0) # M\uc0\u7863 c \u273 \u7883 nh n\u7871 u l\u7895 i\
            optimizer.step()\
        \
        # Step per-batch schedulers\
        if lr_scheduler is not None and hasattr(lr_scheduler, 'step_update') and callable(lr_scheduler.step_update):\
            lr_scheduler.step_update(epoch * num_steps + idx)\
\
\
        if config.LOCAL_RANK != -1: torch.cuda.synchronize()\
\
        loss_meter.update(loss.item(), targets.size(0))\
        if grad_norm_val is not None: norm_meter.update(grad_norm_val.item() if torch.is_tensor(grad_norm_val) else grad_norm_val)\
        batch_time.update(time.time() - end)\
        end = time.time()\
\
        if (idx + 1) % config.PRINT_FREQ == 0 or idx == num_steps - 1:\
            lr = optimizer.param_groups[0]['lr']\
            mem_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if config.LOCAL_RANK != -1 and torch.cuda.is_available() else 0\
            etas = batch_time.avg * (num_steps - idx - 1)\
            current_norm_display = norm_meter.val if norm_meter.count > 0 else (grad_norm_val.item() if grad_norm_val is not None and torch.is_tensor(grad_norm_val) else (grad_norm_val if grad_norm_val is not None else 0.0) )\
            avg_norm_display = norm_meter.avg if norm_meter.count > 0 else 0.0\
            logger.info(\
                f'Train: [\{epoch + 1\}/\{target_total_epochs\}][\{idx + 1\}/\{num_steps\}]\\t'\
                f'eta \{datetime.timedelta(seconds=int(etas))\} lr \{lr:.6f\}\\t'\
                f'time \{batch_time.val:.4f\} (\{batch_time.avg:.4f\})\\t'\
                f'loss \{loss_meter.val:.4f\} (\{loss_meter.avg:.4f\})\\t'\
                f'grad_norm \{current_norm_display:.4f\} (\{avg_norm_display:.4f\})\\t'\
                f'mem \{mem_used:.0f\}MB')\
    epoch_time = time.time() - start\
    logger.info(f"EPOCH \{epoch + 1\} training takes \{datetime.timedelta(seconds=int(epoch_time))\}")\
\
\
@torch.no_grad()\
def validate(config, data_loader, model, logger, world_size):\
    criterion = nn.CrossEntropyLoss()\
    model.eval() # Quan tr\uc0\u7885 ng: \u273 \u7863 t model \u7903  ch\u7871  \u273 \u7897  eval\
    batch_time = AverageMeter()\
    loss_meter = AverageMeter()\
    acc1_meter = AverageMeter()\
    acc5_meter = AverageMeter()\
    end = time.time()\
\
    for idx, (images, target) in enumerate(data_loader):\
        device_to_use = torch.device(config.LOCAL_RANK) if config.LOCAL_RANK != -1 else torch.device('cpu')\
        images = images.to(device_to_use, non_blocking=True if config.LOCAL_RANK != -1 else False)\
        target = target.to(device_to_use, non_blocking=True if config.LOCAL_RANK != -1 else False)\
        \
        output = model(images)\
        loss = criterion(output, target)\
        acc1_val, acc5_val = accuracy(output, target, topk=(1, 5))\
\
        loss = adjusted_reduce_tensor(loss, world_size)\
        acc1 = adjusted_reduce_tensor(acc1_val.clone(), world_size)\
        acc5 = adjusted_reduce_tensor(acc5_val.clone(), world_size)\
\
        loss_meter.update(loss.item(), target.size(0))\
        acc1_meter.update(acc1.item(), target.size(0))\
        acc5_meter.update(acc5.item(), target.size(0))\
        batch_time.update(time.time() - end)\
        end = time.time()\
\
        if (idx + 1) % config.PRINT_FREQ == 0 or idx == len(data_loader) - 1: # In \uc0\u7903  iter cu\u7889 i\
            mem_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if config.LOCAL_RANK != -1 and torch.cuda.is_available() else 0\
            logger.info(\
                f'Test: [\{(idx + 1)\}/\{len(data_loader)\}]\\t'\
                f'Time \{batch_time.val:.3f\} (\{batch_time.avg:.3f\})\\t'\
                f'Loss \{loss_meter.val:.4f\} (\{loss_meter.avg:.4f\})\\t'\
                f'Acc@1 \{acc1_meter.val:.3f\} (\{acc1_meter.avg:.3f\})\\t'\
                f'Acc@5 \{acc5_meter.val:.3f\} (\{acc5_meter.avg:.3f\})\\t'\
                f'Mem \{mem_used:.0f\}MB')\
    logger.info(f' * Acc@1 \{acc1_meter.avg:.3f\} Acc@5 \{acc5_meter.avg:.3f\}')\
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg\
\
\
@torch.no_grad()\
def throughput(data_loader, model, logger, config):\
    model.eval()\
    use_cuda = torch.cuda.is_available() and config.LOCAL_RANK != -1\
    device_to_use_throughput = torch.device(config.LOCAL_RANK) if use_cuda else torch.device('cpu')\
\
    try:\
        images, _ = next(iter(data_loader))\
    except StopIteration:\
        logger.error("Data loader for throughput test is empty.")\
        return\
\
    images = images.to(device_to_use_throughput, non_blocking=use_cuda)\
    batch_size = images.shape[0]\
\
    logger.info(f"Throughput test: Warming up (50 iterations) on device \{device_to_use_throughput\}...")\
    for _ in range(50): model(images)\
    if use_cuda: torch.cuda.synchronize()\
    \
    logger.info(f"Throughput test: Measuring with 30 iterations for batch_size \{batch_size\}")\
    # \uc0\u272 o l\u432 \u7901 ng ch\'ednh x\'e1c h\u417 n\
    repetitions = 30\
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)\
    timings = np.zeros((repetitions, 1))\
\
    # WARMUP GPU\
    for _ in range(10): _ = model(images)\
\
    # MEASURE PERFORMANCE\
    with torch.no_grad():\
        for rep in range(repetitions):\
            starter.record()\
            _ = model(images)\
            ender.record()\
            # WAIT FOR GPU SYNC\
            torch.cuda.synchronize()\
            curr_time = starter.elapsed_time(ender) # milliseconds\
            timings[rep] = curr_time\
            \
    mean_synctime = np.sum(timings) / repetitions\
    std_synctime = np.std(timings)\
    throughput_val = (repetitions * batch_size) / (mean_synctime / 1000.0) # images/sec (chuy\uc0\u7875 n ms sang s)\
    \
    logger.info(f"Batch_size \{batch_size\} -> Mean sync time: \{mean_synctime:.2f\} ms, Std: \{std_synctime:.2f\} ms")\
    logger.info(f"Batch_size \{batch_size\} -> Throughput: \{throughput_val:.2f\} images/sec")\
    return\
\
\
if __name__ == '__main__':\
    # Kh\'f4ng c\uc0\u7847 n \u273 \u7863 t os.environ th\u7911  c\'f4ng \u7903  \u273 \'e2y n\u7871 u d\'f9ng torch.distributed.launch\
    # ho\uc0\u7863 c n\u7871 u logic trong main() x\u7917  l\'fd \u273 \'fang cho single GPU.\
    main()}