# main.py (Version 5)
# Date: 2025-05-13
# Author: Your Name / Adapted from FLatten-Transformer
#
# Fix Notes & Changelog:
# - Solved TypeError: '>' not supported between instances of 'float' and 'tuple'
#   by ensuring acc1_val_epoch (or similar) used for max_accuracy comparison is a float.
# - Refined parse_option for --epochs, --resume, --auto-resume, --amp with appropriate defaults (None)
#   to allow config.py to distinguish between unset and explicitly set by cmd line.
# - Robust DDP/Device setup: Prioritizes env vars from launch utilities, falls back gracefully.
#   Updates config.RANK, config.WORLD_SIZE, config.LOCAL_RANK.
# - LR Scaling uses config.WORLD_SIZE and config.DATA.BATCH_SIZE.
# - Comprehensive logging of effective config values.
# - Refined resume logic:
#   - MODEL.RESUME from cmd line/YAML is checked first.
#   - If MODEL.RESUME is empty, then TRAIN.AUTO_RESUME is considered.
#   - load_checkpoint is called with 5 arguments.
#   - config.TRAIN.START_EPOCH is expected to be updated by load_checkpoint.
# - Training loop correctly iterates from config.TRAIN.START_EPOCH.
# - Validation called conditionally and results unpacked correctly.
# - LR scheduler step logic clarified for epoch-based vs batch-based.
# - train_one_epoch, validate, throughput now take 'config' as primary source for device/DDP info.

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader # build_loader should take (config, is_distributed)
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
# Ensure utils.py defines reduce_tensor and other imported functions
from utils import load_checkpoint, save_checkpoint_new, get_grad_norm, auto_resume_helper, reduce_tensor, load_pretrained

import warnings
warnings.filterwarnings('ignore')

def parse_option():
    parser = argparse.ArgumentParser('FLatten Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE", default=None,
                        help='Path to config file (optional, values can be set via opts or defaults)')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # Config overrides from command line
    parser.add_argument('--batch-size', type=int, default=None, help="Batch size per GPU (overrides config)")
    parser.add_argument('--data-path', type=str, default=None, help='Path to dataset (overrides config)')
    parser.add_argument('--zip', action='store_true', help='Use zipped dataset (sets DATA.ZIP_MODE=True)') # No default, presence means True
    parser.add_argument('--cache-mode', type=str, default=None, choices=['no', 'full', 'part'],
                        help='Dataset cache mode (overrides config)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint (overrides config, use "" for no resume)')
    
    # For store_true, default should be False if not present, or None to let config decide
    # If default=None, and neither --use-checkpoint nor an override in YAML/opts is present, it will take _C's default.
    # If --use-checkpoint is present, args.use_checkpoint becomes True.
    parser.add_argument('--use-checkpoint', action='store_true', default=None) # Default None, if flag present then True
    
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument('--amp', action='store_true', dest='amp', default=None) # if present, args.amp = True
    amp_group.add_argument('--no-amp', action='store_false', dest='amp') # if present, args.amp = False

    parser.add_argument('--output', default=None, type=str, metavar='PATH', help='Root of output folder (overrides config)')
    parser.add_argument('--tag', default=None, type=str, help='Tag of experiment (overrides config)')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--pretrained', type=str, default=None, help='Load pretrained weights (not a full checkpoint).')
    parser.add_argument('--find-unused-params', action='store_true', default=False) # For DDP

    parser.add_argument('--epochs', type=int, default=None, help="Total number of training epochs (overrides config)")

    auto_resume_group = parser.add_mutually_exclusive_group()
    auto_resume_group.add_argument('--auto-resume', action='store_true', dest='auto_resume', default=None)
    auto_resume_group.add_argument('--no-auto-resume', action='store_false', dest='auto_resume')

    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', -1)),
                        help='Local rank for distributed training. Automatically set by launch utility.')

    args = parser.parse_args()
    config = get_config(args) # get_config calls update_config internally

    return args, config

def adjusted_reduce_tensor(tensor, world_size): # Wrapper for utils.reduce_tensor
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        return reduce_tensor(tensor) # Calls the imported reduce_tensor from utils
    return tensor

def main():
    args, config = parse_option()

    # --- DISTRIBUTED/DEVICE SETUP ---
    is_distributed = False
    # Check if launched with torch.distributed.launch or torchrun
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1 :
        is_distributed = True
        # args.local_rank should have been set by the launch utility (or defaulted to env var in parse_option)
        if args.local_rank == -1 : # Should not happen if launched correctly
            args.local_rank = int(os.environ.get("LOCAL_RANK",0)) # Fallback, should be an error though
            print(f"Warning: Distributed mode detected (WORLD_SIZE > 1) but local_rank was -1. Set to env LOCAL_RANK or 0.")
    elif args.local_rank != -1 and torch.cuda.is_available() and torch.cuda.device_count() > 1 :
        # This case is for manually setting up DDP without launch utility, which is complex.
        # It's better to rely on launch utilities. Assume single GPU if not launched.
        print(f"Warning: local_rank={args.local_rank} provided without standard DDP launch. Assuming single GPU or user has set DDP env vars manually.")
        # If user really wants DDP this way, they must set WORLD_SIZE etc. env vars.
        # For simplicity here, we'll treat it as non-distributed if WORLD_SIZE env var isn't > 1.
        is_distributed = False # Revert to non-distributed if WORLD_SIZE isn't proving it.
        args.local_rank = 0 # Default to GPU 0 for non-DDP multi-GPU scenarios (rare for this script)

    current_gpu_or_device_str = 'cpu'
    final_rank = 0
    final_world_size = 1
    final_local_rank = -1 # -1 for CPU, 0 for first/single GPU

    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        try:
            dist.init_process_group(backend='nccl', init_method='env://')
            final_rank = dist.get_rank()
            final_world_size = dist.get_world_size()
            final_local_rank = args.local_rank # This is the device_id for this process
            torch.distributed.barrier()
            print(f"DDP Initialized: GlobalRank={final_rank}, LocalRank={final_local_rank}, WorldSize={final_world_size} on GPU cuda:{final_local_rank}")
            current_gpu_or_device_str = f"cuda:{final_local_rank}"
        except Exception as e:
            print(f"Failed to initialize DDP group: {e}. Running non-distributed.")
            is_distributed = False # Fallback
            final_rank = 0
            final_world_size = 1
            final_local_rank = 0 if torch.cuda.is_available() else -1
    
    if not is_distributed: # Handle non-DDP or fallback
        if torch.cuda.is_available():
            final_local_rank = 0 # Use first GPU
            torch.cuda.set_device(final_local_rank)
            current_gpu_or_device_str = f"cuda:{final_local_rank}"
            print(f"Running on single GPU: {current_gpu_or_device_str}")
        else:
            final_local_rank = -1
            current_gpu_or_device_str = "cpu"
            print("Running on CPU")

    config.defrost()
    config.LOCAL_RANK = final_local_rank
    config.RANK = final_rank
    config.WORLD_SIZE = final_world_size
    config.freeze()

    if is_distributed and torch.cuda.is_available() and dist.get_backend() == 'nccl':
        os.environ["NCCL_BLOCKING_WAIT"] = "1"

    seed = config.SEED + config.RANK # Use global rank for seed consistency
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.enabled = True
    cudnn.benchmark = True

    if config.WORLD_SIZE > 0:
        eff_batch_size = config.DATA.BATCH_SIZE * config.WORLD_SIZE
        base_lr = config.TRAIN.BASE_LR
        warmup_lr = config.TRAIN.WARMUP_LR
        min_lr = config.TRAIN.MIN_LR
        linear_scaled_lr = base_lr * eff_batch_size / 512.0
        linear_scaled_warmup_lr = warmup_lr * eff_batch_size / 512.0
        linear_scaled_min_lr = min_lr * eff_batch_size / 512.0
        config.defrost()
        config.TRAIN.BASE_LR = linear_scaled_lr
        config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
        config.TRAIN.MIN_LR = linear_scaled_min_lr
        config.freeze()
    
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=config.RANK, name=f"{config.MODEL.NAME}")

    if config.RANK == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f: f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    logger.info(f"Effective config:\n{config.dump()}")
    if is_distributed: logger.info(f"Distributed training enabled with {config.WORLD_SIZE} processes.")
    logger.info(f"Running on device: {current_gpu_or_device_str}")
    logger.info(f"Batch size per process/GPU: {config.DATA.BATCH_SIZE}")
    logger.info(f"Total effective batch size: {config.DATA.BATCH_SIZE * config.WORLD_SIZE}")
    logger.info(f"Training for {config.TRAIN.EPOCHS} total epochs.") # Total epochs from config
    logger.info(f"Base LR (after scaling): {config.TRAIN.BASE_LR:.2e}")
    logger.info(f"Warmup LR (after scaling): {config.TRAIN.WARMUP_LR:.2e}, Min LR (after scaling): {config.TRAIN.MIN_LR:.2e}")
    logger.info(f"Warmup epochs: {config.TRAIN.WARMUP_EPOCHS}")
    logger.info(f"AMP enabled: {config.AMP}")

    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config, is_distributed)

    logger.info(f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    if config.LOCAL_RANK != -1: model.cuda(config.LOCAL_RANK)
    # logger.info(str(model))

    optimizer = build_optimizer(config, model)

    if is_distributed:
        broadcast_buffers_cfg = getattr(getattr(config, 'DISTRIBUTED', CN()), 'BROADCAST_BUFFERS', True)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK],
                                                    broadcast_buffers=broadcast_buffers_cfg,
                                                    find_unused_parameters=args.find_unused_params)
    model_without_ddp = model.module if is_distributed else model
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of learnable parameters: {n_parameters/1e6:.2f} M")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    # config.TRAIN.EPOCHS is the total number of epochs to run (e.g., 30)
    # config.TRAIN.START_EPOCH is 0 initially, or updated by load_checkpoint
    total_epochs_config = config.TRAIN.EPOCHS + config.TRAIN.COOLDOWN_EPOCHS

    if config.AUG.MIXUP > 0. or config.AUG.CUTMIX > 0.: criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.: criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else: criterion = nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if hasattr(args, 'pretrained') and args.pretrained and args.pretrained != '':
        logger.info(f"Loading pretrained weights (from --pretrained cmd arg) from: {args.pretrained}")
        load_pretrained(args.pretrained, model_without_ddp, logger)

    # Resume logic:
    # 1. Use args.resume if provided (highest priority, already updated in config.MODEL.RESUME by update_config)
    # 2. If args.resume was not provided (so config.MODEL.RESUME might be from YAML or default ''),
    #    and TRAIN.AUTO_RESUME is True, try auto_resume_helper.
    
    actual_resume_path_final = config.MODEL.RESUME # Value from YAML or overridden by --resume '' or a path
    if not actual_resume_path_final and config.TRAIN.AUTO_RESUME:
        # config.MODEL.RESUME is empty AND auto_resume is True
        logger.info(f"MODEL.RESUME is empty and TRAIN.AUTO_RESUME is True. Attempting auto-resume from {config.OUTPUT}")
        resume_file_auto = auto_resume_helper(config.OUTPUT)
        if resume_file_auto:
            logger.info(f"AUTO_RESUME: Found checkpoint '{resume_file_auto}'.")
            actual_resume_path_final = resume_file_auto
            # Update config so load_checkpoint uses this path
            config.defrost()
            config.MODEL.RESUME = actual_resume_path_final
            config.freeze()
        else:
            logger.info(f"AUTO_RESUME: No checkpoint found in {config.OUTPUT}.")
    
    if actual_resume_path_final and actual_resume_path_final != '':
        logger.info(f"Attempting to load full checkpoint from: {actual_resume_path_final}")
        # load_checkpoint is expected to update config.TRAIN.START_EPOCH internally
        max_accuracy_from_ckpt = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        max_accuracy = max(max_accuracy, max_accuracy_from_ckpt)
        
        if not config.EVAL_MODE:
            logger.info("Validating resumed model...")
            acc1_resumed, acc5_resumed, loss_resumed = validate(config, data_loader_val, model, logger, config.WORLD_SIZE)
            max_accuracy = max(max_accuracy, acc1_resumed)
            logger.info(f"Resumed model validation: Acc@1 {acc1_resumed:.2f}%, Acc@5 {acc5_resumed:.2f}%, Loss {loss_resumed:.4f}")
        if config.LOCAL_RANK != -1: torch.cuda.empty_cache()
    else:
        logger.info("No checkpoint to resume from. Training from scratch or using --pretrained weights (if provided).")


    if config.EVAL_MODE:
        if not config.MODEL.RESUME or config.MODEL.RESUME == '':
            logger.error("EVAL_MODE is True, but no checkpoint specified or found. Cannot evaluate.")
            return
        logger.info(f"--- Starting Evaluation on Validation Set (from checkpoint: {config.MODEL.RESUME}) ---")
        acc1_eval, acc5_eval, loss_eval = validate(config, data_loader_val, model, logger, config.WORLD_SIZE)
        logger.info(f"Evaluation Results: Acc@1 {acc1_eval:.3f} Acc@5 {acc5_eval:.3f} Loss {loss_eval:.4f}")
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1_eval:.1f}%")
        return

    if config.THROUGHPUT_MODE:
        logger.info("--- Starting Throughput Test ---")
        throughput(data_loader_val, model, logger, config)
        return

    # config.TRAIN.START_EPOCH should be 0 if training from scratch,
    # or (epoch_in_ckpt) if resuming (load_checkpoint updates it to completed_epoch)
    # The loop will run for (total_epochs_config - config.TRAIN.START_EPOCH) iterations.
    # Example: epochs=30, resume from ckpt_epoch_1 (which saved epoch=1 as completed)
    # START_EPOCH becomes 1. Loop runs for epoch in range(1, 30) -> 29 epochs (epoch 2 to 30). Correct.
    logger.info(f"--- Starting Training from (completed) epoch {config.TRAIN.START_EPOCH} up to (total configured) epoch {total_epochs_config} ---")
    logger.info(f"   (Will run for {total_epochs_config - config.TRAIN.START_EPOCH} epochs in this session, starting with epoch {config.TRAIN.START_EPOCH +1})")


    start_time_training_session = time.time()
    # Vòng lặp for epoch in range(N, M) sẽ chạy các giá trị N, N+1, ..., M-1
    # Nếu START_EPOCH là epoch đã hoàn thành (ví dụ 0 nếu từ đầu, 1 nếu resume từ ckpt_epoch_1)
    # thì epoch đầu tiên cần chạy là START_EPOCH.
    # Log sẽ in ra epoch+1.
    for epoch in range(config.TRAIN.START_EPOCH, total_epochs_config):
        if is_distributed and hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch) # Sampler DDP cần epoch hiện tại

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, logger, total_epochs_config)
        
        if config.LOCAL_RANK != -1: torch.cuda.empty_cache()

        current_acc1_val = -1.0
        if (epoch + 1) % config.SAVE_FREQ == 0 or (epoch + 1) == total_epochs_config:
            acc1_val_epoch, acc5_val_epoch, loss_val_epoch = validate(config, data_loader_val, model, logger, config.WORLD_SIZE)
            current_acc1_val = acc1_val_epoch # Đây là float
            logger.info(f"Validation after epoch {epoch+1}: Acc@1 {current_acc1_val:.2f}%, Acc@5 {acc5_val_epoch:.2f}%, Loss {loss_val_epoch:.4f}")

            is_best = current_acc1_val > max_accuracy
            max_accuracy = max(max_accuracy, current_acc1_val) # So sánh float với float
            logger.info(f'Max accuracy so far: {max_accuracy:.2f}%')

            if config.RANK == 0: # Chỉ rank 0 lưu checkpoint
                # Lưu checkpoint cho epoch hiện tại (epoch + 1 vì vòng lặp bắt đầu từ 0 hoặc START_EPOCH)
                save_checkpoint_new(config, epoch + 1, model_without_ddp, current_acc1_val, optimizer, lr_scheduler, logger, name=f'ckpt_epoch_{epoch+1}')
                if is_best:
                    save_checkpoint_new(config, epoch + 1, model_without_ddp, current_acc1_val, optimizer, lr_scheduler, logger, name='max_acc')
        
        # Step LR scheduler
        if lr_scheduler is not None:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if current_acc1_val != -1.0: # Cần metric từ validate
                    lr_scheduler.step(current_acc1_val)
                # Nếu không validate ở epoch này, không step ReduceLROnPlateau
            elif not hasattr(lr_scheduler, 'step_update'): # Nếu là scheduler của PyTorch (không phải của timm)
                lr_scheduler.step() # Step sau mỗi epoch
            # Scheduler của timm (step_update) đã được gọi trong train_one_epoch

    total_time_session = time.time() - start_time_training_session
    total_time_str_session = str(datetime.timedelta(seconds=int(total_time_session)))
    logger.info(f'This training session time {total_time_str_session}')

# (train_one_epoch, validate, throughput giữ nguyên như Version 4 đã sửa)
def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, logger, total_epochs):
    model.train()
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    start_time_epoch_loop = time.time() # Đổi tên biến
    end_batch_time_loop = time.time() # Đổi tên biến

    use_amp_scaler = config.AMP and torch.cuda.is_available() and config.LOCAL_RANK != -1
    scaler = GradScaler() if use_amp_scaler else None

    for idx, (samples, targets) in enumerate(data_loader):
        optimizer.zero_grad()
        
        device_to_use = torch.device(config.LOCAL_RANK if config.LOCAL_RANK !=-1 else 'cpu')
        samples = samples.to(device_to_use, non_blocking=True if device_to_use != 'cpu' else False)
        targets = targets.to(device_to_use, non_blocking=True if device_to_use != 'cpu' else False)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        grad_norm_val = None 
        if use_amp_scaler and scaler is not None:
            with autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            if config.TRAIN.CLIP_GRAD and config.TRAIN.CLIP_GRAD > 0:
                scaler.unscale_(optimizer) 
                grad_norm_tensor = nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                grad_norm_val = grad_norm_tensor.item() if torch.is_tensor(grad_norm_tensor) else grad_norm_tensor
            scaler.step(optimizer)
            scaler.update()
            if grad_norm_val is None and not (config.TRAIN.CLIP_GRAD and config.TRAIN.CLIP_GRAD > 0):
                grad_norm_val = get_grad_norm(model.module.parameters() if isinstance(model, nn.parallel.DistributedDataParallel) else model.parameters())
        else:
            outputs = model(samples)
            loss = criterion(outputs, targets)
            loss.backward()
            if config.TRAIN.CLIP_GRAD and config.TRAIN.CLIP_GRAD > 0:
                grad_norm_tensor = nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                grad_norm_val = grad_norm_tensor.item() if torch.is_tensor(grad_norm_tensor) else grad_norm_tensor
            else:
                grad_norm_val = get_grad_norm(model.module.parameters() if isinstance(model, nn.parallel.DistributedDataParallel) else model.parameters())
            optimizer.step()
        
        if lr_scheduler is not None and hasattr(lr_scheduler, 'step_update'): # Chỉ cho timm schedulers (step theo batch)
            lr_scheduler.step_update(epoch * num_steps + idx)


        if config.LOCAL_RANK != -1: torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm_val is not None: norm_meter.update(grad_norm_val)
        batch_time.update(time.time() - end_batch_time_loop)
        end_batch_time_loop = time.time()

        if (idx + 1) % config.PRINT_FREQ == 0 or idx == num_steps - 1:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() and config.LOCAL_RANK != -1 else 0
            etas = batch_time.avg * (num_steps - idx - 1)
            current_norm_display = norm_meter.val if norm_meter.count > 0 else 0.0
            avg_norm_display = norm_meter.avg if norm_meter.count > 0 else 0.0
            logger.info(
                f'Train: [{epoch + 1}/{total_epochs}][{idx + 1}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {current_norm_display:.4f} ({avg_norm_display:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time_val_loop = time.time() - start_time_epoch_loop
    logger.info(f"EPOCH {epoch + 1} training takes {datetime.timedelta(seconds=int(epoch_time_val_loop))}")

@torch.no_grad()
def validate(config, data_loader, model, logger, world_size):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    end_val_time_loop = time.time() # Đổi tên biến

    for idx, (images, target) in enumerate(data_loader):
        device_to_use = torch.device(config.LOCAL_RANK if config.LOCAL_RANK != -1 else 'cpu')
        images = images.to(device_to_use, non_blocking=True if device_to_use !='cpu' else False)
        target = target.to(device_to_use, non_blocking=True if device_to_use !='cpu' else False)
        
        output = model(images)
        loss = criterion(output, target)
        acc1_val_batch, acc5_val_batch = accuracy(output, target, topk=(1, 5))

        loss_reduced = adjusted_reduce_tensor(loss, world_size) # Sử dụng adjusted_reduce_tensor
        acc1_reduced = adjusted_reduce_tensor(acc1_val_batch.clone(), world_size)
        acc5_reduced = adjusted_reduce_tensor(acc5_val_batch.clone(), world_size)

        loss_meter.update(loss_reduced.item(), target.size(0))
        acc1_meter.update(acc1_reduced.item(), target.size(0))
        acc5_meter.update(acc5_reduced.item(), target.size(0))

        batch_time.update(time.time() - end_val_time_loop)
        end_val_time_loop = time.time()

        if (idx + 1) % config.PRINT_FREQ == 0 or idx == len(data_loader) -1:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() and config.LOCAL_RANK != -1 else 0
            logger.info(
                f'Test: [{(idx + 1)}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg # Trả về float

@torch.no_grad()
def throughput(data_loader, model, logger, config): 
    model.eval()
    use_cuda_throughput = torch.cuda.is_available() and config.LOCAL_RANK != -1
    device_to_use_throughput = torch.device(config.LOCAL_RANK if use_cuda_throughput else 'cpu')
    try:
        images, _ = next(iter(data_loader))
    except StopIteration:
        logger.error("Data loader is empty for throughput test.")
        return
    images = images.to(device_to_use_throughput, non_blocking=True if device_to_use_throughput != 'cpu' else False)
    batch_size = images.shape[0]
    warmup_iterations = 50
    measure_iterations = 30
    logger.info(f"Throughput test: Warming up ({warmup_iterations} iterations)...")
    for _ in range(warmup_iterations): model(images)
    if use_cuda_throughput: torch.cuda.synchronize()
    logger.info(f"Throughput test: Measuring with {measure_iterations} iterations for batch_size {batch_size} on device {device_to_use_throughput}")
    tic1 = time.time()
    for _ in range(measure_iterations): model(images)
    if use_cuda_throughput: torch.cuda.synchronize()
    tic2 = time.time()
    elapsed_time = tic2 - tic1
    if elapsed_time == 0: elapsed_time = 1e-6
    throughput_val = measure_iterations * batch_size / elapsed_time
    logger.info(f"Batch_size {batch_size} -> Throughput: {throughput_val:.2f} images/sec")
    return

if __name__ == '__main__':
    main()
