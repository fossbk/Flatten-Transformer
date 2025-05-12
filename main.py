# main.py (Version 5)
# Ghi chú fix lỗi:
# - Sửa lỗi TypeError khi so sánh max_accuracy với acc1_val bằng cách đảm bảo
#   kết quả từ hàm validate được unpack chính xác và acc1_val (hoặc tên biến tương ứng)
#   thực sự là giá trị float của Acc@1.
# - Các thay đổi từ Version 4 (xử lý epoch, resume, single-GPU, logger, v.v.) được giữ nguyên.

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
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint_new, get_grad_norm, auto_resume_helper, reduce_tensor, load_pretrained

import warnings
warnings.filterwarnings('ignore')

# (parse_option giữ nguyên như Version 4)
def parse_option():
    parser = argparse.ArgumentParser('FLatten Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE", default=None,
                        help='Path to config file (optional)')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--batch-size', type=int, default=None, help="Batch size per GPU (overrides config)")
    parser.add_argument('--data-path', type=str, default=None, help='Path to dataset (overrides config)')
    parser.add_argument('--zip', action='store_true', help='Use zipped dataset (sets DATA.ZIP_MODE=True)')
    parser.add_argument('--cache-mode', type=str, default=None, choices=['no', 'full', 'part'],
                        help='Dataset cache mode (overrides config)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint (overrides config, use "" for no resume)')
    parser.add_argument('--use-checkpoint', action='store_true', default=None,
                        help="Enable gradient checkpointing (overrides config)")

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument('--amp', action='store_true', dest='amp', default=None, help="Enable AMP")
    amp_group.add_argument('--no-amp', action='store_false', dest='amp', help="Disable AMP")

    parser.add_argument('--output', default=None, type=str, metavar='PATH', help='Root of output folder (overrides config)')
    parser.add_argument('--tag', default=None, type=str, help='Tag of experiment (overrides config)')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--pretrained', type=str, default=None, help='Load pretrained weights (not a full checkpoint).')
    parser.add_argument('--find-unused-params', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=None, help="Total number of training epochs (overrides config)")

    auto_resume_group = parser.add_mutually_exclusive_group()
    auto_resume_group.add_argument('--auto-resume', action='store_true', dest='auto_resume', default=None)
    auto_resume_group.add_argument('--no-auto-resume', action='store_false', dest='auto_resume')

    parser.add_argument('--local_rank', type=int, default=os.environ.get('LOCAL_RANK', -1),
                        help='Local rank for distributed training. Set by launch utility.')

    args = parser.parse_args()
    config = get_config(args)
    return args, config

def adjusted_reduce_tensor(tensor, world_size):
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        return reduce_tensor(tensor) # Gọi hàm reduce_tensor từ utils.py
    return tensor

def main():
    args, config = parse_option()

    # --- (Phần DISTRIBUTED/DEVICE SETUP giữ nguyên như Version 4) ---
    is_distributed = False
    if 'WORLD_SIZE' in os.environ:
        world_size_env = int(os.environ['WORLD_SIZE'])
        if world_size_env > 1:
            is_distributed = True
            if args.local_rank == -1 and "LOCAL_RANK" in os.environ:
                args.local_rank = int(os.environ['LOCAL_RANK'])
    elif args.local_rank != -1 and torch.cuda.device_count() > 1:
        print("Warning: local_rank is set, attempting DDP. Ensure env vars for DDP are set if not using a launch utility.")
        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
            is_distributed = True
        else:
            print("Warning: local_rank is set but WORLD_SIZE env var suggests single process. Running non-distributed.")
            args.local_rank = 0 if torch.cuda.is_available() else -1

    current_gpu_or_device = 'cpu'
    rank = 0
    world_size = 1
    local_rank_for_config_update = -1

    if is_distributed:
        if args.local_rank == -1:
             args.local_rank = 0
             print("Warning: DDP mode but local_rank is -1. Defaulting to 0. This might be incorrect.")
        torch.cuda.set_device(args.local_rank)
        try:
            dist.init_process_group(backend='nccl', init_method='env://')
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_rank_for_config_update = args.local_rank
            torch.distributed.barrier()
            print(f"DDP Initialized: GlobalRank={rank}, LocalRank={args.local_rank}, WorldSize={world_size} on GPU cuda:{args.local_rank}")
            current_gpu_or_device = args.local_rank
        except Exception as e:
            print(f"Failed to initialize distributed group: {e}. Attempting to run non-distributed on GPU 0 or CPU.")
            is_distributed = False
            rank = 0
            world_size = 1
            local_rank_for_config_update = 0 if torch.cuda.is_available() else -1
            if local_rank_for_config_update != -1:
                torch.cuda.set_device(local_rank_for_config_update)
                print(f"Running non-distributed on single GPU: cuda:{local_rank_for_config_update}")
                current_gpu_or_device = local_rank_for_config_update
            else:
                print("Running non-distributed on CPU")
                current_gpu_or_device = 'cpu'
    else:
        rank = 0
        world_size = 1
        if torch.cuda.is_available():
            local_rank_for_config_update = 0
            torch.cuda.set_device(local_rank_for_config_update)
            print(f"Running on single GPU: cuda:{local_rank_for_config_update}")
            current_gpu_or_device = local_rank_for_config_update
        else:
            local_rank_for_config_update = -1
            print("Running on CPU")
            current_gpu_or_device = 'cpu'

    config.defrost()
    config.LOCAL_RANK = local_rank_for_config_update
    config.RANK = rank
    config.WORLD_SIZE = world_size
    config.freeze()

    if is_distributed and torch.cuda.is_available() and dist.get_backend() == 'nccl':
        os.environ["NCCL_BLOCKING_WAIT"] = "1"

    seed = config.SEED + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.enabled = True
    cudnn.benchmark = True

    if config.WORLD_SIZE > 0:
        eff_batch_size = config.DATA.BATCH_SIZE * config.WORLD_SIZE
        base_lr = config.TRAIN.BASE_LR if hasattr(config.TRAIN, 'BASE_LR') else 5e-4
        warmup_lr = config.TRAIN.WARMUP_LR if hasattr(config.TRAIN, 'WARMUP_LR') else 5e-7
        min_lr = config.TRAIN.MIN_LR if hasattr(config.TRAIN, 'MIN_LR') else 5e-6
        linear_scaled_lr = base_lr * eff_batch_size / 512.0
        linear_scaled_warmup_lr = warmup_lr * eff_batch_size / 512.0
        linear_scaled_min_lr = min_lr * eff_batch_size / 512.0
        config.defrost()
        config.TRAIN.BASE_LR = linear_scaled_lr
        config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
        config.TRAIN.MIN_LR = linear_scaled_min_lr
        config.freeze()
    
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=rank, name=f"{config.MODEL.NAME}")

    if rank == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f: f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    logger.info(f"Effective config:\n{config.dump()}")
    if is_distributed: logger.info(f"Distributed training enabled with {config.WORLD_SIZE} processes.")
    logger.info(f"Running on device: {current_gpu_or_device}") # In ra device đang chạy
    logger.info(f"Batch size per process/GPU: {config.DATA.BATCH_SIZE}")
    logger.info(f"Total effective batch size: {config.DATA.BATCH_SIZE * config.WORLD_SIZE}")
    logger.info(f"Training for {config.TRAIN.EPOCHS} epochs.")
    logger.info(f"Base LR (after scaling): {config.TRAIN.BASE_LR:.2e}")
    logger.info(f"Warmup LR (after scaling): {config.TRAIN.WARMUP_LR:.2e}, Min LR (after scaling): {config.TRAIN.MIN_LR:.2e}")
    logger.info(f"Warmup epochs: {config.TRAIN.WARMUP_EPOCHS}")
    logger.info(f"AMP enabled: {config.AMP}")

    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config, is_distributed)

    logger.info(f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    if config.LOCAL_RANK != -1: model.cuda(config.LOCAL_RANK)
    # logger.info(str(model)) # Comment lại để output gọn hơn

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
    total_epochs = config.TRAIN.EPOCHS + config.TRAIN.COOLDOWN_EPOCHS

    if config.AUG.MIXUP > 0. or config.AUG.CUTMIX > 0.: criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.: criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else: criterion = nn.CrossEntropyLoss()

    max_accuracy = 0.0 # Khởi tạo max_accuracy

    if hasattr(args, 'pretrained') and args.pretrained and args.pretrained != '':
        logger.info(f"Loading pretrained weights (from --pretrained) from: {args.pretrained}")
        load_pretrained(args.pretrained, model_without_ddp, logger)

    actual_resume_path = config.MODEL.RESUME
    if not actual_resume_path and config.TRAIN.AUTO_RESUME:
        resume_file_auto = auto_resume_helper(config.OUTPUT)
        if resume_file_auto:
            logger.info(f"AUTO_RESUME: Found checkpoint '{resume_file_auto}' in output directory.")
            actual_resume_path = resume_file_auto
        else:
            logger.info(f"AUTO_RESUME: No checkpoint found in {config.OUTPUT}.")
    
    if actual_resume_path and actual_resume_path != '':
        logger.info(f"Attempting to load full checkpoint from: {actual_resume_path}")
        if config.MODEL.RESUME != actual_resume_path:
             config.defrost()
             config.MODEL.RESUME = actual_resume_path
             config.freeze()
        
        # load_checkpoint nên cập nhật config.TRAIN.START_EPOCH bên trong nó
        # và trả về max_accuracy từ checkpoint
        max_accuracy_from_ckpt = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        max_accuracy = max(max_accuracy, max_accuracy_from_ckpt) # Cập nhật max_accuracy
        
        if not config.EVAL_MODE: # Chỉ validate nếu không phải EVAL_MODE
            logger.info("Validating resumed model...")
            # ---- SỬA LỖI TypeError Ở ĐÂY ----
            acc1_resumed, acc5_resumed, loss_resumed = validate(config, data_loader_val, model, logger, config.WORLD_SIZE)
            max_accuracy = max(max_accuracy, acc1_resumed) # Sử dụng acc1_resumed (float)
            logger.info(f"Resumed model validation: Acc@1 {acc1_resumed:.2f}%, Acc@5 {acc5_resumed:.2f}%, Loss {loss_resumed:.4f}")
        if config.LOCAL_RANK != -1: torch.cuda.empty_cache()
    else:
        logger.info("No checkpoint to resume from. Training from scratch or using --pretrained weights (if provided).")

    if config.EVAL_MODE:
        if not config.MODEL.RESUME or config.MODEL.RESUME == '':
            logger.error("EVAL_MODE is True, but no checkpoint specified or found. Cannot evaluate.")
            return
        logger.info(f"--- Starting Evaluation on Validation Set (from checkpoint: {config.MODEL.RESUME}) ---")
        acc1_eval, acc5_eval, loss_eval = validate(config, data_loader_val, model, logger, config.WORLD_SIZE) # Unpack đúng
        logger.info(f"Evaluation Results: Acc@1 {acc1_eval:.3f} Acc@5 {acc5_eval:.3f} Loss {loss_eval:.4f}")
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1_eval:.1f}%")
        return

    if config.THROUGHPUT_MODE:
        logger.info("--- Starting Throughput Test ---")
        throughput(data_loader_val, model, logger, config)
        return

    logger.info(f"--- Starting Training from epoch {config.TRAIN.START_EPOCH + 1} up to total_epochs {total_epochs} ---") # Sửa log

    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, total_epochs):
        if is_distributed and hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, logger, total_epochs)
        
        if config.LOCAL_RANK != -1: torch.cuda.empty_cache()

        current_acc1 = -1.0 # Khởi tạo để dùng cho scheduler nếu không validate ở epoch này
        if (epoch + 1) % config.SAVE_FREQ == 0 or (epoch + 1) == total_epochs:
            # ---- SỬA LỖI TypeError Ở ĐÂY ----
            acc1_val_epoch, acc5_val_epoch, loss_val_epoch = validate(config, data_loader_val, model, logger, config.WORLD_SIZE) # Unpack đúng 3 giá trị
            current_acc1 = acc1_val_epoch # Gán giá trị float cho current_acc1
            logger.info(f"Validation after epoch {epoch+1}: Acc@1 {current_acc1:.2f}%, Acc@5 {acc5_val_epoch:.2f}%, Loss {loss_val_epoch:.4f}")

            is_best = current_acc1 > max_accuracy # So sánh float với float
            max_accuracy = max(max_accuracy, current_acc1) # max(float, float)
            logger.info(f'Max accuracy so far: {max_accuracy:.2f}%')

            if rank == 0:
                save_checkpoint_new(config, epoch + 1, model_without_ddp, current_acc1, optimizer, lr_scheduler, logger, name=f'ckpt_epoch_{epoch+1}')
                if is_best:
                    save_checkpoint_new(config, epoch + 1, model_without_ddp, current_acc1, optimizer, lr_scheduler, logger, name='max_acc')
        
        if lr_scheduler is not None:
            # Đối với ReduceLROnPlateau, step với metric (ví dụ acc1)
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if current_acc1 != -1.0 : # Chỉ step nếu có metric từ validate của epoch này
                    lr_scheduler.step(current_acc1)
            # Đối với các scheduler khác step theo epoch (ví dụ StepLR, MultiStepLR, CosineAnnealingLR của PyTorch)
            elif not hasattr(lr_scheduler, 'step_update'): # Nếu không phải scheduler của timm step theo batch
                lr_scheduler.step() # Gọi sau mỗi epoch

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Total training time {total_time_str}')


# --- Dán các hàm train_one_epoch, validate, throughput đã được sửa từ Version 4 vào đây ---
# (Đảm bảo chúng nhận config và sử dụng config.LOCAL_RANK, config.WORLD_SIZE, config.AMP đúng cách)
def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, logger, total_epochs):
    model.train()
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    start_time_epoch = time.time()
    end_batch_time = time.time()

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
                grad_norm_val = get_grad_norm(model.parameters()) # Có thể cần model_without_ddp.parameters() nếu DDP
        else:
            outputs = model(samples)
            loss = criterion(outputs, targets)
            loss.backward()
            if config.TRAIN.CLIP_GRAD and config.TRAIN.CLIP_GRAD > 0:
                grad_norm_tensor = nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                grad_norm_val = grad_norm_tensor.item() if torch.is_tensor(grad_norm_tensor) else grad_norm_tensor
            else:
                grad_norm_val = get_grad_norm(model.parameters()) # Có thể cần model_without_ddp.parameters() nếu DDP
            optimizer.step()
        
        if lr_scheduler is not None and hasattr(lr_scheduler, 'step_update'): # Chỉ cho timm schedulers
            lr_scheduler.step_update(epoch * num_steps + idx)


        if config.LOCAL_RANK != -1: torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm_val is not None: norm_meter.update(grad_norm_val)
        batch_time.update(time.time() - end_batch_time)
        end_batch_time = time.time()

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
    epoch_time_val = time.time() - start_time_epoch
    logger.info(f"EPOCH {epoch + 1} training takes {datetime.timedelta(seconds=int(epoch_time_val))}")

@torch.no_grad()
def validate(config, data_loader, model, logger, world_size):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    end_val_time = time.time()

    for idx, (images, target) in enumerate(data_loader):
        device_to_use = torch.device(config.LOCAL_RANK if config.LOCAL_RANK != -1 else 'cpu')
        images = images.to(device_to_use, non_blocking=True if device_to_use !='cpu' else False)
        target = target.to(device_to_use, non_blocking=True if device_to_use !='cpu' else False)
        
        output = model(images)
        loss = criterion(output, target)
        acc1_val_batch, acc5_val_batch = accuracy(output, target, topk=(1, 5))

        # adjusted_reduce_tensor sẽ gọi utils.reduce_tensor nếu cần
        loss_reduced = adjusted_reduce_tensor(loss, world_size)
        acc1_reduced = adjusted_reduce_tensor(acc1_val_batch.clone(), world_size)
        acc5_reduced = adjusted_reduce_tensor(acc5_val_batch.clone(), world_size)

        loss_meter.update(loss_reduced.item(), target.size(0))
        acc1_meter.update(acc1_reduced.item(), target.size(0))
        acc5_meter.update(acc5_reduced.item(), target.size(0))

        batch_time.update(time.time() - end_val_time)
        end_val_time = time.time()

        if (idx + 1) % config.PRINT_FREQ == 0 or idx == len(data_loader) -1:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() and config.LOCAL_RANK != -1 else 0
            logger.info(
                f'Test: [{(idx + 1)}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t' # Đây là acc1 của batch cuối cùng (sau reduce)
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t' # Đây là acc5 của batch cuối cùng (sau reduce)
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}') # Đây là acc trung bình của toàn bộ val set
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg # Trả về các giá trị float

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
