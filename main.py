# main.py (Version 5)
# Ghi chú fix lỗi và cải tiến:
# - Hoàn thiện parse_option: thêm --epochs, --auto-resume/--no-auto-resume, --amp/--no-amp.
#   Đặt default=None cho các args để phân biệt với giá trị mặc định trong config.
# - Distributed/Device Setup: Logic rõ ràng hơn, ưu tiên env vars từ launch utility.
#   Cập nhật config.RANK, config.WORLD_SIZE, config.LOCAL_RANK.
# - Learning Rate Scaling: Sử dụng config.WORLD_SIZE và config.DATA.BATCH_SIZE.
# - Logging: In ra nhiều thông tin config hiệu lực hơn.
# - Resume Logic: Xử lý config.TRAIN.AUTO_RESUME và config.MODEL.RESUME.
#   load_checkpoint được gọi với 5 tham số.
#   config.TRAIN.START_EPOCH được cập nhật đúng sau khi resume từ checkpoint['epoch'].
# - Training Loop:
#   - Gọi validate và UNPACK KẾT QUẢ ĐÚNG CÁCH để sửa lỗi TypeError khi cập nhật max_accuracy.
#   - Lưu checkpoint, step_epoch cho scheduler.
# - Các hàm train_one_epoch, validate, throughput nhận config để truy cập LOCAL_RANK, AMP, etc.
# - Import reduce_tensor từ utils.

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
from data import build_loader # build_loader nhận is_distributed
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
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
    parser.add_argument('--zip', action='store_true', default=None, help='Use zipped dataset (sets DATA.ZIP_MODE=True)') # Default None
    parser.add_argument('--cache-mode', type=str, default=None, choices=['no', 'full', 'part'],
                        help='Dataset cache mode (overrides config)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint (overrides config, use "" for no resume)')
    parser.add_argument('--use-checkpoint', action='store_true', default=None,
                        help="Enable gradient checkpointing (overrides config if True)")
    
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument('--amp', action='store_true', dest='amp', default=None, help="Enable AMP")
    amp_group.add_argument('--no-amp', action='store_false', dest='amp', help="Disable AMP")

    parser.add_argument('--output', default=None, type=str, metavar='PATH', help='Root of output folder (overrides config)')
    parser.add_argument('--tag', default=None, type=str, help='Tag of experiment (overrides config)')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--pretrained', type=str, default=None, help='Load pretrained weights (not a full checkpoint, for fine-tuning).')
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
    # Sử dụng hàm reduce_tensor đã import từ utils.py
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        return reduce_tensor(tensor) 
    return tensor

def main():
    args, config = parse_option()

    is_distributed = False
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        is_distributed = True
        if args.local_rank == -1 and "LOCAL_RANK" in os.environ: 
            args.local_rank = int(os.environ['LOCAL_RANK'])
    
    elif args.local_rank != -1 and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print("Warning: local_rank is set, attempting DDP. Ensure env vars for DDP are set if not using a launch utility.")
        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
            is_distributed = True
        else: # Fallback nếu WORLD_SIZE không được set đúng
            print("Warning: local_rank is set but WORLD_SIZE env var suggests single process. Running non-distributed.")
            is_distributed = False
            args.local_rank = 0 if torch.cuda.is_available() else -1


    current_gpu_or_device_str = 'cpu' 
    rank = 0
    world_size = 1
    
    if is_distributed:
        if args.local_rank == -1: 
             args.local_rank = 0 
             print("Warning: DDP mode but local_rank is invalid (-1). Defaulting to 0. This may lead to errors.")
        
        torch.cuda.set_device(args.local_rank)
        try:
            # init_method='env://' sẽ tự động đọc RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT từ biến môi trường
            dist.init_process_group(backend='nccl', init_method='env://')
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            torch.distributed.barrier() # Đảm bảo tất cả các process đã init xong
            print(f"DDP Initialized: GlobalRank={rank}, LocalRank={args.local_rank}, WorldSize={world_size} on GPU cuda:{args.local_rank}")
            current_gpu_or_device_str = f"cuda:{args.local_rank}"
        except Exception as e:
            print(f"Failed to initialize distributed group: {e}. Attempting to run non-distributed.")
            is_distributed = False # Fallback
            rank = 0
            world_size = 1
            # Fallback local_rank cho config
            args.local_rank = 0 if torch.cuda.is_available() else -1 
            if args.local_rank != -1:
                torch.cuda.set_device(args.local_rank)
                print(f"Running non-distributed on single GPU: cuda:{args.local_rank}")
                current_gpu_or_device_str = f"cuda:{args.local_rank}"
            else:
                print("Running non-distributed on CPU")
                current_gpu_or_device_str = 'cpu'
    else: # Non-distributed
        rank = 0
        world_size = 1
        if torch.cuda.is_available():
            args.local_rank = 0 # Chạy trên GPU đầu tiên
            torch.cuda.set_device(args.local_rank)
            print(f"Running on single GPU: cuda:{args.local_rank}")
            current_gpu_or_device_str = f"cuda:{args.local_rank}"
        else:
            args.local_rank = -1 # Cho CPU
            print("Running on CPU")
            current_gpu_or_device_str = 'cpu'

    config.defrost()
    config.LOCAL_RANK = args.local_rank # local_rank thực sự đang dùng (device_id hoặc -1 cho CPU)
    config.RANK = rank                  # global rank
    config.WORLD_SIZE = world_size      # tổng số process
    config.freeze()

    if is_distributed and torch.cuda.is_available() and dist.get_backend() == 'nccl':
        os.environ["NCCL_BLOCKING_WAIT"] = "1"

    seed = config.SEED + rank 
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.enabled = True
    cudnn.benchmark = True 

    if config.WORLD_SIZE > 0: # Nên luôn > 0
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
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=rank, name=f"{config.MODEL.NAME}")

    if rank == 0:
        path = os.path.join(config.OUTPUT, "config_final.json") # Đổi tên để tránh ghi đè config.json nếu có
        with open(path, "w") as f: f.write(config.dump())
        logger.info(f"Full effective config saved to {path}")

    logger.info(f"Effective config:\n{config.dump()}")
    if is_distributed: logger.info(f"Distributed training enabled with {config.WORLD_SIZE} processes.")
    logger.info(f"Running on device: {current_gpu_or_device_str}")
    logger.info(f"Batch size per process/GPU: {config.DATA.BATCH_SIZE}")
    logger.info(f"Total effective batch size: {config.DATA.BATCH_SIZE * config.WORLD_SIZE}")
    logger.info(f"Target total training epochs: {config.TRAIN.EPOCHS}")
    logger.info(f"Base LR (after scaling for total batch size {config.DATA.BATCH_SIZE * config.WORLD_SIZE}): {config.TRAIN.BASE_LR:.2e}")
    logger.info(f"Warmup LR (after scaling): {config.TRAIN.WARMUP_LR:.2e}, Min LR (after scaling): {config.TRAIN.MIN_LR:.2e}")
    logger.info(f"Warmup epochs: {config.TRAIN.WARMUP_EPOCHS}")
    logger.info(f"AMP enabled: {config.AMP}")

    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config, is_distributed)

    logger.info(f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    if config.LOCAL_RANK != -1: model.cuda(config.LOCAL_RANK)
    
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
    
    # config.TRAIN.EPOCHS là tổng số epoch muốn huấn luyện đến
    # config.TRAIN.START_EPOCH là epoch bắt đầu (0 nếu từ đầu, hoặc giá trị từ checkpoint)
    total_epochs_target = config.TRAIN.EPOCHS # Tổng số epoch mục tiêu
    # cooldown_epochs sẽ được cộng vào total_epochs_target nếu có
    effective_total_epochs_for_loop = total_epochs_target + config.TRAIN.COOLDOWN_EPOCHS


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
            logger.info(f"AUTO_RESUME: No checkpoint found in {config.OUTPUT} (path was: {config.OUTPUT}).")
    
    if actual_resume_path and actual_resume_path != '':
        logger.info(f"Attempting to load full checkpoint from: {actual_resume_path}")
        if config.MODEL.RESUME != actual_resume_path: # Cập nhật config nếu auto_resume tìm thấy file khác
             config.defrost()
             config.MODEL.RESUME = actual_resume_path
             config.freeze()
        
        # load_checkpoint cập nhật config.TRAIN.START_EPOCH bên trong nó
        max_accuracy_from_ckpt = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        max_accuracy = max(max_accuracy, max_accuracy_from_ckpt) 
        
        if not config.EVAL_MODE: 
            logger.info("Validating resumed model...")
            acc1_resumed, acc5_resumed, loss_resumed = validate(config, data_loader_val, model, logger, config.WORLD_SIZE)
            max_accuracy = max(max_accuracy, acc1_resumed) # Đảm bảo acc1_resumed là float
            logger.info(f"Resumed model validation: Acc@1 {acc1_resumed:.2f}%, Acc@5 {acc5_resumed:.2f}%, Loss {loss_resumed:.4f}")
        if config.LOCAL_RANK != -1: torch.cuda.empty_cache()
    else:
        logger.info("No checkpoint to resume from. Training from scratch or using --pretrained weights (if provided).")
        # Đảm bảo START_EPOCH là 0 nếu không resume
        config.defrost()
        config.TRAIN.START_EPOCH = 0
        config.freeze()


    if config.EVAL_MODE:
        if not config.MODEL.RESUME or config.MODEL.RESUME == '':
            logger.error("EVAL_MODE is True, but no checkpoint specified or found. Cannot evaluate.")
            return
        logger.info(f"--- Starting Evaluation on Validation Set (from checkpoint: {config.MODEL.RESUME}) ---")
        acc1_eval, acc5_eval, loss_eval = validate(config, data_loader_val, model, logger, config.WORLD_SIZE) # Unpack đúng
        logger.info(f"Evaluation Results: Acc@1 {acc1_eval:.3f} Acc@5 {acc5_eval:.3f} Loss {loss_eval:.4f}")
        # logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1_eval:.1f}%") # Có thể bị lặp
        return

    if config.THROUGHPUT_MODE:
        logger.info("--- Starting Throughput Test ---")
        throughput(data_loader_val, model, logger, config)
        return

    # config.TRAIN.START_EPOCH là epoch đã hoàn thành + 1 (hoặc 0 nếu từ đầu)
    # Vòng lặp for epoch in range(start, end) sẽ chạy từ start đến end-1.
    # Epoch hiển thị cho người dùng thường là epoch_index_trong_loop + 1.
    # Nếu START_EPOCH là epoch tiếp theo cần chạy (ví dụ, resume từ ckpt epoch 1 -> START_EPOCH=1)
    # thì vòng lặp for epoch in range(config.TRAIN.START_EPOCH, effective_total_epochs_for_loop)
    # sẽ chạy từ epoch=1 (là epoch thứ 2) đến effective_total_epochs_for_loop - 1.
    # Logger sẽ in ra epoch + 1.
    start_epoch_loop = config.TRAIN.START_EPOCH 
    if actual_resume_path and actual_resume_path != '': # Nếu resume, START_EPOCH đã được load_checkpoint đặt là completed_epoch
        logger.info(f"Resumed: START_EPOCH in config is {config.TRAIN.START_EPOCH} (completed epochs). Loop will start from this value.")
        start_epoch_loop = config.TRAIN.START_EPOCH # vòng lặp for sẽ bắt đầu từ epoch này (ví dụ epoch 1)
                                                    # và logger sẽ in epoch+1 (tức là epoch 2)

    logger.info(f"--- Starting Training from actual loop index {start_epoch_loop} (display epoch {start_epoch_loop + 1}) ---")
    logger.info(f"--- Target total epochs: {total_epochs_target} (loop runs up to {effective_total_epochs_for_loop-1}) ---")


    start_time = time.time()
    for epoch in range(start_epoch_loop, effective_total_epochs_for_loop): 
        if is_distributed and hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch) # Truyền epoch hiện tại của vòng lặp

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, logger, effective_total_epochs_for_loop)
        
        if config.LOCAL_RANK != -1: torch.cuda.empty_cache()

        acc1_val_epoch, acc5_val_epoch, loss_val_epoch = -1.0, -1.0, -1.0 
        if (epoch + 1) % config.SAVE_FREQ == 0 or (epoch + 1) == effective_total_epochs_for_loop:
            acc1_val_epoch, acc5_val_epoch, loss_val_epoch = validate(config, data_loader_val, model, logger, config.WORLD_SIZE)
            logger.info(f"Validation after epoch {epoch+1}: Acc@1 {acc1_val_epoch:.2f}%, Acc@5 {acc5_val_epoch:.2f}%, Loss {loss_val_epoch:.4f}")

            is_best = acc1_val_epoch > max_accuracy
            max_accuracy = max(max_accuracy, acc1_val_epoch) 
            logger.info(f'Max accuracy so far: {max_accuracy:.2f}%')

            if rank == 0: 
                save_checkpoint_new(config, epoch + 1, model_without_ddp, acc1_val_epoch, optimizer, lr_scheduler, logger, name=f'ckpt_epoch_{epoch+1}')
                if is_best:
                    save_checkpoint_new(config, epoch + 1, model_without_ddp, acc1_val_epoch, optimizer, lr_scheduler, logger, name='max_acc')
        
        # Step LR scheduler
        if lr_scheduler is not None:
            # Đối với ReduceLROnPlateau, step với metric sau validate
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if acc1_val_epoch != -1.0: # Chỉ step nếu đã validate ở epoch này
                    lr_scheduler.step(acc1_val_epoch)
            # Đối với các scheduler khác của PyTorch (không phải của timm) step theo epoch
            elif isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler) and not hasattr(lr_scheduler, 'step_update'):
                 lr_scheduler.step() # Gọi sau mỗi epoch
            # Scheduler của timm có thể có step_epoch (ít phổ biến) hoặc step_update (theo batch, đã gọi trong train_one_epoch)
            elif hasattr(lr_scheduler, 'step_epoch') and not hasattr(lr_scheduler, 'step_update'):
                 lr_scheduler.step_epoch(epoch + 1)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Total training time for this run: {total_time_str}')


# --- Các hàm train_one_epoch, validate, throughput (phiên bản Version 4 đã cung cấp trước đó) ---
# (Đảm bảo chúng sử dụng `config` để lấy LOCAL_RANK, WORLD_SIZE, AMP)
def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, logger, total_epochs_for_loop):
    model.train()
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    start_time_batch_loop = time.time() 
    end_time_batch = time.time()

    use_amp_scaler = config.AMP and torch.cuda.is_available() and config.LOCAL_RANK != -1
    scaler = GradScaler() if use_amp_scaler else None

    for idx, (samples, targets) in enumerate(data_loader):
        optimizer.zero_grad()
        
        device_to_use = torch.device(config.LOCAL_RANK if config.LOCAL_RANK !=-1 else 'cpu')
        samples = samples.to(device_to_use, non_blocking=True if str(device_to_use) != 'cpu' else False)
        targets = targets.to(device_to_use, non_blocking=True if str(device_to_use) != 'cpu' else False)


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
                # grad_norm_val = get_grad_norm(model.parameters()) # Có thể lấy ở đây nếu cần
                pass
        else:
            outputs = model(samples)
            loss = criterion(outputs, targets)
            loss.backward()
            if config.TRAIN.CLIP_GRAD and config.TRAIN.CLIP_GRAD > 0:
                grad_norm_tensor = nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                grad_norm_val = grad_norm_tensor.item() if torch.is_tensor(grad_norm_tensor) else grad_norm_tensor
            else:
                grad_norm_val = get_grad_norm(model.parameters()) # Cần cẩn thận với DDP, nên dùng model_without_ddp
            optimizer.step()
        
        if lr_scheduler is not None and hasattr(lr_scheduler, 'step_update'): # Chỉ cho timm schedulers step theo batch
            lr_scheduler.step_update(epoch * num_steps + idx)


        if config.LOCAL_RANK != -1: torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm_val is not None: norm_meter.update(grad_norm_val)
        batch_time.update(time.time() - end_time_batch)
        end_time_batch = time.time()

        if (idx + 1) % config.PRINT_FREQ == 0 or idx == num_steps - 1:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() and config.LOCAL_RANK != -1 else 0
            etas = batch_time.avg * (num_steps - idx - 1)
            current_norm_display = norm_meter.val if norm_meter.count > 0 else 0.0
            avg_norm_display = norm_meter.avg if norm_meter.count > 0 else 0.0
            logger.info(
                f'Train: [{epoch + 1}/{total_epochs_for_loop}][{idx + 1}/{num_steps}]\t' # Hiển thị epoch thực tế
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {current_norm_display:.4f} ({avg_norm_display:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time_val = time.time() - start_time_batch_loop
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
        images = images.to(device_to_use, non_blocking=True if str(device_to_use) != 'cpu' else False)
        target = target.to(device_to_use, non_blocking=True if str(device_to_use) != 'cpu' else False)
        
        output = model(images)
        loss = criterion(output, target)
        acc1_val_batch, acc5_val_batch = accuracy(output, target, topk=(1, 5))

        loss_reduced = adjusted_reduce_tensor(loss, world_size) # Sử dụng hàm đã được sửa trong main.py
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
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg # Trả về 3 giá trị float

@torch.no_grad()
def throughput(data_loader, model, logger, config): 
    model.eval()
    use_cuda_throughput = torch.cuda.is_available() and config.LOCAL_RANK != -1
    # Sửa: device_to_use_throughput nên là torch.device object
    device_to_use_throughput = torch.device(config.LOCAL_RANK if use_cuda_throughput else 'cpu')

    try:
        images, _ = next(iter(data_loader))
    except StopIteration:
        logger.error("Data loader is empty for throughput test.")
        return

    images = images.to(device_to_use_throughput, non_blocking=True if str(device_to_use_throughput) != 'cpu' else False)
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
