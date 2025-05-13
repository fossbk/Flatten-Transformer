# main.py (Version 5.1)
# Ghi chú fix lỗi và cải tiến:
# - Sửa lỗi TypeError khi so sánh max_accuracy với acc1_val bằng cách đảm bảo
#   kết quả từ hàm validate được unpack chính xác và biến chứa Acc@1 (float)
#   được sử dụng đúng cách trong hàm max().
# - Hoàn thiện parse_option: thêm --epochs, --auto-resume/--no-auto-resume, --amp/--no-amp.
#   Đặt default=None cho các args để phân biệt với giá trị mặc định trong config.
# - Distributed/Device Setup: Logic rõ ràng hơn, ưu tiên env vars từ launch utility.
#   Cập nhật config.RANK, config.WORLD_SIZE, config.LOCAL_RANK.
# - Learning Rate Scaling: Sử dụng config.WORLD_SIZE và config.DATA.BATCH_SIZE.
# - Logging: In ra nhiều thông tin config hiệu lực hơn.
# - Resume Logic: Xử lý config.TRAIN.AUTO_RESUME và config.MODEL.RESUME.
#   load_checkpoint được gọi với 5 tham số.
#   config.TRAIN.START_EPOCH được cập nhật đúng sau khi resume từ checkpoint['epoch'].
# - Training Loop: Gọi validate có điều kiện, lưu checkpoint, step_epoch cho scheduler.
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

from config import get_config # Nên là Version 4 của config.py
from models import build_model
from data import build_loader # Nên là Version 4 của data/build.py
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint_new, get_grad_norm, auto_resume_helper, reduce_tensor, load_pretrained # Nên là Version 4 của utils.py

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
    parser.add_argument('--zip', action='store_true', default=None, help='Use zipped dataset (sets DATA.ZIP_MODE=True if present)')
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
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        return reduce_tensor(tensor) 
    return tensor

def main():
    args, config = parse_option()

    is_distributed = False
    rank = 0 
    world_size = 1
    
    if 'WORLD_SIZE' in os.environ:
        world_size_env = int(os.environ['WORLD_SIZE'])
        if world_size_env > 1:
            is_distributed = True
            if args.local_rank == -1 and "LOCAL_RANK" in os.environ: 
                args.local_rank = int(os.environ['LOCAL_RANK'])
    elif args.local_rank != -1 and torch.cuda.is_available() and torch.cuda.device_count() > 1 : # User manually set local_rank for DDP
        print("Warning: local_rank is set, attempting DDP. Ensure MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE env vars are set if not using a launch utility or if init_method is 'env://'.")
        # Trong trường hợp này, WORLD_SIZE và RANK nên được lấy từ env var hoặc opts
        # Để đơn giản, nếu dùng --local_rank khác -1 và có nhiều GPU, ta giả định người dùng muốn DDP
        # và các env var khác sẽ được cung cấp hoặc init_method không phải 'env://'
        # Hoặc code có thể cần phức tạp hơn để tự tính WORLD_SIZE và RANK nếu không có env var
        # Tạm thời, nếu WORLD_SIZE env var không có, ta sẽ không kích hoạt DDP để tránh lỗi init.
        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
            is_distributed = True
        else:
            print("Warning: local_rank is set but WORLD_SIZE env var suggests single process or is not set. Running non-distributed.")
            is_distributed = False # Fallback to non-distributed
            args.local_rank = 0 if torch.cuda.is_available() else -1 # Chạy trên GPU đầu tiên nếu có


    current_gpu_or_device_str = 'cpu'
    
    if is_distributed:
        if args.local_rank == -1 : # Should have been set by launch utility or logic above
             args.local_rank = 0 # Default, but problematic for DDP if not rank 0
             print("CRITICAL Warning: DDP mode but local_rank is -1. Defaulting to 0. This WILL LIKELY BE INCORRECT if not actually process 0.")
        
        torch.cuda.set_device(args.local_rank)
        try:
            dist.init_process_group(backend='nccl', init_method='env://')
            rank = dist.get_rank()
            world_size = dist.get_world_size() # Lấy world_size thực tế từ DDP
            torch.distributed.barrier()
            print(f"DDP Initialized: GlobalRank={rank}, LocalRank={args.local_rank}, WorldSize={world_size} on GPU cuda:{args.local_rank}")
            current_gpu_or_device_str = f"cuda:{args.local_rank}"
        except Exception as e:
            print(f"Failed to initialize distributed group: {e}. Running non-distributed.")
            is_distributed = False
            rank = 0
            world_size = 1
            args.local_rank = 0 if torch.cuda.is_available() else -1
            if args.local_rank != -1:
                torch.cuda.set_device(args.local_rank)
                current_gpu_or_device_str = f"cuda:{args.local_rank}"
                print(f"Running non-distributed on single GPU: {current_gpu_or_device_str}")
            else:
                current_gpu_or_device_str = 'cpu'
                print("Running non-distributed on CPU")
    else: 
        rank = 0
        world_size = 1
        if torch.cuda.is_available():
            args.local_rank = 0 
            torch.cuda.set_device(args.local_rank)
            current_gpu_or_device_str = f"cuda:{args.local_rank}"
            print(f"Running on single GPU: {current_gpu_or_device_str}")
        else:
            args.local_rank = -1 
            current_gpu_or_device_str = 'cpu'
            print("Running on CPU")

    config.defrost()
    config.LOCAL_RANK = args.local_rank 
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
        path = os.path.join(config.OUTPUT, "config_final_effective.json") 
        with open(path, "w") as f: f.write(config.dump())
        logger.info(f"Full effective config saved to {path}")

    logger.info(f"Effective config:\n{config.dump()}")
    if is_distributed: logger.info(f"Distributed training enabled with {config.WORLD_SIZE} processes.")
    logger.info(f"Running on device: {current_gpu_or_device_str}")
    logger.info(f"Batch size per process/GPU: {config.DATA.BATCH_SIZE}")
    logger.info(f"Total effective batch size: {config.DATA.BATCH_SIZE * config.WORLD_SIZE}")
    logger.info(f"Target total training epochs: {config.TRAIN.EPOCHS}") # Tổng số epoch muốn đạt đến
    logger.info(f"Base LR (scaled): {config.TRAIN.BASE_LR:.2e}, Warmup LR (scaled): {config.TRAIN.WARMUP_LR:.2e}, Min LR (scaled): {config.TRAIN.MIN_LR:.2e}")
    logger.info(f"Warmup epochs in config: {config.TRAIN.WARMUP_EPOCHS}")
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
    # config.TRAIN.START_EPOCH là epoch bắt đầu (0 nếu từ đầu, hoặc giá trị từ checkpoint là epoch đã hoàn thành)
    total_epochs_target = config.TRAIN.EPOCHS
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
        if config.MODEL.RESUME != actual_resume_path:
             config.defrost(); config.MODEL.RESUME = actual_resume_path; config.freeze()
        
        # load_checkpoint cập nhật config.TRAIN.START_EPOCH bên trong nó (thành completed_epoch)
        # và trả về max_accuracy từ checkpoint
        max_accuracy_from_ckpt = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        max_accuracy = max(max_accuracy, max_accuracy_from_ckpt) 
        
        if not config.EVAL_MODE: 
            logger.info("Validating resumed model...")
            # --- ĐẢM BẢO UNPACK ĐÚNG KẾT QUẢ TỪ VALIDATE ---
            acc1_resumed_float, acc5_resumed_float, loss_resumed_float = validate(config, data_loader_val, model, logger, config.WORLD_SIZE)
            max_accuracy = max(max_accuracy, acc1_resumed_float) # Sử dụng giá trị float đã unpack
            logger.info(f"Resumed model validation: Acc@1 {acc1_resumed_float:.2f}%, Acc@5 {acc5_resumed_float:.2f}%, Loss {loss_resumed_float:.4f}")
        if config.LOCAL_RANK != -1: torch.cuda.empty_cache()
    else:
        logger.info("No checkpoint to resume from. Training from scratch or using --pretrained weights (if provided).")
        config.defrost(); config.TRAIN.START_EPOCH = 0; config.freeze() # Đảm bảo START_EPOCH là 0 nếu không resume


    if config.EVAL_MODE:
        if not config.MODEL.RESUME or config.MODEL.RESUME == '':
            logger.error("EVAL_MODE is True, but no checkpoint specified or found. Cannot evaluate.")
            return
        logger.info(f"--- Starting Evaluation on Validation Set (from checkpoint: {config.MODEL.RESUME}) ---")
        acc1_eval, acc5_eval, loss_eval = validate(config, data_loader_val, model, logger, config.WORLD_SIZE) # Unpack đúng
        logger.info(f"Evaluation Results: Acc@1 {acc1_eval:.3f} Acc@5 {acc5_eval:.3f} Loss {loss_eval:.4f}")
        return

    if config.THROUGHPUT_MODE:
        logger.info("--- Starting Throughput Test ---")
        throughput(data_loader_val, model, logger, config)
        return

    # config.TRAIN.START_EPOCH là epoch đã hoàn thành (ví dụ, 0 nếu từ đầu, 1 nếu resume từ ckpt_epoch_1)
    # Vòng lặp for epoch in range(start_epoch, total_epochs_target) sẽ chạy từ start_epoch đến total_epochs_target - 1
    # Logger sẽ in ra epoch + 1 làm số epoch hiển thị
    start_epoch_for_loop = config.TRAIN.START_EPOCH # epoch đã hoàn thành
    
    logger.info(f"--- Starting Training from completed epoch {start_epoch_for_loop} (next epoch to run: {start_epoch_for_loop + 1}) ---")
    logger.info(f"--- Target total epochs: {total_epochs_target} (loop will run up to epoch index {effective_total_epochs_for_loop -1}) ---")

    start_time = time.time()
    # Vòng lặp chạy từ epoch đã hoàn thành (ví dụ 0) đến tổng số epoch mục tiêu (ví dụ 30)
    # epoch_idx sẽ là 0, 1, ..., 29 nếu START_EPOCH=0 và total_epochs_target=30
    for epoch_idx in range(start_epoch_for_loop, effective_total_epochs_for_loop):
        # epoch_display là epoch_idx + 1, là số epoch hiển thị cho người dùng (1-based)
        current_display_epoch = epoch_idx + 1

        if is_distributed and hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch_idx) # Truyền epoch_idx (0-based)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch_idx, mixup_fn, lr_scheduler, logger, effective_total_epochs_for_loop)
        
        if config.LOCAL_RANK != -1: torch.cuda.empty_cache()

        current_epoch_acc1 = -1.0 
        if (current_display_epoch % config.SAVE_FREQ == 0) or (current_display_epoch == effective_total_epochs_for_loop):
            # ---- ĐẢM BẢO UNPACK ĐÚNG KẾT QUẢ TỪ VALIDATE ----
            acc1_val_epoch_float, acc5_val_epoch_float, loss_val_epoch_float = validate(config, data_loader_val, model, logger, config.WORLD_SIZE)
            current_epoch_acc1 = acc1_val_epoch_float # Gán giá trị float
            logger.info(f"Validation after epoch {current_display_epoch}: Acc@1 {current_epoch_acc1:.2f}%, Acc@5 {acc5_val_epoch_float:.2f}%, Loss {loss_val_epoch_float:.4f}")

            is_best = current_epoch_acc1 > max_accuracy # So sánh float với float
            max_accuracy = max(max_accuracy, current_epoch_acc1) # max(float, float)
            logger.info(f'Max accuracy so far: {max_accuracy:.2f}%')

            if rank == 0: 
                save_checkpoint_new(config, current_display_epoch, model_without_ddp, current_epoch_acc1, optimizer, lr_scheduler, logger, name=f'ckpt_epoch_{current_display_epoch}')
                if is_best:
                    save_checkpoint_new(config, current_display_epoch, model_without_ddp, current_epoch_acc1, optimizer, lr_scheduler, logger, name='max_acc')
        
        if lr_scheduler is not None:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if current_epoch_acc1 != -1.0 : 
                    lr_scheduler.step(current_epoch_acc1)
            elif not hasattr(lr_scheduler, 'step_update'): # Schedulers của PyTorch gốc
                 lr_scheduler.step() # Gọi sau mỗi epoch_idx

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Total training time for this run: {total_time_str}')

# --- Các hàm train_one_epoch, validate, throughput (phiên bản Version 4) ---
# (Dán lại toàn bộ 3 hàm này từ phiên bản utils.py Version 4 / main.py Version 4 vào đây)
# Đảm bảo chúng sử dụng `config` để lấy LOCAL_RANK, WORLD_SIZE, AMP
# và hàm validate trả về 3 giá trị float.

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch_idx, mixup_fn, lr_scheduler, logger, total_epochs_for_loop):
    model.train()
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    start_time_batch_loop = time.time() 
    end_time_batch = time.time()

    use_amp_scaler = config.AMP and torch.cuda.is_available() and config.LOCAL_RANK != -1
    scaler = GradScaler() if use_amp_scaler else None

    for batch_idx, (samples, targets) in enumerate(data_loader):
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
                # grad_norm_val = get_grad_norm(model.parameters())
                pass # Hoặc tính grad_norm sau scaler.update() nếu cần
        else:
            outputs = model(samples)
            loss = criterion(outputs, targets)
            loss.backward()
            model_params_for_grad_norm = model.module.parameters() if isinstance(model, nn.parallel.DistributedDataParallel) else model.parameters()
            if config.TRAIN.CLIP_GRAD and config.TRAIN.CLIP_GRAD > 0:
                grad_norm_tensor = nn.utils.clip_grad_norm_(model_params_for_grad_norm, config.TRAIN.CLIP_GRAD)
                grad_norm_val = grad_norm_tensor.item() if torch.is_tensor(grad_norm_tensor) else grad_norm_tensor
            else:
                grad_norm_val = get_grad_norm(model_params_for_grad_norm)
            optimizer.step()
        
        if lr_scheduler is not None and hasattr(lr_scheduler, 'step_update'): # timm schedulers
            lr_scheduler.step_update(epoch_idx * num_steps + batch_idx)


        if config.LOCAL_RANK != -1: torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm_val is not None: norm_meter.update(grad_norm_val)
        batch_time.update(time.time() - end_time_batch)
        end_time_batch = time.time()

        if (batch_idx + 1) % config.PRINT_FREQ == 0 or batch_idx == num_steps - 1:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() and config.LOCAL_RANK != -1 else 0
            etas = batch_time.avg * (num_steps - batch_idx - 1)
            current_norm_display = norm_meter.val if norm_meter.count > 0 else (grad_norm_val if grad_norm_val is not None else 0.0)
            avg_norm_display = norm_meter.avg if norm_meter.count > 0 else 0.0
            logger.info(
                f'Train: [{epoch_idx + 1}/{total_epochs_for_loop}][{batch_idx + 1}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {current_norm_display:.4f} ({avg_norm_display:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time_val = time.time() - start_time_batch_loop
    logger.info(f"EPOCH {epoch_idx + 1} training takes {datetime.timedelta(seconds=int(epoch_time_val))}")

@torch.no_grad()
def validate(config, data_loader, model, logger, world_size):
    criterion = nn.CrossEntropyLoss()
    model.eval() # Quan trọng: đặt model ở chế độ eval
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

        loss_reduced = adjusted_reduce_tensor(loss.clone(), world_size) # Clone trước khi reduce
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
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg

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
    if elapsed_time == 0: elapsed_time = 1e-6 # Tránh chia cho 0
    throughput_val = measure_iterations * batch_size / elapsed_time
    logger.info(f"Batch_size {batch_size} -> Throughput: {throughput_val:.2f} images/sec")
    return

if __name__ == '__main__':
    main()
