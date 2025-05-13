# utils.py (Version 4)
# Ghi chú fix lỗi:
# - Thêm lại hàm `reduce_tensor` đã bị thiếu.
# - load_checkpoint:
#   - Thêm weights_only=False vào torch.load().
#   - Đảm bảo nhận logger làm tham số và sử dụng nó.
#   - Xử lý state_dict DDP/non-DDP một cách linh hoạt.
#   - Cập nhật config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1.
#   - Load optimizer, scheduler, max_accuracy nếu có và không ở EVAL_MODE.
#   - Thêm cảnh báo nếu NUM_CLASSES không khớp.
# - save_checkpoint_new:
#   - Lưu model.state_dict() từ model_without_ddp.
#   - Lưu epoch, optimizer, lr_scheduler, max_accuracy.
#   - KHÔNG lưu toàn bộ đối tượng config (CfgNode).

import os
import torch
import torch.distributed as dist
import pickle
from yacs.config import CfgNode as CN # Cần cho isinstance nếu kiểm tra config trong checkpoint
# from logger import create_logger # Không nên import create_logger ở đây, logger được truyền vào


# THÊM LẠI HÀM REDUCE_TENSOR
def reduce_tensor(tensor):
    """
    Reduces a tensor from all processes to rank 0 and averages it.
    Assumes tensor is on the correct device.
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def load_pretrained(pretrained_path, model, logger):
    logger.info(f"==============> Loading weight {pretrained_path} for fine-tuning......")
    if not os.path.isfile(pretrained_path):
        logger.error(f"Pretrained file not found at {pretrained_path}. Skipping load_pretrained.")
        return

    try:
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False) # Cho phép load object
    except Exception as e:
        logger.error(f"Failed to load pretrained file from {pretrained_path}: {e}")
        return

    state_dict_to_load = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    if not isinstance(state_dict_to_load, dict):
        logger.error(f"Pretrained file {pretrained_path} does not contain a 'model' or 'state_dict' key, or it's not a dictionary.")
        return

    # Xử lý prefix 'module.'
    new_sd = {}
    has_module_prefix = all(k.startswith('module.') for k in state_dict_to_load.keys())
    is_model_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)

    if has_module_prefix and not is_model_ddp: # Ckpt DDP, model non-DDP
        logger.info("Pretrained checkpoint is from DDP, removing 'module.' prefix for non-DDP model.")
        for k, v in state_dict_to_load.items():
            new_sd[k[7:]] = v
        state_dict_to_load = new_sd
    elif not has_module_prefix and is_model_ddp: # Ckpt non-DDP, model DDP
        logger.info("Pretrained checkpoint is non-DDP, adding 'module.' prefix for DDP model.")
        for k, v in state_dict_to_load.items():
            new_sd['module.' + k] = v
        state_dict_to_load = new_sd
    
    model_dict = model.state_dict()
    
    # Chỉ giữ lại các key có trong model hiện tại và khớp kích thước
    pretrained_dict_filtered = {
        k: v for k, v in state_dict_to_load.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }

    loaded_keys = set(pretrained_dict_filtered.keys())
    model_keys = set(model_dict.keys())
    missing_in_model = loaded_keys - model_keys # Keys trong ckpt nhưng không có trong model (ít xảy ra nếu lọc đúng)
    missing_in_ckpt = model_keys - loaded_keys  # Keys trong model nhưng không có trong ckpt (quan trọng, ví dụ head)

    if len(pretrained_dict_filtered) == 0:
        logger.warning("No weights were loaded from pretrained checkpoint after filtering. Check keys and shapes.")
    
    model_dict.update(pretrained_dict_filtered)
    msg = model.load_state_dict(model_dict, strict=False)
    
    if msg.missing_keys: logger.warning(f"During load_pretrained, missing keys in model's state_dict: {msg.missing_keys}")
    if msg.unexpected_keys: logger.warning(f"During load_pretrained, unexpected keys in checkpoint's state_dict: {msg.unexpected_keys}")
    
    # In ra các key không được load từ checkpoint do khác shape hoặc không tồn tại trong model
    not_loaded_from_ckpt = set(state_dict_to_load.keys()) - loaded_keys
    if not_loaded_from_ckpt:
        logger.warning(f"Weights from checkpoint not loaded into model (due to name/shape mismatch or not in model): {sorted(list(not_loaded_from_ckpt))}")

    # In ra các key của model không được cập nhật từ checkpoint
    if missing_in_ckpt:
        logger.warning(f"Model keys not found in (or not loaded from) pretrained checkpoint: {sorted(list(missing_in_ckpt))}")

    logger.info(f"=> Successfully processed pretrained weights from '{pretrained_path}'")
    del checkpoint, state_dict_to_load, pretrained_dict_filtered, model_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_checkpoint(config, model, optimizer, lr_scheduler, logger): # 5 tham số
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if not os.path.isfile(config.MODEL.RESUME):
        logger.error(f"Checkpoint file not found at '{config.MODEL.RESUME}'. Cannot resume.")
        return 0.0, config.TRAIN.START_EPOCH # Trả về max_acc và start_epoch hiện tại

    logger.info(f"Attempting to load checkpoint with weights_only=False. Path: {config.MODEL.RESUME}")
    try:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu', weights_only=False)
        logger.info("Successfully loaded checkpoint file with weights_only=False.")
    except pickle.UnpicklingError as e_pickle:
        logger.error(f"Pickle UnpicklingError during torch.load for '{config.MODEL.RESUME}': {e_pickle}")
        raise
    except Exception as e:
        logger.error(f"Failed to load checkpoint (path: {config.MODEL.RESUME}): {e}")
        return 0.0, config.TRAIN.START_EPOCH

    if 'model' not in checkpoint:
        logger.error("'model' key not found in checkpoint. Cannot load model weights.")
        return 0.0, config.TRAIN.START_EPOCH
        
    state_dict = checkpoint['model']
    new_state_dict = {}
    is_model_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    is_ckpt_ddp = all(k.startswith('module.') for k in state_dict.keys())

    if is_ckpt_ddp and not is_model_ddp:
        logger.info("Checkpoint is DDP, model is not. Removing 'module.' prefix.")
        for k, v in state_dict.items(): new_state_dict[k[7:]] = v
        model_sd_to_load = new_state_dict
    elif not is_ckpt_ddp and is_model_ddp:
        logger.info("Checkpoint is not DDP, model is DDP. Adding 'module.' prefix.")
        for k, v in state_dict.items(): new_state_dict['module.' + k] = v
        model_sd_to_load = new_state_dict
    else:
        model_sd_to_load = state_dict

    load_strict = True
    ckpt_num_classes = -1
    # Xác định số lớp của head trong checkpoint một cách an toàn hơn
    head_key_prefix = None
    if 'head.weight' in model_sd_to_load: head_key_prefix = 'head.'
    elif 'fc.weight' in model_sd_to_load: head_key_prefix = 'fc.'
    # Thêm các prefix head khác nếu model có thể có (ví dụ 'classifier.')

    if head_key_prefix and f'{head_key_prefix}weight' in model_sd_to_load:
        ckpt_num_classes = model_sd_to_load[f'{head_key_prefix}weight'].shape[0]

    if hasattr(config.MODEL, 'NUM_CLASSES') and ckpt_num_classes != -1 and config.MODEL.NUM_CLASSES != ckpt_num_classes:
        logger.warning(f"NUM_CLASSES MISMATCH! Checkpoint head has {ckpt_num_classes} classes, "
                       f"current config.MODEL.NUM_CLASSES is {config.MODEL.NUM_CLASSES}. "
                       "Will load with strict=False and head weights will likely be skipped.")
        keys_to_remove = [k for k in model_sd_to_load if head_key_prefix and k.startswith(head_key_prefix)]
        if keys_to_remove:
            logger.info(f"Removing head keys from checkpoint state_dict: {keys_to_remove}")
            for k_rem in keys_to_remove: del model_sd_to_load[k_rem]
        load_strict = False
    
    msg = model.load_state_dict(model_sd_to_load, strict=load_strict)
    logger.info(f"Model state_dict loaded. Strict: {load_strict}. Message: {msg}")
    if msg.missing_keys: logger.warning(f"Missing keys in model when loading checkpoint: {msg.missing_keys}")
    if msg.unexpected_keys: logger.warning(f"Unexpected keys in checkpoint when loading model: {msg.unexpected_keys}")

    max_accuracy = 0.0
    start_epoch_from_ckpt = config.TRAIN.START_EPOCH # Giữ nguyên nếu không có trong checkpoint

    if not config.EVAL_MODE:
        if 'optimizer' in checkpoint and optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                logger.info("Optimizer state_dict loaded.")
            except Exception as e: logger.warning(f"Could not load optimizer state_dict: {e}.")
        else: logger.warning("Optimizer state_dict not found in checkpoint or optimizer is None.")

        if 'lr_scheduler' in checkpoint and lr_scheduler is not None:
            try:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                logger.info("LR scheduler state_dict loaded.")
            except Exception as e: logger.warning(f"Could not load LR scheduler state_dict: {e}.")
        else: logger.warning("LR scheduler state_dict not found or lr_scheduler is None.")

        if 'epoch' in checkpoint:
            completed_epoch = checkpoint['epoch']
            # Cập nhật config.TRAIN.START_EPOCH trực tiếp
            config.defrost()
            config.TRAIN.START_EPOCH = completed_epoch + 1
            config.freeze()
            start_epoch_from_ckpt = config.TRAIN.START_EPOCH
            logger.info(f"Training will resume from epoch {start_epoch_from_ckpt} (completed epoch {completed_epoch}).")
        else:
            logger.warning(f"Epoch not found in checkpoint. Training will start from config.TRAIN.START_EPOCH (currently {config.TRAIN.START_EPOCH}).")
            start_epoch_from_ckpt = config.TRAIN.START_EPOCH


    if 'max_accuracy' in checkpoint:
        max_accuracy = checkpoint['max_accuracy']
        logger.info(f"Max accuracy loaded from checkpoint: {max_accuracy:.2f}%")
    
    logger.info(f"=> Loaded checkpoint '{config.MODEL.RESUME}' (saved at epoch {checkpoint.get('epoch', 'N/A')}) successfully.")
    
    del checkpoint
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # Hàm này trong main.py hiện tại gán giá trị trả về cho max_accuracy
    # và không dùng start_epoch trả về. config.TRAIN.START_EPOCH đã được cập nhật.
    return max_accuracy


def save_checkpoint_new(config, epoch, model_without_ddp, acc1, optimizer, lr_scheduler, logger, name=None):
    save_state = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'max_accuracy': acc1, 
        'epoch': epoch, # epoch vừa hoàn thành
        'config_dump_str': config.dump() # Lưu config hiện tại dưới dạng chuỗi YAML
    }

    if name:
        save_path = os.path.join(config.OUTPUT, f'{name}.pth')
    else:
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')

    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2.0): # Đảm bảo norm_type là float
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    if not parameters: # Nếu không có tham số nào có grad
        return torch.tensor(0.) # Trả về tensor 0
    norm_type = float(norm_type)
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type)
    return total_norm.item()


def auto_resume_helper(output_dir):
    # Tạo logger tạm thời nếu cần, hoặc truyền logger vào
    # print(f"Auto-resuming: Checking directory {output_dir}")
    if not os.path.isdir(output_dir):
        # print(f"Auto-resuming: Output directory {output_dir} does not exist. No checkpoint to resume.")
        return None
        
    checkpoints = [f for f in os.listdir(output_dir) if f.endswith('pth') and f.startswith('ckpt_epoch_')]
    # print(f"Auto-resuming: Found pth files: {checkpoints}")
    if checkpoints:
        # Sắp xếp theo số epoch (ví dụ: ckpt_epoch_10.pth > ckpt_epoch_9.pth)
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_checkpoint = os.path.join(output_dir, checkpoints[-1])
        # print(f"Auto-resuming: Latest checkpoint is {latest_checkpoint}")
        return latest_checkpoint
    # print(f"Auto-resuming: No checkpoint files starting with 'ckpt_epoch_' found in {output_dir}.")
    return None

# Các hàm khác nếu có trong utils.py của bạn...
