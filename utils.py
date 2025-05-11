{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # File: /content/drive/MyDrive/00 AI/MultiView_SLR/FLatten-Transformer/utils.py\
import os\
import torch\
import torch.distributed as dist\
# import pickle # Kh\'f4ng c\uc0\u7847 n thi\u7871 t\
# from yacs.config import CfgNode as CN # Kh\'f4ng c\uc0\u7847 n thi\u7871 t n\u7871 u kh\'f4ng l\u432 u/load CfgNode tr\u7921 c ti\u7871 p\
\
# --- C\'c1C H\'c0M KH\'c1C (get_grad_norm, auto_resume_helper, reduce_tensor, load_pretrained) ---\
# --- GI\uc0\u7918  NGUY\'caN NH\u431  TRONG REPOSITORY G\u7888 C HO\u7862 C \u272 \u7842 M B\u7842 O CH\'daNG T\u431 \u416 NG TH\'cdCH ---\
\
# V\'ed d\uc0\u7909 :\
def get_grad_norm(parameters, norm_type=2):\
    if isinstance(parameters, torch.Tensor):\
        parameters = [parameters]\
    parameters = list(filter(lambda p: p.grad is not None, parameters))\
    norm_type = float(norm_type)\
    if len(parameters) == 0:\
        return torch.tensor(0.)\
    device = parameters[0].grad.device\
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)\
    return total_norm\
\
def reduce_tensor(tensor): # H\'e0m n\'e0y c\'f3 th\uc0\u7875  \u273 \u432 \u7907 c thay th\u7871  b\u7857 ng adjusted_reduce_tensor trong main.py\
    rt = tensor.clone()\
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)\
    rt /= dist.get_world_size()\
    return rt\
\
def auto_resume_helper(output_dir):\
    checkpoints = os.listdir(output_dir)\
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]\
    print(f"All checkpoints founded in \{output_dir\}: \{checkpoints\}") # S\uc0\u7917 a l\u7895 i \u273 \'e1nh m\'e1y\
    if not checkpoints:\
        return None\
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.startswith('ckpt_epoch_')] # Ch\uc0\u7881  x\'e9t c\'e1c ckpt epoch\
    if not checkpoints:\
         # N\uc0\u7871 u kh\'f4ng c\'f3 ckpt_epoch_, th\u7917  t\'ecm max_acc.pth\
        if 'max_acc.pth' in os.listdir(output_dir):\
            print(f"Found max_acc.pth in \{output_dir\}")\
            return os.path.join(output_dir, 'max_acc.pth')\
        return None\
\
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0])) # S\uc0\u7855 p x\u7871 p theo s\u7889  epoch\
    latest_checkpoint = os.path.join(output_dir, checkpoints[-1])\
    print(f"Auto-resume picking latest checkpoint: \{latest_checkpoint\}")\
    return latest_checkpoint\
\
# load_pretrained c\'f3 th\uc0\u7875  c\u7847 n config n\u7871 u n\'f3 thay \u273 \u7893 i head c\u7911 a model d\u7921 a tr\'ean num_classes\
def load_pretrained(pretrained_path, model, logger, config=None):\
    logger.info(f"==============> Loading pretrained weights from \{pretrained_path\}")\
    if not os.path.isfile(pretrained_path):\
        logger.error(f"=> No pretrained weights found at '\{pretrained_path\}'")\
        return\
\
    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=True) # \uc0\u431 u ti\'ean True cho pretrained\
\
    if 'model' in checkpoint:\
        state_dict = checkpoint['model']\
    elif 'state_dict' in checkpoint:\
        state_dict = checkpoint['state_dict']\
    else:\
        state_dict = checkpoint # Gi\uc0\u7843  s\u7917  file ch\u7881  ch\u7913 a state_dict\
\
    # X\uc0\u7917  l\'fd prefix 'module.'\
    if all(k.startswith('module.') for k in state_dict.keys()):\
        logger.info("Removing 'module.' prefix from pretrained DDP weights.")\
        state_dict = \{k.replace('module.', ''): v for k, v in state_dict.items()\}\
    \
    # X\uc0\u7917  l\'fd num_classes mismatch cho fine-tuning\
    if config and hasattr(config.MODEL, 'NUM_CLASSES'):\
        current_num_classes = config.MODEL.NUM_CLASSES\
        head_key_weight = 'head.weight' # Ho\uc0\u7863 c t\'ean kh\'e1c t\'f9y model\
        head_key_bias = 'head.bias'\
\
        if head_key_weight in state_dict:\
            ckpt_num_classes = state_dict[head_key_weight].shape[0]\
            if ckpt_num_classes != current_num_classes:\
                logger.warning(f"PRETRAINED: Num classes mismatch. Checkpoint head has \{ckpt_num_classes\} classes, "\
                               f"current model requires \{current_num_classes\}. Removing head weights from pretrained.")\
                keys_to_remove = [k for k in state_dict if k.startswith('head.')]\
                for k_remove in keys_to_remove:\
                    del state_dict[k_remove]\
    \
    msg = model.load_state_dict(state_dict, strict=False)\
    logger.info(f"Pretrained weights loaded. Missing keys: \{msg.missing_keys\}. Unexpected keys: \{msg.unexpected_keys\}")\
    del checkpoint\
    torch.cuda.empty_cache()\
\
\
# --- H\'c0M load_checkpoint \uc0\u272 \'c3 \u272 \u431 \u7906 C S\u7916 A ---\
def load_checkpoint(config, model, optimizer, lr_scheduler, logger):\
    start_epoch_from_ckpt = 0 # M\uc0\u7863 c \u273 \u7883 nh n\u7871 u kh\'f4ng c\'f3 'epoch' trong ckpt\
    max_accuracy_from_ckpt = 0.0 # M\uc0\u7863 c \u273 \u7883 nh\
\
    if not config.MODEL.RESUME or config.MODEL.RESUME == '': # Ki\uc0\u7875 m tra ngay t\u7915  \u273 \u7847 u\
        logger.info("No resume path specified in config.MODEL.RESUME. Skipping checkpoint loading.")\
        return max_accuracy_from_ckpt, start_epoch_from_ckpt\
\
    logger.info(f"==============> Resuming training from checkpoint: \{config.MODEL.RESUME\}")\
    if not os.path.isfile(config.MODEL.RESUME):\
        logger.error(f"=> No checkpoint found at '\{config.MODEL.RESUME\}'")\
        return max_accuracy_from_ckpt, start_epoch_from_ckpt\
\
    try:\
        # *** S\uc0\u7916  D\u7908 NG weights_only=False V\'cc CHECKPOINT N\'c0Y L\'c0 DO B\u7840 N T\u7840 O RA V\'c0 C\'d3 TH\u7874  CH\u7912 A config.dump() (l\'e0 str) ho\u7863 c CfgNode (n\u7871 u l\u432 u c\u361 ) ***\
        logger.info(f"Attempting to load checkpoint with weights_only=False from: \{config.MODEL.RESUME\}")\
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu', weights_only=False)\
        logger.info("Successfully loaded checkpoint file with weights_only=False.")\
    except Exception as e:\
        logger.error(f"Failed to load checkpoint (path: \{config.MODEL.RESUME\}) with weights_only=False: \{e\}")\
        logger.error("The checkpoint file might be corrupted or not a valid PyTorch checkpoint.")\
        return max_accuracy_from_ckpt, start_epoch_from_ckpt\
\
\
    if 'model' not in checkpoint:\
        logger.error("No 'model' key found in checkpoint. Cannot resume model weights.")\
        return max_accuracy_from_ckpt, start_epoch_from_ckpt\
        \
    # Load model state\
    state_dict = checkpoint['model']\
    # X\uc0\u7917  l\'fd prefix 'module.' n\u7871 u checkpoint \u273 \u432 \u7907 c l\u432 u t\u7915  DDP v\'e0 model hi\u7879 n t\u7841 i kh\'f4ng ph\u7843 i DDP\
    is_current_model_ddp = isinstance(model, nn.parallel.DistributedDataParallel)\
    is_ckpt_ddp = all(k.startswith('module.') for k in state_dict.keys())\
\
    if is_ckpt_ddp and not is_current_model_ddp:\
        logger.info("Removing 'module.' prefix from DDP checkpoint for single GPU/CPU model.")\
        state_dict = \{k.replace('module.', ''): v for k, v in state_dict.items()\}\
    elif not is_ckpt_ddp and is_current_model_ddp:\
        logger.info("Adding 'module.' prefix to load single GPU checkpoint into DDP model.")\
        state_dict = \{'module.' + k: v for k, v in state_dict.items()\}\
    \
    # X\uc0\u7917  l\'fd NUM_CLASSES mismatch tr\u432 \u7899 c khi load (quan tr\u7885 ng cho fine-tuning)\
    num_classes_ckpt = 0\
    # Gi\uc0\u7843  s\u7917  head l\'e0 'head.weight' ho\u7863 c 'fc.weight' ho\u7863 c 'classifier.weight'\
    head_keys_to_check = ['head.weight', 'fc.weight', 'classifier.weight', 'classif.weight'] # M\uc0\u7903  r\u7897 ng c\'e1c key c\'f3 th\u7875 \
    actual_head_weight_key = None\
    for h_key in head_keys_to_check:\
        if h_key in state_dict:\
            actual_head_weight_key = h_key\
            break\
            \
    if actual_head_weight_key:\
        num_classes_ckpt = state_dict[actual_head_weight_key].shape[0]\
        if config.MODEL.NUM_CLASSES != num_classes_ckpt:\
            logger.warning(f"NUM_CLASSES MISMATCH: Checkpoint head (\{actual_head_weight_key\}) has \{num_classes_ckpt\} classes, "\
                           f"but current config.MODEL.NUM_CLASSES is \{config.MODEL.NUM_CLASSES\}.")\
            logger.warning("NOT loading weights for the classification head from checkpoint.")\
            # Lo\uc0\u7841 i b\u7887  c\'e1c key c\u7911 a head (v\'ed d\u7909 : head.weight, head.bias)\
            keys_to_remove = [k for k in state_dict if k.startswith(actual_head_weight_key.split('.')[0] + '.')]\
            for k_remove in keys_to_remove:\
                del state_dict[k_remove]\
                logger.info(f"Removed key \{k_remove\} from checkpoint state_dict.")\
    else:\
        logger.warning("Could not find a recognizable classification head in checkpoint to check NUM_CLASSES.")\
\
    msg = model.load_state_dict(state_dict, strict=False)\
    logger.info(f"Model state_dict loaded. Missing keys: \{msg.missing_keys\}. Unexpected keys: \{msg.unexpected_keys\}")\
\
    # Load optimizer, scheduler, epoch, max_accuracy (ch\uc0\u7881  khi kh\'f4ng ph\u7843 i EVAL_MODE)\
    if not config.EVAL_MODE:\
        if 'optimizer' in checkpoint and optimizer is not None:\
            try:\
                optimizer.load_state_dict(checkpoint['optimizer'])\
                logger.info("Optimizer state_dict loaded.")\
            except Exception as e:\
                logger.warning(f"Could not load optimizer state_dict: \{e\}. Optimizer will start from scratch.")\
        else:\
            logger.warning("No 'optimizer' in checkpoint or optimizer is None. Optimizer starts from scratch.")\
\
        if 'lr_scheduler' in checkpoint and lr_scheduler is not None:\
            try:\
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])\
                logger.info("LR scheduler state_dict loaded.")\
            except Exception as e:\
                logger.warning(f"Could not load lr_scheduler state_dict: \{e\}. LR scheduler starts from scratch.")\
        else:\
            logger.warning("No 'lr_scheduler' in checkpoint or lr_scheduler is None. LR scheduler starts from scratch.")\
        \
        if 'epoch' in checkpoint:\
            # checkpoint['epoch'] l\'e0 epoch V\uc0\u7914 A HO\'c0N TH\'c0NH\
            start_epoch_from_ckpt = checkpoint['epoch'] # S\uc0\u7869  \u273 \u432 \u7907 c +1 trong main.py cho v\'f2ng l\u7863 p\
            config.defrost()\
            config.TRAIN.START_EPOCH = checkpoint['epoch'] # L\uc0\u432 u epoch \u273 \'e3 ho\'e0n th\'e0nh\
            config.freeze()\
            logger.info(f"Training will resume from NEXT epoch: \{start_epoch_from_ckpt + 1\}")\
        else:\
            logger.warning("No 'epoch' key in checkpoint. Training will start from epoch defined in config (or 0).")\
        \
        if 'max_accuracy' in checkpoint:\
            max_accuracy_from_ckpt = checkpoint['max_accuracy']\
            logger.info(f"Resuming with max_accuracy from checkpoint: \{max_accuracy_from_ckpt:.2f\}%")\
\
    elif 'max_accuracy' in checkpoint:\
        max_accuracy_from_ckpt = checkpoint['max_accuracy']\
        logger.info(f"Checkpoint max_accuracy (EVAL_MODE): \{max_accuracy_from_ckpt:.2f\}%")\
\
    logger.info(f"=> Checkpoint '\{config.MODEL.RESUME\}' (saved at end of epoch \{checkpoint.get('epoch', 'N/A')\}) loaded successfully.")\
    \
    del checkpoint\
    if torch.cuda.is_available() and config.LOCAL_RANK != -1:\
        torch.cuda.empty_cache()\
        \
    return max_accuracy_from_ckpt, config.TRAIN.START_EPOCH # Tr\uc0\u7843  v\u7873  START_EPOCH t\u7915  config (\u273 \'e3 \u273 \u432 \u7907 c c\u7853 p nh\u7853 t)\
\
\
def save_checkpoint_new(config, epoch_completed, model, current_acc, optimizer, lr_scheduler, logger, name=None):\
    """Saves checkpoint.\
    Args:\
        epoch_completed: The epoch number that has just been completed.\
        current_acc: Accuracy of the model at this epoch (e.g., validation acc)\
    """\
    save_state = \{\
        'model': model.state_dict(),\
        'optimizer': optimizer.state_dict(),\
        'lr_scheduler': lr_scheduler.state_dict(),\
        'max_accuracy': config.MODEL.MAX_ACCURACY if hasattr(config.MODEL, 'MAX_ACCURACY') else current_acc, # L\uc0\u432 u max_accuracy t\u7893 ng th\u7875 \
        'current_accuracy_this_ckpt': current_acc, # L\uc0\u432 u acc c\u7911 a ckpt n\'e0y\
        'epoch': epoch_completed, # L\uc0\u432 u epoch V\u7914 A HO\'c0N TH\'c0NH\
        'config_dump': config.dump() # L\uc0\u432 u config d\u432 \u7899 i d\u7841 ng text \u273 \u7875  tham kh\u7843 o\
    \}\
    if name is None:\
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_\{epoch_completed\}.pth')\
    else:\
        save_path = os.path.join(config.OUTPUT, f'\{name\}.pth')\
    \
    logger.info(f"Saving checkpoint to \{save_path\} (epoch \{epoch_completed\} completed, current_acc \{current_acc:.2f\}%)")\
    torch.save(save_state, save_path)\
    logger.info(f"Checkpoint \{save_path\} saved successfully!")}