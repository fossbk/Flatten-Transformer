# data/build.py (Version 5)
# Ghi chú fix lỗi & cải tiến (so với các phiên bản trước đó trong quá trình gỡ lỗi):
# - v1 (Gốc): Lỗi ImportError: cannot import name '_pil_interp'.
# - v2 (Sửa lỗi _pil_interp): Thêm hàm get_pil_interp từ PIL.Image và thay thế các lời gọi _pil_interp.
# - v3 (Hỗ trợ single-GPU/CPU cho build_loader):
#   - build_loader nhận tham số is_distributed.
#   - Chọn Sampler (Random/Sequential vs Distributed/SubsetRandom) dựa trên is_distributed và num_tasks.
#   - Xử lý current_rank, num_tasks linh hoạt hơn.
#   - Cải thiện logic cập nhật config.MODEL.NUM_CLASSES.
# - v4 (Thêm debug cho FileNotFoundError): Thêm các lệnh print chi tiết trong build_dataset
#   để kiểm tra config.DATA.DATA_PATH, prefix, root và nội dung thư mục.
#   Ném lại FileNotFoundError từ ImageFolder để dừng sớm nếu đường dẫn không hợp lệ.
#   Cải thiện logic suy ra nb_classes và xử lý NUM_CLASSES từ config.
# - v5 (Phiên bản này - dựa trên các bản vá đã chạy được):
#   - Giữ lại các bản vá từ v4.
#   - Đảm bảo các print debug vẫn hữu ích.
#   - Tinh chỉnh nhỏ để rõ ràng hơn.

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from PIL import Image # Đảm bảo đã import Image

# Các import cục bộ từ project
from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler

# HÀM LẤY PHƯƠNG THỨC NỘI SUY TỪ PILLOW (đã ổn định từ các phiên bản trước)
def get_pil_interp(method_str: str):
    method_str = method_str.lower()
    if hasattr(Image, 'Resampling'):
        if method_str == 'bilinear': return Image.Resampling.BILINEAR
        elif method_str == 'bicubic': return Image.Resampling.BICUBIC
        elif method_str == 'lanczos': return Image.Resampling.LANCZOS
        elif method_str == 'nearest': return Image.Resampling.NEAREST
        elif method_str == 'box': return Image.Resampling.BOX
        elif method_str == 'hamming': return Image.Resampling.HAMMING
        else:
            print(f"Warning (get_pil_interp): Interpolation method '{method_str}' not explicitly handled, defaulting to BICUBIC (Pillow >= 9.0.0).")
            return Image.Resampling.BICUBIC
    else:
        if method_str == 'bilinear': return Image.BILINEAR
        elif method_str == 'bicubic': return Image.BICUBIC
        elif method_str == 'lanczos': return Image.LANCZOS
        elif method_str == 'nearest': return Image.NEAREST
        else:
            print(f"Warning (get_pil_interp): Interpolation method '{method_str}' not explicitly handled (Pillow < 9.0.0), defaulting to BICUBIC.")
            return Image.BICUBIC

# HÀM BUILD_LOADER ĐÃ ĐƯỢC CẬP NHẬT
def build_loader(config, is_distributed):
    config.defrost()
    dataset_train, num_classes_from_build_dataset = build_dataset(is_train=True, config=config)

    if not hasattr(config.MODEL, 'NUM_CLASSES') or \
       config.MODEL.NUM_CLASSES == 0 or \
       (hasattr(config.MODEL, 'NUM_CLASSES') and config.MODEL.NUM_CLASSES != num_classes_from_build_dataset and num_classes_from_build_dataset != 0) :
        if num_classes_from_build_dataset != 0:
             print(f"build_loader: Updating config.MODEL.NUM_CLASSES from {getattr(config.MODEL, 'NUM_CLASSES', 'Not Set')} to {num_classes_from_build_dataset}")
             config.MODEL.NUM_CLASSES = num_classes_from_build_dataset
        elif getattr(config.MODEL, 'NUM_CLASSES', 0) == 0 and num_classes_from_build_dataset == 0 :
             # Nên raise lỗi ở build_dataset nếu nb_classes = 0
             print(f"ERROR: build_loader - Could not determine NUM_CLASSES. This indicates an issue with dataset loading or configuration.")
             raise ValueError("NUM_CLASSES could not be determined in build_loader.")
    config.freeze()

    current_rank = config.RANK # Lấy từ config (đã được main.py cập nhật)
    num_tasks = config.WORLD_SIZE # Lấy từ config

    print(f"build_loader: local_rank={config.LOCAL_RANK}, global_rank={current_rank}, world_size={num_tasks}, is_distributed={is_distributed}")
    print(f"build_loader: Successfully built train dataset (size: {len(dataset_train)}) with {config.MODEL.NUM_CLASSES} classes.")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"build_loader: Successfully built val dataset (size: {len(dataset_val)}) with {config.MODEL.NUM_CLASSES} classes.")


    if is_distributed and num_tasks > 1 :
        if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
            train_indices = np.arange(current_rank, len(dataset_train), num_tasks)
            sampler_train = SubsetRandomSampler(train_indices)
            val_indices = np.arange(current_rank, len(dataset_val), num_tasks)
            sampler_val = SubsetRandomSampler(val_indices)
            print("build_loader: Using SubsetRandomSampler for DDP (ZIP_MODE and CACHE_MODE='part')")
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=current_rank, shuffle=True, seed=config.SEED
            )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=current_rank, shuffle=config.TEST.SHUFFLE, seed=config.SEED
            )
            print(f"build_loader: Using DistributedSampler for DDP. Train shuffle: True, Val shuffle: {config.TEST.SHUFFLE}")
    else:
        g = torch.Generator()
        g.manual_seed(config.SEED)
        sampler_train = torch.utils.data.RandomSampler(dataset_train, generator=g)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        print("build_loader: Using RandomSampler (train) and SequentialSampler (val) for non-distributed mode.")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        if config.MODEL.NUM_CLASSES == 0:
            print("ERROR: build_loader - config.MODEL.NUM_CLASSES is 0 when Mixup is active. Cannot create Mixup function.")
        else:
            mixup_fn = Mixup(
                mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
                prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
                label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
            print(f"build_loader: Mixup function created with num_classes={config.MODEL.NUM_CLASSES}")

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    nb_classes = 0

    if config.DATA.DATASET.lower() == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            zip_prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, zip_prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
            if hasattr(dataset, 'get_len'): # Dummy check, CachedImageFolder cần cách xác định số lớp
                pass
            if hasattr(config.MODEL, 'NUM_CLASSES') and config.MODEL.NUM_CLASSES != 0:
                nb_classes = config.MODEL.NUM_CLASSES
            else:
                nb_classes = 1000 # Mặc định cho ImageNet
                print(f"Warning: build_dataset (ZIP_MODE) - Assuming {nb_classes} classes. Set config.MODEL.NUM_CLASSES if different.")
        else: # Không phải ZIP_MODE
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            # --- PHẦN DEBUG CHO FileNotFoundError ---
            print(f"\n[DEBUG] build_dataset: is_train={is_train}")
            print(f"[DEBUG] build_dataset: config.DATA.DATA_PATH='{config.DATA.DATA_PATH}'")
            print(f"[DEBUG] build_dataset: prefix='{prefix}'")
            print(f"[DEBUG] build_dataset: Calculated root for ImageFolder='{root}'")

            if not os.path.exists(root):
                print(f"[DEBUG] ERROR: Root directory '{root}' DOES NOT EXIST.")
                raise FileNotFoundError(f"Calculated root directory '{root}' does not exist. Check --data-path ('{config.DATA.DATA_PATH}') and ensure it contains '{prefix}' subdirectory.")
            else:
                print(f"[DEBUG] Root directory '{root}' EXISTS.")
                print(f"[DEBUG] Contents of '{root}' (first few entries):")
                try:
                    listed_items = os.listdir(root)
                    for item_idx, item_name in enumerate(listed_items):
                        if item_idx >= 5: break # Chỉ in 5 item đầu
                        item_path_debug = os.path.join(root, item_name)
                        is_dir_item_debug = os.path.isdir(item_path_debug)
                        print(f"  - '{item_name}' (is_dir: {is_dir_item_debug})")
                    if not any(os.path.isdir(os.path.join(root, item)) for item in listed_items):
                         print(f"[DEBUG] WARNING: No subdirectories (potential classes) found in '{root}'. ImageFolder might fail or infer 0 classes.")
                except Exception as e:
                    print(f"[DEBUG] Error listing contents of '{root}': {e}")
            # --- KẾT THÚC PHẦN DEBUG ---
            try:
                dataset = datasets.ImageFolder(root, transform=transform)
                if hasattr(dataset, 'classes') and dataset.classes: # Kiểm tra dataset.classes không rỗng
                    nb_classes = len(dataset.classes)
                    print(f"build_dataset: Inferred {nb_classes} classes from ImageFolder at '{root}'")
                else:
                    nb_classes = 0 # Đặt là 0 nếu không suy ra được
                    print(f"Warning: build_dataset - Could not infer classes from ImageFolder at '{root}'. dataset.classes might be empty or missing.")
            except FileNotFoundError as e:
                print(f"ERROR in datasets.ImageFolder when accessing '{root}': {e}")
                raise
            except Exception as e_img_folder: # Bắt các lỗi khác từ ImageFolder
                print(f"ERROR during datasets.ImageFolder initialization for '{root}': {e_img_folder}")
                raise

        # Ghi đè nb_classes nếu config.MODEL.NUM_CLASSES được đặt và hợp lệ
        if hasattr(config.MODEL, 'NUM_CLASSES') and config.MODEL.NUM_CLASSES > 0 and \
           (nb_classes == 0 or config.MODEL.NUM_CLASSES != nb_classes) : # Ghi đè nếu nb_classes là 0 hoặc khác
            print(f"Info: build_dataset - Overriding nb_classes for {config.DATA.DATASET} from {nb_classes if nb_classes > 0 else 'undetermined'} to {config.MODEL.NUM_CLASSES} (from config)")
            nb_classes = config.MODEL.NUM_CLASSES
        elif nb_classes == 0 and hasattr(config.MODEL, 'NUM_CLASSES') and config.MODEL.NUM_CLASSES > 0:
             nb_classes = config.MODEL.NUM_CLASSES
             print(f"Info: build_dataset - Using NUM_CLASSES={nb_classes} from config as it could not be inferred from dataset.")

    # Thêm các dataset khác ở đây nếu cần
    # elif config.DATA.DATASET.lower() == 'cifar10':
    #     dataset = datasets.CIFAR10(root=config.DATA.DATA_PATH, train=is_train, download=True, transform=transform)
    #     nb_classes = 10
    #     if hasattr(config.MODEL, 'NUM_CLASSES') and config.MODEL.NUM_CLASSES != 0 and config.MODEL.NUM_CLASSES != nb_classes:
    #         nb_classes = config.MODEL.NUM_CLASSES
    else:
        raise NotImplementedError(f"Dataset '{config.DATA.DATASET}' not implemented in build_dataset.")

    if nb_classes == 0:
        raise ValueError(f"Error: build_dataset - Number of classes (nb_classes) resolved to 0 for dataset '{config.DATA.DATASET}'. This is usually due to an incorrect DATA_PATH or an empty dataset directory. Please check the path: '{config.DATA.DATA_PATH}' and its subdirectories '{prefix}'.")
        
    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    interpolation_method_str = config.DATA.INTERPOLATION

    if is_train:
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT.lower() != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=interpolation_method_str,
        )
        if not resize_im:
            if hasattr(transform, 'transforms') and len(transform.transforms) > 0:
                 print(f"Modifying train transform for small image size ({config.DATA.IMG_SIZE}x{config.DATA.IMG_SIZE})")
                 if isinstance(transform.transforms[0], transforms.RandomResizedCrop):
                    transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
                 else:
                    print(f"Warning (build_transform): Expected RandomResizedCrop at T.transforms[0], but found {type(transform.transforms[0])}. Not replacing.")
            else:
                 print(f"Warning (build_transform): Could not modify train T for small image size. T.transforms list might be empty or not as expected.")
        print(f"Train transforms: {transform}")
        return transform

    # Build validation/testing transform
    t = []
    pil_interpolation_mode = get_pil_interp(interpolation_method_str)

    if resize_im:
        crop_pct = config.DATA.CROP_PCT if hasattr(config.DATA, 'CROP_PCT') and config.DATA.CROP_PCT > 0 else 0.875
        
        if config.TEST.CROP:
            import math
            scale_size = int(math.floor(config.DATA.IMG_SIZE / crop_pct))
            t.append(transforms.Resize(scale_size, interpolation=pil_interpolation_mode))
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
            print(f"Validation transform: Resize shortest to {scale_size} (interp: {interpolation_method_str}), then CenterCrop to {config.DATA.IMG_SIZE}")
        else:
            t.append(transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE), interpolation=pil_interpolation_mode))
            print(f"Validation transform: Resize to ({config.DATA.IMG_SIZE},{config.DATA.IMG_SIZE}) (interp: {interpolation_method_str})")

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    print(f"Validation/Test transforms: {transforms.Compose(t)}")
    return transforms.Compose(t)
