# File: /content/drive/MyDrive/00 AI/MultiView_SLR/FLatten-Transformer/data/build.py
# Phiên bản đầy đủ, đã cập nhật để vá lỗi _pil_interp và hỗ trợ single GPU/CPU

import os
import torch
import numpy as np
import torch.distributed as dist # Sẽ chỉ được sử dụng nếu is_distributed là True
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform # create_transform của timm sẽ tự xử lý nhiều logic augmentation
# from timm.data.transforms import _pil_interp # Dòng import gốc gây lỗi - ĐÃ BÌNH LUẬN/XÓA

# THÊM IMPORT TỪ PILLOW
from PIL import Image

# Các import cục bộ từ project (đảm bảo các file này tồn tại và đúng đường dẫn)
from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler

# HÀM MỚI ĐỂ LẤY PHƯƠNG THỨC NỘI SUY TỪ PILLOW
def get_pil_interp(method_str: str):
    """
    Gets PIL interpolation mode from a string.
    Handles compatibility with newer Pillow versions (>= 9.0.0) using Image.Resampling.
    """
    method_str = method_str.lower() # Chuyển về chữ thường để dễ so sánh
    if hasattr(Image, 'Resampling'):  # Dùng cho Pillow >= 9.0.0
        if method_str == 'bilinear':
            return Image.Resampling.BILINEAR
        elif method_str == 'bicubic':
            return Image.Resampling.BICUBIC
        elif method_str == 'lanczos':
            return Image.Resampling.LANCZOS
        elif method_str == 'nearest':
            return Image.Resampling.NEAREST
        elif method_str == 'box':
            return Image.Resampling.BOX
        elif method_str == 'hamming':
            return Image.Resampling.HAMMING
        else:
            print(f"Warning (get_pil_interp): Interpolation method '{method_str}' not explicitly handled, defaulting to BICUBIC (Pillow >= 9.0.0).")
            return Image.Resampling.BICUBIC
    else:  # Dùng cho Pillow < 9.0.0
        if method_str == 'bilinear':
            return Image.BILINEAR
        elif method_str == 'bicubic':
            return Image.BICUBIC
        elif method_str == 'lanczos':
            return Image.LANCZOS
        elif method_str == 'nearest':
            return Image.NEAREST
        # 'box' và 'hamming' có thể không có hoặc tên khác ở các phiên bản Pillow rất cũ
        # Ví dụ, Image.BOX có từ Pillow 2.7.0. Image.HAMMING từ Pillow 2.7.0.
        else:
            print(f"Warning (get_pil_interp): Interpolation method '{method_str}' not explicitly handled (Pillow < 9.0.0), defaulting to BICUBIC.")
            return Image.BICUBIC

# HÀM BUILD_LOADER ĐÃ ĐƯỢC CẬP NHẬT
def build_loader(config, is_distributed): # Nhận is_distributed từ main.py
    config.defrost()
    dataset_train, num_classes_from_build_dataset = build_dataset(is_train=True, config=config)

    # Cập nhật NUM_CLASSES trong config nếu cần thiết.
    if not hasattr(config.MODEL, 'NUM_CLASSES') or \
       config.MODEL.NUM_CLASSES == 0 or \
       (hasattr(config.MODEL, 'NUM_CLASSES') and config.MODEL.NUM_CLASSES != num_classes_from_build_dataset and num_classes_from_build_dataset != 0):
        if num_classes_from_build_dataset != 0:
             print(f"build_loader: Updating config.MODEL.NUM_CLASSES from {getattr(config.MODEL, 'NUM_CLASSES', 'Not Set')} to {num_classes_from_build_dataset}")
             config.MODEL.NUM_CLASSES = num_classes_from_build_dataset
        elif getattr(config.MODEL, 'NUM_CLASSES', 0) == 0 and num_classes_from_build_dataset == 0 :
             print(f"ERROR: build_loader - Could not determine NUM_CLASSES from dataset and config.MODEL.NUM_CLASSES is also {getattr(config.MODEL, 'NUM_CLASSES', 'Not Set')}. This will cause issues.")
             # Có thể raise lỗi ở đây để dừng sớm nếu số lớp không xác định được
             raise ValueError("NUM_CLASSES could not be determined. Please check dataset or config.")
    config.freeze()

    current_rank = 0
    num_tasks = 1 # Mặc định cho single GPU/CPU
    if is_distributed:
        try:
            if dist.is_initialized(): # Chỉ gọi nếu dist đã được init
                current_rank = dist.get_rank()
                num_tasks = dist.get_world_size()
            elif 'RANK' in os.environ: # Fallback nếu launch tool đã set env var nhưng dist chưa init trong build_loader
                current_rank = int(os.environ['RANK'])
                num_tasks = int(os.environ['WORLD_SIZE'])
                print("build_loader: Using RANK/WORLD_SIZE from env vars as torch.distributed not yet initialized here.")
            else: # Vẫn coi như single GPU nếu không có thông tin gì
                is_distributed = False
                print("build_loader: is_distributed was True, but torch.distributed not initialized and no RANK env var. Treating as non-distributed.")
        except Exception as e:
            print(f"build_loader: Error getting distributed info (is_distributed={is_distributed}): {e}. Defaulting to non-distributed.")
            is_distributed = False # Fallback an toàn
            current_rank = 0
            num_tasks = 1
    # Gán lại vào config nếu main.py chưa làm
    if not hasattr(config, 'LOCAL_RANK') or config.LOCAL_RANK == -1 and is_distributed :
        config.defrost()
        config.LOCAL_RANK = current_rank # Giả định local_rank = global_rank trong single-node
        config.freeze()

    print(f"build_loader: local_rank={config.LOCAL_RANK}, global_rank={current_rank}, world_size={num_tasks}, is_distributed={is_distributed}")
    print(f"build_loader: Successfully built train dataset (size: {len(dataset_train)})")
    dataset_val, _ = build_dataset(is_train=False, config=config) # NUM_CLASSES chỉ cần lấy 1 lần
    print(f"build_loader: Successfully built val dataset (size: {len(dataset_val)})")

    if is_distributed and num_tasks > 1:
        if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
            # Logic sampler gốc của repo cho trường hợp ZIP_MODE và CACHE_MODE='part' khi phân tán
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
    else: # Chế độ không phân tán (single GPU hoặc CPU)
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
        shuffle=False, # Sampler đã xử lý shuffle; cho val thường là False
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        if config.MODEL.NUM_CLASSES == 0:
            print("ERROR: build_loader - config.MODEL.NUM_CLASSES is 0 when Mixup is active. Cannot create Mixup function.")
            # Hoặc raise ValueError("NUM_CLASSES must be set for Mixup.")
        else:
            mixup_fn = Mixup(
                mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
                prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
                label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
            print(f"build_loader: Mixup function created with num_classes={config.MODEL.NUM_CLASSES}")

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    nb_classes = 0 # Khởi tạo

    if config.DATA.DATASET.lower() == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            zip_prefix = prefix + ".zip@/" # Đổi tên biến để tránh nhầm lẫn
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, zip_prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
            # CachedImageFolder có thể cần cách khác để lấy số lớp, hoặc dựa vào config
            if hasattr(dataset, 'get_len'): # Một cách suy đoán
                 pass # Số lớp có thể được đọc từ ann_file hoặc tương tự
            if config.MODEL.NUM_CLASSES != 0: # Ưu tiên config nếu có
                nb_classes = config.MODEL.NUM_CLASSES
            else: # Mặc định nếu không có gì khác
                nb_classes = 1000
                print(f"Warning: build_dataset (ZIP_MODE) - Assuming {nb_classes} classes for ImageNet. Override with config.MODEL.NUM_CLASSES if needed.")

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
                print(f"[DEBUG] Contents of '{root}' (first few):")
                try:
                    listed_items = os.listdir(root)
                    for item in listed_items[:5]: # In 5 item đầu
                        item_path = os.path.join(root, item)
                        is_dir_item = os.path.isdir(item_path)
                        print(f"  - '{item}' (is_dir: {is_dir_item})")
                        if is_dir_item and item in ["n01981276", "n04429376"]: # Các lớp từ lỗi trước
                             print(f"    [DEBUG] Contents of class '{item}':")
                             try:
                                 class_contents = os.listdir(item_path)
                                 print(f"      - Files (first few): {class_contents[:5]}")
                                 if not class_contents: print(f"      - WARNING: Class directory '{item}' is empty.")
                             except Exception as e_inner:
                                 print(f"      - Error listing contents of class '{item}': {e_inner}")

                    if not any(os.path.isdir(os.path.join(root, item)) for item in listed_items):
                         print(f"[DEBUG] WARNING: No subdirectories (potential classes) found directly in '{root}'. ImageFolder might fail.")
                except Exception as e:
                    print(f"[DEBUG] Error listing contents of '{root}': {e}")
            # --- KẾT THÚC PHẦN DEBUG ---
            try:
                dataset = datasets.ImageFolder(root, transform=transform)
                if hasattr(dataset, 'classes'):
                    nb_classes = len(dataset.classes)
                    print(f"build_dataset: Inferred {nb_classes} classes from ImageFolder at '{root}'")
                else: # Fallback nếu không lấy được từ dataset.classes
                    nb_classes = 1000
                    print(f"Warning: build_dataset - Could not infer num_classes from dataset.classes. Assuming {nb_classes} for ImageNet.")
            except FileNotFoundError as e: # Bắt lỗi FileNotFoundError từ ImageFolder
                print(f"ERROR in datasets.ImageFolder: {e}")
                print("Please check if the 'root' path for ImageFolder is correct and contains subdirectories for classes, and those subdirectories contain valid image files.")
                raise e # Ném lại lỗi để dừng chương trình

        # Ghi đè nb_classes nếu config.MODEL.NUM_CLASSES được đặt và khác với giá trị suy ra/mặc định
        if hasattr(config.MODEL, 'NUM_CLASSES') and config.MODEL.NUM_CLASSES != 0 and config.MODEL.NUM_CLASSES != nb_classes:
            print(f"Info: build_dataset - Overriding nb_classes for {config.DATA.DATASET} from {nb_classes} to {config.MODEL.NUM_CLASSES}")
            nb_classes = config.MODEL.NUM_CLASSES
        elif nb_classes == 0 and hasattr(config.MODEL, 'NUM_CLASSES') and config.MODEL.NUM_CLASSES != 0:
             nb_classes = config.MODEL.NUM_CLASSES
             print(f"Info: build_dataset - Using NUM_CLASSES={nb_classes} from config as it was 0 from dataset inference.")


    # Ví dụ thêm xử lý cho các bộ dữ liệu khác nếu cần
    # elif config.DATA.DATASET.lower() == 'cifar10':
    #     dataset = datasets.CIFAR10(root=config.DATA.DATA_PATH, train=is_train, download=True, transform=transform)
    #     nb_classes = 10
    #     if hasattr(config.MODEL, 'NUM_CLASSES') and config.MODEL.NUM_CLASSES != 0 and config.MODEL.NUM_CLASSES != nb_classes:
    #         nb_classes = config.MODEL.NUM_CLASSES
    else:
        raise NotImplementedError(f"Dataset {config.DATA.DATASET} not supported or NUM_CLASSES cannot be determined.")

    if nb_classes == 0:
        raise ValueError(f"Error: build_dataset - Number of classes (nb_classes) is 0 for dataset '{config.DATA.DATASET}'. Cannot proceed.")
        
    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    interpolation_method_str = config.DATA.INTERPOLATION # Chuỗi từ config (ví dụ: 'bicubic')

    if is_train:
        # create_transform của timm đã xử lý việc chuyển chuỗi interpolation thành hằng số PIL
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT.lower() != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=interpolation_method_str, # Truyền chuỗi vào đây
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
        return transform

    # Build validation/testing transform
    t = []
    # Lấy hằng số PIL từ hàm get_pil_interp đã định nghĩa
    pil_interpolation_mode = get_pil_interp(interpolation_method_str)

    if resize_im:
        crop_pct = getattr(config.DATA, 'CROP_PCT', 0.875) # Mặc định thường là 0.875 hoặc 0.9
        if config.TEST.CROP: # Nếu config.TEST.CROP là True
            import math
            # scale_size tính theo crop_pct là chuẩn của timm
            scale_size = int(math.floor(config.DATA.IMG_SIZE / crop_pct))
            t.append(
                transforms.Resize(scale_size, interpolation=pil_interpolation_mode),
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
            print(f"Validation transform: Resize shortest to {scale_size} (interp: {interpolation_method_str}), then CenterCrop to {config.DATA.IMG_SIZE}")
        else: # Nếu không crop, chỉ resize về đúng IMG_SIZE
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=pil_interpolation_mode)
            )
            print(f"Validation transform: Resize to ({config.DATA.IMG_SIZE},{config.DATA.IMG_SIZE}) (interp: {interpolation_method_str})")

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
