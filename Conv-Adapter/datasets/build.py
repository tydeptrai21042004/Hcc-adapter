import os
from collections import Counter

import torch
import torchvision.transforms as T
from torchvision import datasets
from torchvision.datasets import CocoDetection

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD  = (0.229, 0.224, 0.225)


def _img_transforms(args, is_train: bool, *, force_rgb: bool = False, enable_flip: bool = True):
    """
    Common 224px transforms for classification backbones (e.g., ResNet-50).

    - force_rgb: convert 1-ch images (e.g., MNIST) to 3-ch
    - enable_flip: disable for digit datasets where mirroring is undesirable
    """
    size = args.input_size
    crop_ratio = getattr(args, "crop_ratio", 0.875) or 0.875

    if is_train:
        ops = [
            T.RandomResizedCrop(size, scale=(0.6, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        ]
        if enable_flip:
            ops.append(T.RandomHorizontalFlip())
    else:
        resize = int(size / crop_ratio + 0.5)
        ops = [
            T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size),
        ]

    if force_rgb:
        # Convert to 3 channels (e.g., MNIST is grayscale 1-ch)
        ops.append(T.Grayscale(num_output_channels=3))

    ops.extend([
        T.ToTensor(),
        T.Normalize(IMNET_MEAN, IMNET_STD)
        if getattr(args, "imagenet_default_mean_and_std", True)
        else T.Lambda(lambda x: x),
    ])
    return T.Compose(ops)


# ------------------------------
# Dataset builders
# ------------------------------

def _build_cifar10(args, is_train: bool):
    tfm = _img_transforms(args, is_train)
    ds = datasets.CIFAR10(root=args.data_path, train=is_train, download=True, transform=tfm)
    return ds, 10

def _build_cifar100(args, is_train: bool):
    tfm = _img_transforms(args, is_train)
    ds = datasets.CIFAR100(root=args.data_path, train=is_train, download=True, transform=tfm)
    return ds, 100

def _build_mnist(args, is_train: bool):
    # digits: force_rgb & disable flip
    tfm = _img_transforms(args, is_train, force_rgb=True, enable_flip=False)
    ds = datasets.MNIST(root=args.data_path, train=is_train, download=True, transform=tfm)
    return ds, 10

def _build_fashion_mnist(args, is_train: bool):
    tfm = _img_transforms(args, is_train, force_rgb=True, enable_flip=False)
    ds = datasets.FashionMNIST(root=args.data_path, train=is_train, download=True, transform=tfm)
    return ds, 10

def _build_svhn(args, is_train: bool):
    # SVHN uses split='train'/'test' (there is also 'extra'); disable flip for digits
    split = 'train' if is_train else 'test'
    tfm = _img_transforms(args, is_train, enable_flip=False)
    ds = datasets.SVHN(root=args.data_path, split=split, download=True, transform=tfm)
    return ds, 10

def _build_stl10(args, is_train: bool):
    split = 'train' if is_train else 'test'
    tfm = _img_transforms(args, is_train)
    ds = datasets.STL10(root=args.data_path, split=split, download=True, transform=tfm)
    return ds, 10

def _build_food101(args, is_train: bool):
    split = 'train' if is_train else 'test'
    tfm = _img_transforms(args, is_train)
    ds = datasets.Food101(root=args.data_path, split=split, download=True, transform=tfm)
    return ds, 101

def _build_pets(args, is_train: bool):
    # Oxford-IIIT Pet uses 'trainval'/'test'
    split = 'trainval' if is_train else 'test'
    tfm = _img_transforms(args, is_train)
    ds = datasets.OxfordIIITPet(root=args.data_path, split=split, download=True, transform=tfm)
    return ds, 37

def _build_flowers102(args, is_train: bool):
    # Flowers102 supports 'train'/'val'/'test'
    split = 'train' if is_train else 'val'
    tfm = _img_transforms(args, is_train)
    ds = datasets.Flowers102(root=args.data_path, split=split, download=True, transform=tfm)
    return ds, 102

def _build_cars(args, is_train: bool):
    # Stanford Cars: 'train'/'test'. The official URL is down; download=True is a no-op.
    # Prepare the data manually under args.data_path before use.
    split = 'train' if is_train else 'test'
    tfm = _img_transforms(args, is_train)
    ds = datasets.StanfordCars(root=args.data_path, split=split, download=False, transform=tfm)
    return ds, 196


# ------------------------------
# COCO (single-label classification wrapper)
# ------------------------------

class _CocoSingleLabel(torch.utils.data.Dataset):
    """
    Wraps torchvision CocoDetection and maps each image -> single class id
    using the majority category among its annotations.

    Note: COCO is inherently multi-label; this is a pragmatic simplification
    that lets you keep using a standard CrossEntropyLoss.
    """
    def __init__(self, img_root: str, ann_file: str, transform):
        # base detection dataset (no transforms here; we apply ours to image only)
        self.base = CocoDetection(img_root, ann_file, transform=None, target_transform=None)
        self.transform = transform

        # Build a contiguous label mapping (COCO category ids are non-contiguous)
        cat_ids = sorted(self.base.coco.getCatIds())
        self.catid2idx = {cid: i for i, cid in enumerate(cat_ids)}
        self.idx2catid = cat_ids
        self.num_classes = len(cat_ids)  # typically 80 for COCO 2017

        # Precompute indices to keep and their single label
        keep, labels = [], []
        for i, img_id in enumerate(self.base.ids):
            ann_ids = self.base.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            if not ann_ids:
                continue
            anns = self.base.coco.loadAnns(ann_ids)
            counts = Counter(
                self.catid2idx[a["category_id"]] for a in anns if "category_id" in a
            )
            if not counts:
                continue
            label = counts.most_common(1)[0][0]
            keep.append(i)
            labels.append(label)

        self.keep = keep
        self.labels = labels

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, idx: int):
        base_idx = self.keep[idx]
        img, _ = self.base[base_idx]  # ignore original target (we precomputed label)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def _build_coco(args, is_train: bool):
    # Expecting COCO 2017 layout:
    # {data_path}/train2017, {data_path}/val2017, {data_path}/annotations/instances_*.json
    split_dir = "train2017" if is_train else "val2017"
    img_root = os.path.join(args.data_path, split_dir)
    ann_file = os.path.join(args.data_path, "annotations", f"instances_{split_dir}.json")

    # Fail fast if paths are missing
    if not os.path.isdir(img_root):
        raise FileNotFoundError(f"[COCO] Missing images folder: {img_root}")
    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"[COCO] Missing annotations file: {ann_file}")

    tfm = _img_transforms(args, is_train)
    ds = _CocoSingleLabel(img_root, ann_file, tfm)
    return ds, ds.num_classes


# ------------------------------
# Router
# ------------------------------

def build_dataset(args, is_train: bool):
    name = (args.dataset or "").lower().replace("-", "_")

    if name in ("cifar10",):
        return _build_cifar10(args, is_train)
    if name in ("cifar100", "cifar_100"):
        return _build_cifar100(args, is_train)

    if name in ("mnist",):
        return _build_mnist(args, is_train)
    if name in ("fashion_mnist", "fashionmnist"):
        return _build_fashion_mnist(args, is_train)
    if name in ("svhn",):
        return _build_svhn(args, is_train)
    if name in ("stl10",):
        return _build_stl10(args, is_train)

    if name in ("food101", "food_101"):
        return _build_food101(args, is_train)
    if name in ("oxfordiiitpet", "oxford_iiit_pet", "pets", "oxford_pets"):
        return _build_pets(args, is_train)
    if name in ("flowers102", "oxfordflowers102", "oxford_flowers102"):
        return _build_flowers102(args, is_train)
    if name in ("stanford_cars", "stanfordcars", "cars"):
        return _build_cars(args, is_train)

    if name in ("coco", "coco2017", "mscoco", "mscoco2017"):
        return _build_coco(args, is_train)

    raise NotImplementedError(f"Unsupported dataset '{args.dataset}'.")
