import os
from collections import Counter

import torch
import torchvision.transforms as T
from torchvision import datasets
from torchvision.datasets import CocoDetection

# ------------------------------
# Minimal transforms: Resize -> ToTensor
# - No normalization, flips, crops, jitters, etc.
# - If input is grayscale (1ch), duplicate to 3ch after ToTensor for 3-ch models.
# ------------------------------
def _img_transforms(args, is_train: bool):
    size = int(getattr(args, "input_size", 224))
    # Use fixed-size resize; keep it identical for train/val
    ops = [
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),  # converts to float in [0,1] for PIL inputs
        T.Lambda(lambda x: x.expand(3, -1, -1) if x.shape[0] == 1 else x),  # 1ch -> 3ch
    ]
    return T.Compose(ops)

# ------------------------------
# Dataset builders (unchanged behavior, new transforms)
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
    tfm = _img_transforms(args, is_train)
    ds = datasets.MNIST(root=args.data_path, train=is_train, download=True, transform=tfm)
    return ds, 10

def _build_fashion_mnist(args, is_train: bool):
    tfm = _img_transforms(args, is_train)
    ds = datasets.FashionMNIST(root=args.data_path, train=is_train, download=True, transform=tfm)
    return ds, 10

def _build_svhn(args, is_train: bool):
    split = 'train' if is_train else 'test'
    tfm = _img_transforms(args, is_train)
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
    split = 'trainval' if is_train else 'test'
    tfm = _img_transforms(args, is_train)
    ds = datasets.OxfordIIITPet(root=args.data_path, split=split, download=True, transform=tfm)
    return ds, 37

def _build_flowers102(args, is_train: bool):
    split = 'train' if is_train else 'val'
    tfm = _img_transforms(args, is_train)
    ds = datasets.Flowers102(root=args.data_path, split=split, download=True, transform=tfm)
    return ds, 102

def _build_cars(args, is_train: bool):
    split = 'train' if is_train else 'test'
    tfm = _img_transforms(args, is_train)
    ds = datasets.StanfordCars(root=args.data_path, split=split, download=False, transform=tfm)
    return ds, 196

# ------------------------------
# COCO (single-label classification wrapper)
# ------------------------------
class _CocoSingleLabel(torch.utils.data.Dataset):
    """Wrap CocoDetection and assign ONE label per image = majority category."""
    def __init__(self, img_root: str, ann_file: str, transform):
        self.base = CocoDetection(img_root, ann_file, transform=None, target_transform=None)
        self.transform = transform

        cat_ids = sorted(self.base.coco.getCatIds())
        self.catid2idx = {cid: i for i, cid in enumerate(cat_ids)}
        self.idx2catid = cat_ids
        self.num_classes = len(cat_ids)  # typically 80

        keep, labels = [], []
        for i, img_id in enumerate(self.base.ids):
            ann_ids = self.base.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            if not ann_ids:
                continue
            anns = self.base.coco.loadAnns(ann_ids)
            counts = Counter(self.catid2idx[a["category_id"]] for a in anns if "category_id" in a)
            if not counts:
                continue
            keep.append(i)
            labels.append(counts.most_common(1)[0][0])

        self.keep = keep
        self.labels = labels

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, idx: int):
        base_idx = self.keep[idx]
        img, _ = self.base[base_idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def _build_coco(args, is_train: bool):
    split_dir = "train2017" if is_train else "val2017"
    img_root = os.path.join(args.data_path, split_dir)
    ann_file = os.path.join(args.data_path, "annotations", f"instances_{split_dir}.json")

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
