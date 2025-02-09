# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
from PIL import Image

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torch.utils.data import Dataset

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import concurrent.futures
import random

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


class CustomizedImagenetDataset(Dataset):
    def __init__(self, root, transform=None, num_workers=2):
        self.root = root
        self.transform = transform
        
        # 获取所有类别目录，并进行排序
        self.classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        # 构造类别到索引的映射，例如 {'n01440764': 0, 'n01443537': 1, ...}
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # 遍历每个类别，收集 (图像路径, 标签) 对
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root, cls_name)
            label = self.class_to_idx[cls_name]
            path_with_label = []
            for fname in os.listdir(cls_dir):
                # 这里过滤 jpg、jpeg、png 格式的图像（可根据需要扩展）
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path_with_label.append([os.path.join(cls_dir, fname), label])
            # 多线程加载图片
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                image_with_label = list(executor.map(self._load_image, path_with_label))

            self.samples.extend(image_with_label)
        
            print(label)
    
    def _load_image(self, path_with_label):
        """
        加载单张图片并转换为 RGB 格式。
        """
        try:
            path, label = path_with_label
            if random.random() < 0.5:
                image = Image.open(path).convert('RGB')
            else:
                image = path
            return (image, label)
        except Exception as e:
            print(f"加载图片 {path} 时出错：{e}")
            # 发生错误时可以返回 None 或者一个默认图片，根据需求处理
            return None
                        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        image, label = self.samples[index]
        if type(image) is str:
            image = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        if False: #is_train:
            root = os.path.join(args.data_path, 'train')
            dataset = CustomizedImagenetDataset(root, transform=transform)
        else:
            root = os.path.join(args.data_path, 'train' if is_train else 'val')
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    if "MlpMixer" in args.model:
        t.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    else:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
