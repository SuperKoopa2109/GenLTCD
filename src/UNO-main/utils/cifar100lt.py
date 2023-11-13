import numpy as np
from .sampler import ClassAwareSampler

import torch
import torchvision
from torchvision import transforms
import torchvision.datasets
import pytorch_lightning as pl


class IMBALANCECIFAR100(torchvision.datasets.CIFAR100):
    cls_num = 100

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 num_labeled_classes=80,
                 num_unlabeled_classes=20,
                 labeled=True):
        super(IMBALANCECIFAR100, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        if num_labeled_classes + num_unlabeled_classes != 100:
            print("Warning: number of unlabeled classes + number of labeled classes != 100")
            print("number of unlabeled classes: ", num_unlabeled_classes)
            print("number of labeled classes: ", num_labeled_classes)
            print("Sum: ", num_labeled_classes+num_unlabeled_classes)
        self.num_labeled_classes = num_labeled_classes
        self.num_unlabeled_classes = num_unlabeled_classes
        self.train = train
        self.labeled = labeled
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)


    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            if self.labeled:
                for cls_idx in range(self.num_labeled_classes):
                    num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
                img_num_per_cls.extend([0] * self.num_unlabeled_classes)
            else: 
                img_num_per_cls.extend([0] * self.num_labeled_classes)
                for cls_idx in range(self.num_labeled_classes, self.num_labeled_classes + self.num_unlabeled_classes):
                    num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))

        elif imb_type == 'step':
            print("Warning: not well implemented")
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max) if self.labeled else 0] * self.num_labeled_classes)
            img_num_per_cls.extend([0 if self.labeled else int(img_max)] * self.num_unlabeled_classes)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class CIFAR100_LT_discover(pl.LightningDataModule):
    def __init__(self, distributed, num_labeled_classes, num_unlabeled_classes, root='./data/cifar100', imb_type='exp',
                 imb_factor=0.01, batch_size=128, num_works=8,):
        super().__init__()
        self.eval = None
        self.train_balance = None
        self.num_labeled_classes = num_labeled_classes
        self.num_unlabeled_classes = num_unlabeled_classes
        
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.batch_size = batch_size

        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.num_works = num_works
        self.train_dataset = IMBALANCECIFAR100(root=root, imb_type="balanced", imb_factor=imb_factor, rand_number=0,
                                               train=True, download=True, transform=train_transform, labeled=True)

        self.eval_dataset = IMBALANCECIFAR100(root=root, imb_type="exp", imb_factor=imb_factor, rand_number=0, train=False, download=True, transform=eval_transform, labeled=True)

        val_subset_unlab_train = IMBALANCECIFAR100(root=root, imb_type="balanced", labeled=False, imb_factor=imb_factor, rand_number=0, train=True, download=True, transform=train_transform)
        val_subset_unlab_test = IMBALANCECIFAR100(root=root, imb_type="exp", labeled=False, imb_factor=imb_factor, rand_number=0, train=False, download=True, transform=eval_transform)
        val_subset_lab_test = IMBALANCECIFAR100(root=root, imb_type="exp", labeled=True, imb_factor=imb_factor, rand_number=0, train=False, download=True, transform=eval_transform)

        self.val_datasets = [val_subset_unlab_train, val_subset_unlab_test, val_subset_lab_test]

    def prepare_data(self):
        pass

    @property
    def dataloader_mapping(self):
        return {0: "unlab/train", 1: "unlab/test", 2: "lab/test"}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_works,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return [
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_works,
                pin_memory=True,
                drop_last=False,
            )
            for dataset in self.val_datasets
        ]

class CIFAR100_LT_pretrain(pl.LightningDataModule):
    def __init__(self, distributed, num_labeled_classes, num_unlabeled_classes, root='./data/cifar100', imb_type='exp',
                 imb_factor=0.01, batch_size=128, num_works=8,):
        super().__init__()
        self.eval = None
        self.train_balance = None
        self.num_labeled_classes = num_labeled_classes
        self.num_unlabeled_classes = num_unlabeled_classes
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.batch_size = batch_size

        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.num_works = num_works
        self.train_dataset = IMBALANCECIFAR100(root=root, imb_type="balanced", imb_factor=imb_factor, rand_number=0,
                                               train=True, download=True, transform=train_transform)

        self.eval_dataset = IMBALANCECIFAR100(root=root, imb_type="exp", imb_factor=imb_factor, rand_number=0, train=False, download=True, transform=eval_transform)

        self.cls_num_list = self.train_dataset.get_cls_num_list()

    def prepare_data(self):
        pass

    def train_dataloader(self):
        self.train_balance = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_works,
            pin_memory=True,
            sampler=self.balance_sampler,
            drop_last=True)
        return self.train_balance

    def val_dataloader(self):
        self.eval = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_works,
            pin_memory=True,
            drop_last=False)
        return self.eval  
