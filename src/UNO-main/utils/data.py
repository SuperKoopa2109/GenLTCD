import torch
import torchvision
import pytorch_lightning as pl

from utils.transforms import get_transforms
from utils.transforms import DiscoverTargetTransform

import numpy as np
import os


def get_datamodule(args, mode):
    if mode == "pretrain":
        if args.dataset == "ImageNet":
            return PretrainImageNetDataModule(args)
        else:
            return PretrainCIFARDataModule(args)
    elif mode == "discover":
        if args.dataset == "ImageNet":
            return DiscoverImageNetDataModule(args)
        else:
            return DiscoverCIFARDataModule(args)


class PretrainCIFARDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.download = args.download
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes
        self.dataset_class = getattr(torchvision.datasets, args.dataset)
        self.transform_train = get_transforms("unsupervised", args.dataset)
        self.transform_val = get_transforms("eval", args.dataset)

    def prepare_data(self):
        self.dataset_class(self.data_dir, train=True, download=self.download)
        self.dataset_class(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):

        labeled_classes = range(self.num_labeled_classes)

        # load the train dataset
        self.train_dataset = self.dataset_class(
            self.data_dir, train=True, transform=self.transform_train
        )

        # load the val dataset
        self.val_dataset = self.dataset_class(
            self.data_dir, train=False, transform=self.transform_val
        )


        # --------------

        # https://github.com/ryanchankh/cifar100coarse/blob/master/cifar100coarse.py

        # update labels
        coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                    3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                    6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                                    0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                    5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                    16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                    10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                                    2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
        targets = coarse_labels[self.train_dataset.targets]

        # update classes
        classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                        ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                        ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                        ['bottle', 'bowl', 'can', 'cup', 'plate'],
                        ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                        ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                        ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                        ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                        ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                        ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                        ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                        ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                        ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                        ['crab', 'lobster', 'snail', 'spider', 'worm'],
                        ['baby', 'boy', 'girl', 'man', 'woman'],
                        ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                        ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                        ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                        ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                        ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]

        num_classes = 100

        train_num_samples_per_class = []
        val_num_samples_per_class = []

        assigned_labeled_train_classes = {}
        labeled_train_classes = []
        assigned_labeled_val_classes = {}
        labeled_val_classes = []
        assigned_unlabeled_classes = {}
        unlabeled_classes = []

        # starting from class 0, perform long-tailed sampling by using a decreasing exponential function
        # the number of samples considered per class is stored as a list
        for class_id in range(num_classes):
            superclass = coarse_labels[class_id]
            sample_size = int(500  * (0.96 ** class_id))
            sample_size_val = int(100  * (0.96 ** class_id))

            # if superclass is not assigned yet, assign to dictionary
            if superclass not in assigned_labeled_train_classes.keys():
                assigned_labeled_train_classes[superclass] = []
                assigned_labeled_val_classes[superclass] = []
            
            # Check if super class has 4 or more classes
            if len( assigned_labeled_train_classes[superclass] ) <= 3:
                # Add class to dictionary to check how many classes have been appended
                assigned_labeled_train_classes[superclass].append(class_id)
                # Add sample size and class id to array
                
                labeled_train_classes.append((class_id, sample_size))
                
                assigned_labeled_val_classes[superclass].append(class_id)
                
                labeled_val_classes.append((class_id, sample_size_val))
                
                # If there are already 4 classes assigned to superclass add last class to unlabeled dataset -> split of highly correlated semantic data 4/5 to 1/5
            else:
                assigned_unlabeled_classes[superclass] = []
                assigned_unlabeled_classes[superclass].append(class_id)
            

            
            train_num_samples_per_class.append(int(500  * (0.96 ** class_id)))
            val_num_samples_per_class.append(int(100  * (0.96 ** class_id)))

            # since the elements of the train/val dataset variables are not sorted by the numeric value of the class, we first filter each sample by its class yielding always
            # 500 indices for train and 100 samples for val. Then, the long-tailed sampling is performed by taking only as much samples as the value at position i in train/val_num_samples_per_class
            # i.e., the first class 0 will have the most samples and the last class will have the least samples
            # the Subset class is then used to take the corresponding samples from the dataset and at the same time drop the remaining classes from the dataset

            index_list_final_train = []
            index_list_final_val = []
            for class_id, sample_size in labeled_train_classes:

                index_list_final_train += np.where(np.isin(np.array(self.train_dataset.targets), [class_id]))[0][:sample_size].tolist()

            for class_id, sample_size in labeled_val_classes:
                index_list_final_val += np.where(np.isin(np.array(self.val_dataset.targets), [class_id]))[0][:sample_size].tolist()

            self.train_dataset = torch.utils.data.Subset(self.train_dataset, index_list_final_train)
            self.val_dataset = torch.utils.data.Subset(self.val_dataset, index_list_final_val)

        # --------------

        # # starting from class 0, store the number of samples considered per class as a list
        # train_num_samples_per_class = []
        # val_num_samples_per_class = []
        # for i in labeled_classes:
        #     train_num_samples_per_class.append(int(500 * self.g(i)))
        #     val_num_samples_per_class.append(int(100 * self.g(i)))

        # print(train_num_samples_per_class)
        # print(val_num_samples_per_class)
        # print(len(train_num_samples_per_class))
        # print(len(val_num_samples_per_class))

        # # since the elements of the train/val dataset variables are not sorted by the numeric value of the class,
        # # we first filter each sample by its class yielding always 500 indices for train and 100 indices for val.
        # # Then, the long-tailed sampling is performed by taking only as much samples as the value at position i in
        # # train/val_num_samples_per_class i.e., the first class 0 will have the most samples and the last class will
        # # have the least samples. The Subset class is then used to take the corresponding samples from the dataset and
        # # at the same time drop the remaining classes from the dataset
        # index_list_final_train = []
        # index_list_final_val = []
        # for i in labeled_classes:
        #     index_list_final_train += np.where(np.isin(np.array(self.train_dataset.targets), [i]))[0][
        #                               :train_num_samples_per_class[i]].tolist()
        #     index_list_final_val += np.where(np.isin(np.array(self.val_dataset.targets), [i]))[0][
        #                             :val_num_samples_per_class[i]].tolist()

        # self.train_dataset = torch.utils.data.Subset(self.train_dataset, index_list_final_train)
        # self.val_dataset = torch.utils.data.Subset(self.val_dataset, index_list_final_val)

    def g(self, i):
        return 0.96 ** i

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )


class DiscoverCIFARDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.download = args.download
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes
        self.dataset_class = getattr(torchvision.datasets, args.dataset)
        self.transform_train = get_transforms(
            "unsupervised",
            args.dataset,
            multicrop=args.multicrop,
            num_large_crops=args.num_large_crops,
            num_small_crops=args.num_small_crops,
        )
        self.transform_val = get_transforms("eval", args.dataset)

    def prepare_data(self):
        self.dataset_class(self.data_dir, train=True, download=self.download)
        self.dataset_class(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        labeled_classes = range(self.num_labeled_classes)
        unlabeled_classes = range(
            self.num_labeled_classes, self.num_labeled_classes + self.num_unlabeled_classes
        )

        # train dataset
        self.train_dataset = self.dataset_class(
            self.data_dir, train=True, transform=self.transform_train
        )

        # val datasets
        val_dataset_train = self.dataset_class(
            self.data_dir, train=True, transform=self.transform_val
        )
        val_dataset_test = self.dataset_class(
            self.data_dir, train=False, transform=self.transform_val
        )
        # unlabeled classes, train set
        val_indices_unlab_train = np.where(
            np.isin(np.array(val_dataset_train.targets), unlabeled_classes)
        )[0]
        val_subset_unlab_train = torch.utils.data.Subset(val_dataset_train, val_indices_unlab_train)
        # unlabeled classes, test set
        val_indices_unlab_test = np.where(
            np.isin(np.array(val_dataset_test.targets), unlabeled_classes)
        )[0]
        val_subset_unlab_test = torch.utils.data.Subset(val_dataset_test, val_indices_unlab_test)
        # labeled classes, test set
        val_indices_lab_test = np.where(
            np.isin(np.array(val_dataset_test.targets), labeled_classes)
        )[0]
        val_subset_lab_test = torch.utils.data.Subset(val_dataset_test, val_indices_lab_test)

        self.val_datasets = [val_subset_unlab_train, val_subset_unlab_test, val_subset_lab_test]

    @property
    def dataloader_mapping(self):
        return {0: "unlab/train", 1: "unlab/test", 2: "lab/test"}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return [
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            for dataset in self.val_datasets
        ]


IMAGENET_CLASSES_118 = [
    "n01498041",
    "n01537544",
    "n01580077",
    "n01592084",
    "n01632777",
    "n01644373",
    "n01665541",
    "n01675722",
    "n01688243",
    "n01729977",
    "n01775062",
    "n01818515",
    "n01843383",
    "n01883070",
    "n01950731",
    "n02002724",
    "n02013706",
    "n02092339",
    "n02093256",
    "n02095314",
    "n02097130",
    "n02097298",
    "n02098413",
    "n02101388",
    "n02106382",
    "n02108089",
    "n02110063",
    "n02111129",
    "n02111500",
    "n02112350",
    "n02115913",
    "n02117135",
    "n02120505",
    "n02123045",
    "n02125311",
    "n02134084",
    "n02167151",
    "n02190166",
    "n02206856",
    "n02231487",
    "n02256656",
    "n02398521",
    "n02480855",
    "n02481823",
    "n02490219",
    "n02607072",
    "n02666196",
    "n02672831",
    "n02704792",
    "n02708093",
    "n02814533",
    "n02817516",
    "n02840245",
    "n02843684",
    "n02870880",
    "n02877765",
    "n02966193",
    "n03016953",
    "n03017168",
    "n03026506",
    "n03047690",
    "n03095699",
    "n03134739",
    "n03179701",
    "n03255030",
    "n03388183",
    "n03394916",
    "n03424325",
    "n03467068",
    "n03476684",
    "n03483316",
    "n03627232",
    "n03658185",
    "n03710193",
    "n03721384",
    "n03733131",
    "n03785016",
    "n03786901",
    "n03792972",
    "n03794056",
    "n03832673",
    "n03843555",
    "n03877472",
    "n03899768",
    "n03930313",
    "n03935335",
    "n03954731",
    "n03995372",
    "n04004767",
    "n04037443",
    "n04065272",
    "n04069434",
    "n04090263",
    "n04118538",
    "n04120489",
    "n04141975",
    "n04152593",
    "n04154565",
    "n04204347",
    "n04208210",
    "n04209133",
    "n04258138",
    "n04311004",
    "n04326547",
    "n04367480",
    "n04447861",
    "n04483307",
    "n04522168",
    "n04548280",
    "n04554684",
    "n04597913",
    "n04612504",
    "n07695742",
    "n07697313",
    "n07697537",
    "n07716906",
    "n12998815",
    "n13133613",
]

IMAGENET_CLASSES_30 = {
    "A": [
        "n01580077",
        "n01688243",
        "n01883070",
        "n02092339",
        "n02095314",
        "n02098413",
        "n02108089",
        "n02120505",
        "n02123045",
        "n02256656",
        "n02607072",
        "n02814533",
        "n02840245",
        "n02843684",
        "n02877765",
        "n03179701",
        "n03424325",
        "n03483316",
        "n03627232",
        "n03658185",
        "n03785016",
        "n03794056",
        "n03899768",
        "n04037443",
        "n04069434",
        "n04118538",
        "n04154565",
        "n04311004",
        "n04522168",
        "n07695742",
    ],
    "B": [
        "n01883070",
        "n02013706",
        "n02093256",
        "n02097130",
        "n02101388",
        "n02106382",
        "n02112350",
        "n02167151",
        "n02490219",
        "n02814533",
        "n02843684",
        "n02870880",
        "n03017168",
        "n03047690",
        "n03134739",
        "n03394916",
        "n03424325",
        "n03483316",
        "n03658185",
        "n03721384",
        "n03733131",
        "n03786901",
        "n03843555",
        "n04120489",
        "n04152593",
        "n04208210",
        "n04258138",
        "n04522168",
        "n04554684",
        "n12998815",
    ],
    "C": [
        "n01580077",
        "n01592084",
        "n01632777",
        "n01775062",
        "n01818515",
        "n02097130",
        "n02097298",
        "n02098413",
        "n02111500",
        "n02115913",
        "n02117135",
        "n02398521",
        "n02480855",
        "n02817516",
        "n02843684",
        "n02877765",
        "n02966193",
        "n03095699",
        "n03394916",
        "n03424325",
        "n03710193",
        "n03733131",
        "n03785016",
        "n03995372",
        "n04090263",
        "n04120489",
        "n04326547",
        "n04522168",
        "n07697537",
        "n07716906",
    ],
}


class DiscoverDataset:
    def __init__(self, labeled_dataset, unlabeled_dataset):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset

    def __len__(self):
        return max([len(self.labeled_dataset), len(self.unlabeled_dataset)])

    def __getitem__(self, index):
        labeled_index = index % len(self.labeled_dataset)
        labeled_data = self.labeled_dataset[labeled_index]
        unlabeled_index = index % len(self.unlabeled_dataset)
        unlabeled_data = self.unlabeled_dataset[unlabeled_index]
        return (*labeled_data, *unlabeled_data)


class DiscoverImageNetDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.imagenet_split = args.imagenet_split
        self.dataset_class = torchvision.datasets.ImageFolder
        self.transform_train = get_transforms(
            "unsupervised",
            args.dataset,
            multicrop=args.multicrop,
            num_large_crops=args.num_large_crops,
            num_small_crops=args.num_small_crops,
        )
        self.transform_val = get_transforms("eval", args.dataset)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_data_dir = os.path.join(self.data_dir, "train")
        val_data_dir = os.path.join(self.data_dir, "val")

        # train dataset
        train_dataset = self.dataset_class(train_data_dir, transform=self.transform_train)

        # split classes
        mapping = {c[:9]: i for c, i in train_dataset.class_to_idx.items()}
        labeled_classes = list(set(mapping.keys()) - set(IMAGENET_CLASSES_118))
        labeled_classes.sort()
        labeled_class_idxs = [mapping[c] for c in labeled_classes]
        unlabeled_classes = IMAGENET_CLASSES_30[self.imagenet_split]
        unlabeled_classes.sort()
        unlabeled_class_idxs = [mapping[c] for c in unlabeled_classes]

        # target transform
        all_classes = labeled_classes + unlabeled_classes
        target_transform = DiscoverTargetTransform(
            {mapping[c]: i for i, c in enumerate(all_classes)}
        )
        train_dataset.target_transform = target_transform

        # train set
        targets = np.array([img[1] for img in train_dataset.imgs])
        labeled_idxs = np.where(np.isin(targets, np.array(labeled_class_idxs)))[0]
        labeled_subset = torch.utils.data.Subset(train_dataset, labeled_idxs)
        unlabeled_idxs = np.where(np.isin(targets, np.array(unlabeled_class_idxs)))[0]
        unlabeled_subset = torch.utils.data.Subset(train_dataset, unlabeled_idxs)
        self.train_dataset = DiscoverDataset(labeled_subset, unlabeled_subset)

        # val datasets
        val_dataset_train = self.dataset_class(
            train_data_dir, transform=self.transform_val, target_transform=target_transform
        )
        val_dataset_test = self.dataset_class(
            val_data_dir, transform=self.transform_val, target_transform=target_transform
        )
        targets_train = np.array([img[1] for img in val_dataset_train.imgs])
        targets_test = np.array([img[1] for img in val_dataset_test.imgs])
        # unlabeled classes, train set
        unlabeled_idxs = np.where(np.isin(targets_train, np.array(unlabeled_class_idxs)))[0]
        unlabeled_subset_train = torch.utils.data.Subset(val_dataset_train, unlabeled_idxs)
        # unlabeled classes, test set
        unlabeled_idxs = np.where(np.isin(targets_test, np.array(unlabeled_class_idxs)))[0]
        unlabeled_subset_test = torch.utils.data.Subset(val_dataset_test, unlabeled_idxs)
        # labeled classes, test set
        labeled_idxs = np.where(np.isin(targets_test, np.array(labeled_class_idxs)))[0]
        labeled_subset_test = torch.utils.data.Subset(val_dataset_test, labeled_idxs)

        self.val_datasets = [unlabeled_subset_train, unlabeled_subset_test, labeled_subset_test]

    @property
    def dataloader_mapping(self):
        return {0: "unlab/train", 1: "unlab/test", 2: "lab/test"}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size // 2,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return [
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            for dataset in self.val_datasets
        ]


class PretrainImageNetDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.dataset_class = torchvision.datasets.ImageFolder
        self.transform_train = get_transforms("unsupervised", args.dataset)
        self.transform_val = get_transforms("eval", args.dataset)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_data_dir = os.path.join(self.data_dir, "train")
        val_data_dir = os.path.join(self.data_dir, "val")

        # train dataset
        train_dataset = self.dataset_class(train_data_dir, transform=self.transform_train)

        # find labeled classes
        mapping = {c[:9]: i for c, i in train_dataset.class_to_idx.items()}
        labeled_classes = list(set(mapping.keys()) - set(IMAGENET_CLASSES_118))
        labeled_classes.sort()
        labeled_class_idxs = [mapping[c] for c in labeled_classes]

        # target transform
        target_transform = DiscoverTargetTransform(
            {mapping[c]: i for i, c in enumerate(labeled_classes)}
        )
        train_dataset.target_transform = target_transform

        # train set
        targets = np.array([img[1] for img in train_dataset.imgs])
        labeled_idxs = np.where(np.isin(targets, np.array(labeled_class_idxs)))[0]
        self.train_dataset = torch.utils.data.Subset(train_dataset, labeled_idxs)

        # val datasets
        val_dataset = self.dataset_class(
            val_data_dir, transform=self.transform_val, target_transform=target_transform
        )
        targets = np.array([img[1] for img in val_dataset.imgs])
        # labeled classes, test set
        labeled_idxs = np.where(np.isin(targets, np.array(labeled_class_idxs)))[0]
        self.val_dataset = torch.utils.data.Subset(val_dataset, labeled_idxs)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
