from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import shutil
from PIL import Image
from torchvision.datasets import VisionDataset
from typing import Callable, Optional, Tuple, Any
from flwr.dataset.utils.common import create_lda_partitions


def get_dataset_double(path_to_data1: Path, path_to_data2:Path, cid: str, partition: str):

    # generate path to cid's data
    path_to_data1 = path_to_data1 / cid / (partition + ".pt")
    path_to_data2 = path_to_data2 / cid / (partition + ".pt")

    return TorchVision_FL(path_to_data1, path_to_data2, transform1=cifar10Transformation(), transform2=stl10Transformation())

def get_dataloader(
    path_to_data: str, cid: str, is_train: bool, batch_size: int, workers: int
    ):
        """Generates trainset/valset object and returns appropiate dataloader."""

        partition = "train" if is_train else "val"
        dataset = get_dataset(Path(path_to_data), cid, partition)

        # we use as number of workers all the cpu cores assigned to this actor
        kwargs = {"num_workers": workers, "pin_memory": True, "drop_last": False}
        return DataLoader(dataset, batch_size=batch_size, **kwargs)


def get_dataloader_double(
    path_to_data1: str, path_to_data2:str,  cid: str, is_train: bool, batch_size: int, workers: int
):
    """Generates trainset/valset object and returns appropiate dataloader."""

    partition = "train" if is_train else "val"
    dataset = get_dataset_double(Path(path_to_data1), Path(path_to_data2), cid, partition)

    # we use as number of workers all the cpu cores assigned to this actor
    kwargs = {"num_workers": workers, "pin_memory": True, "drop_last": False, "shuffle":True}
    return DataLoader(dataset, batch_size=batch_size, **kwargs)


def get_random_id_splits(total: int, val_ratio: float, shuffle: bool = True):
    """splits a list of length `total` into two following a
    (1-val_ratio):val_ratio partitioning.

    By default the indices are shuffled before creating the split and
    returning.
    """

    if isinstance(total, int):
        indices = list(range(total))
    else:
        indices = total

    split = int(np.floor(val_ratio * len(indices)))
    # print(f"Users left out for validation (ratio={val_ratio}) = {split} ")
    if shuffle:
        np.random.shuffle(indices)
    return indices[split:], indices[:split]


def do_fl_partitioning(path_to_dataset, pool_size, alpha, num_classes, val_ratio=0.0):
    """Torchvision (e.g. CIFAR-10) datasets using LDA."""

    images, labels = torch.load(path_to_dataset)
    idx = np.array(range(len(images)))
    dataset = [idx, labels]
    partitions, _ = create_lda_partitions(
        dataset, num_partitions=pool_size, concentration=alpha, accept_imbalanced=True
    )

    # Show label distribution for first partition (purely informative)
    partition_zero = partitions[0][1]
    hist, _ = np.histogram(partition_zero, bins=list(range(num_classes + 1)))
    print(
        f"Class histogram for 0-th partition (alpha={alpha}, {num_classes} classes): {hist}"
    )

    # now save partitioned dataset to disk
    # first delete dir containing splits (if exists), then create it
    splits_dir = path_to_dataset.parent / "federated"
    if splits_dir.exists():
        shutil.rmtree(splits_dir)
    Path.mkdir(splits_dir, parents=True)

    for p in range(pool_size):

        labels = partitions[p][1]
        image_idx = partitions[p][0]
        imgs = images[image_idx]

        # create dir
        Path.mkdir(splits_dir / str(p))

        if val_ratio > 0.0:
            # split data according to val_ratio
            train_idx, val_idx = get_random_id_splits(len(labels), val_ratio)
            val_imgs = imgs[val_idx]
            val_labels = labels[val_idx]

            with open(splits_dir / str(p) / "val.pt", "wb") as f:
                torch.save([val_imgs, val_labels], f)

            # remaining images for training
            imgs = imgs[train_idx]
            labels = labels[train_idx]

        with open(splits_dir / str(p) / "train.pt", "wb") as f:
            torch.save([imgs, labels], f)

    return splits_dir

def cifar10TestTransformation():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

def cifar10Transformation():

    return transforms.Compose(
        
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            #transforms.Normalize((0.4467, 0.4398, 0.4065), (0.2023, 0.1994, 0.2010)),
        ]
    )

def stl10TestTransformation():
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
            #0.446710620659723 0.4398098398352401 0.40664644709967407 0.2603409782662332 0.2565772731134434 0.2712673814522549
        ]
    )

def stl10Transformation():

    return transforms.Compose(
        
        [
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
        ]
    )



class TorchVision_FL(VisionDataset):
    """This is just a trimmed down version of torchvision.datasets.MNIST.

    Use this class by either passing a path to a torch file (.pt)
    containing (data, targets) or pass the data, targets directly
    instead.
    """

    def __init__(
        self,
        path_to_data1=None,
        path_to_data2=None,
        data=None,
        targets=None,
        transform1: Optional[Callable] = None,
        transform2: Optional[Callable] = None,
    ) -> None:
        path1 = path_to_data1.parent if path_to_data1 else None
        path2 = path_to_data2.parent if path_to_data2 else None
        super(TorchVision_FL, self).__init__(path1, transform=transform1)
        self.transform1 = transform1
        self.transform2 = transform2

        if path_to_data1:
            # load data and targets (path_to_data points to an specific .pt file)
            self.data1, self.targets1 = torch.load(path_to_data1)
            self.data2, self.targets2 = torch.load(path_to_data2)
        else:
            self.data1 = data
            self.targets1 = targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if index < len(self.data1):
            img, target = self.data1[index], int(self.targets1[index])
            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            if not isinstance(img, Image.Image):  # if not PIL image
                if not isinstance(img, np.ndarray):  # if torch tensor
                    img = img.numpy()

                img = Image.fromarray(img)

            if self.transform1 is not None:
                img = self.transform1(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        else : 
            img, target = self.data2[index - len(self.data1)], int(self.targets2[index - len(self.data1)])
            if target == 2: 
                target = 1
            elif target == 1:
                target = 2
            elif target == 6:
                target = 7
            elif target == 7:
                target = 6
            if not isinstance(img, Image.Image):
                img_np = img.numpy() if isinstance(img, torch.Tensor) else img
                # print('dataset3_shape:', img_np.shape)
                img = Image.fromarray(img_np.transpose((1,2,0)))
            # if not isinstance(img, Image.Image):  # if not PIL image
            #     if not isinstance(img, np.ndarray):  # if torch tensor
            #         img = img.numpy()

            #     img = Image.fromarray(img)

            if self.transform2 is not None:
                img = self.transform2(img)
            
            if self.target_transform is not None:
                target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.data1) + len(self.data2)


def getCIFAR10(path_to_data="./data"):
    """Downloads CIFAR10 dataset and generates a unified training set (it will
    be partitioned later using the LDA partitioning mechanism."""

    # download dataset and load train set
    train_set = datasets.CIFAR10(root=path_to_data, train=True, download=True)

    # fuse all data splits into a single "training.pt"
    data_loc = Path(path_to_data) / "cifar-10-batches-py"
    training_data = data_loc / "training.pt"
    print("Generating unified CIFAR dataset")
    torch.save([train_set.data, np.array(train_set.targets)], training_data)
    test_set = datasets.CIFAR10(
        root=path_to_data, train=False, transform=cifar10TestTransformation()
    )

    # returns path where training data is and testset
    return training_data, test_set

def getSTL10(path_to_data="./data"):
    train_set = datasets.STL10(root=path_to_data, split='train', download=True)
    test_set = datasets.STL10(root=path_to_data, split='test', download=True)
    training_data_path = Path(path_to_data) / "stl10_binary" / "training.pt"
    combined_data = np.concatenate((train_set.data, test_set.data), axis=0)
    combined_labels = np.concatenate((np.array(train_set.labels), np.array(test_set.labels)), axis=0)
    torch.save([combined_data, combined_labels], training_data_path)
    return training_data_path, test_set
