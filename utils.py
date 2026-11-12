import os
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms


class CarsDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = ImageFolder(
            root=os.path.join(data_path, 'train'), transform=self.transforms)
        testset = ImageFolder(
            root=os.path.join(data_path, 'test'), transform=self.transforms)
        self.dataset = ConcatDataset([trainset, testset])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
