import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import PIL.Image


class AugmentedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, num_augmentations=50):
        super().__init__(root, transform)
        self.augmented_samples = []
        for img_path, label in self.samples:
            img = PIL.Image.open(img_path)
            for _ in range(num_augmentations):
                augmented_img = self.transform(img)
                self.augmented_samples.append((augmented_img, label))

    def __getitem__(self, index):
        return self.augmented_samples[index]

    def __len__(self):
        return len(self.augmented_samples)


def load_data(data_dir: str, total_train_samples, batch_size, labeled_unlabeled_split=(0.25,0.75),
              total_test_samples=100, seed=42):
    """
    Loads and preprocesses the X-ray image dataset using PyTorch's DataLoader.

    Args:
        data_dir (str): Directory where data is stored.
        total_train_samples (int): Number of train samples including labeled and the pool of unlabeled samples
        batch_size (int): Batch size for training.
        labeled_unlabeled_split (tuple): the ratio between labeled (1st in tuple) and unlabeled (2nd in tuple) samples
        total_test_samples (int): Number of train samples including labeled and the pool of unlabeled samples

    Returns:
        train_loader_labeled, train_loader_unlabeled, val_loader, test_loader: Data loaders for training, validation,
        and test sets.
    """
    torch.manual_seed(seed)
    # Resize images to 312x312 pixels -> Convert to PyTorch tensors. -> Normalizes to a standard range
    transform = transforms.Compose([
        transforms.Resize((312, 312)),
        transforms.RandomRotation(degrees=(-15, 15)),  # Randomly rotate images by -15 to 15 degrees
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Randomly translate and scale
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    # Set the paths for the data directories
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # randomly select some of the samples in each dataset, and split the test set to labeled and unlabeled sets
    # train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    train_dataset = AugmentedImageFolder(train_dir, transform=transform)
    print(train_dataset[0])
    print(train_dataset[50])
    print(train_dataset[100])
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    train_dataset = torch.utils.data.Subset(train_dataset, np.random.choice(len(train_dataset), total_train_samples, replace=False))
    train_dataset_labeled, train_dataset_unlabeled = torch.utils.data.random_split(train_dataset, labeled_unlabeled_split)
    test_dataset = torch.utils.data.Subset(test_dataset, np.random.choice(len(test_dataset), total_test_samples, replace=False))

    train_loader_labeled = DataLoader(train_dataset_labeled, batch_size=batch_size, shuffle=True)
    train_loader_unlabeled = DataLoader(train_dataset_unlabeled, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader_labeled, train_loader_unlabeled, val_loader, test_loader
