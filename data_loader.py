import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


def load_data(data_dir: str, total_train_samples, batch_size, labeled_unlabeled_split=(0.25, 0.75),
              total_test_samples=100, total_val_samples=100, seed=42):
    """
    Loads and preprocesses the X-ray image dataset using PyTorch's DataLoader.

    Args:
        data_dir (str): Directory where data is stored.
        total_train_samples (int): Number of train samples including labeled and the pool of unlabeled samples
        batch_size (int): Batch size for training.
        labeled_unlabeled_split (tuple): the ratio between labeled (1st in tuple) and unlabeled (2nd in tuple) samples
        total_test_samples (int): Number of test samples
        total_val_samples (int): Target number of validation samples (from unused training data).
        seed (int): Random seed

    Returns:
        train_loader_labeled, train_loader_unlabeled, val_loader, test_loader: Data loaders for training, validation,
        and test sets.
    """
    torch.manual_seed(seed)

    transform = transforms.Compose([
        transforms.Resize((312, 312)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    # Subset train_dataset to total_train_samples
    train_dataset = torch.utils.data.Subset(
        train_dataset, np.random.choice(len(train_dataset), total_train_samples, replace=False))

    train_dataset_labeled, train_dataset_unlabeled = torch.utils.data.random_split(
        train_dataset, labeled_unlabeled_split)

    # Handle unused training samples for validation
    unused_train_samples = len(train_dataset_unlabeled.indices)
    needed_val_samples = total_val_samples - len(val_dataset)

    if needed_val_samples > 0 and unused_train_samples > 0:
        num_to_add = min(needed_val_samples, unused_train_samples)
        additional_val_indices = np.random.choice(train_dataset_unlabeled.indices, num_to_add, replace=False)
        val_dataset = torch.utils.data.ConcatDataset(
            [val_dataset, torch.utils.data.Subset(train_dataset_unlabeled.dataset, additional_val_indices)])

    test_dataset = torch.utils.data.Subset(
        test_dataset, np.random.choice(len(test_dataset), total_test_samples, replace=False))

    train_loader_labeled = DataLoader(train_dataset_labeled, batch_size=batch_size, shuffle=True)
    train_loader_unlabeled = DataLoader(train_dataset_unlabeled, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader_labeled, train_loader_unlabeled, val_loader, test_loader

