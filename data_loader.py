import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

def load_data(data_dir: str, total_train_samples,batch_size, labeled_unlabeled_split=(0.25,0.75)):
    """
    Loads and preprocesses the X-ray image dataset using PyTorch's DataLoader.

    Args:
        data_dir (str): Directory where data is stored.
        batch_size (int): Batch size for training.

    Returns:
        train_loader, val_loader, test_loader: Data loaders for training, validation, and test sets.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    train_dataset = torch.utils.data.Subset(train_dataset, np.random.choice(len(train_dataset), total_train_samples, replace=False))
    train_dataset_labeled, train_dataset_unlabeled = torch.utils.data.random_split(train_dataset, labeled_unlabeled_split)

    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    train_loader_labeled = DataLoader(train_dataset_labeled, batch_size=batch_size, shuffle=True)
    train_loader_unlabeled = DataLoader(train_dataset_unlabeled, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader_labeled, train_loader_unlabeled, val_loader, test_loader
