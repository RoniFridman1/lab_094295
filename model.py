import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model_downloader import download_model


def initialize_model(model_name: str):
    """
    Initializes a pre-trained model downloaded via `model_downloader.py`.

    Args:
        model_name (str): The name of the model to initialize (e.g., 'resnet18', 'vgg16').

    Returns:
        model (torch.nn.Module): The initialized model.
    """
    model = download_model(model_name)

    # Modify the final layer for binary classification (Pneumonia vs. Normal)
    if model_name == 'resnet18':
        # ResNet18
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1)  # Update the fully connected layer for binary classification

    elif model_name == 'vgg16':
        # VGG16
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, 1)  # Update the last layer in classifier for binary classification

    return model


def train_model(model, train_loader, val_loader, epochs=10, learning_rate=1e-3):
    """
    Trains the model on the provided data loaders.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for optimizer.

    Returns:
        model (torch.nn.Module): Trained model.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

        # Validation step
        evaluate_model(model, val_loader)

    return model


def evaluate_model(model, data_loader):
    """
    Evaluates the model on a given dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the dataset.

    Returns:
        None
    """
    model.eval()
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Val Accuracy: {accuracy:.2f}%")
