import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import os
import json
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class ActiveLearningVgg16:
    def __init__(self):
        """ Initializes a pre-trained vgg16 model downloaded via `_load_or_download_vgg16`. """
        vgg16 = self._load_or_download_vgg16()
        num_features = vgg16.classifier[-1].in_features
        vgg16.classifier[-1] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 1))  # Update the last layer in classifier for binary classification
        self.model = vgg16

    def _load_or_download_vgg16(self, model_dir: str = "models"):
        """
        Downloads a pre-trained vgg16 model if not already present in the 'models' directory.

        Args:
            model_dir (str): Directory to store the downloaded models.

        Returns:
            model (torch.nn.Module): The requested pre-trained model.
        """
        model_path = os.path.join(model_dir, "vgg16.pth")

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Check if the model already exists
        if not os.path.exists(model_path):
            print(f"Downloading vgg16 model...")
            model = models.vgg16(pretrained=True)
            torch.save(model.state_dict(), model_path)

        else:
            print(f"Loading vgg16 from {model_path}...")
            model = models.vgg16()
            model.load_state_dict(torch.load(model_path))

        return model

    def _evaluate(self, train_loader, val_loader):
        """
        Evaluates the model on a given dataset.

        Args:
            train_loader (DataLoader): DataLoader for the train dataset
            val_loader (DataLoader): DataLoader for the validation dataset.

        Returns:
            None
        """
        self.model.eval()
        correct_val = 0
        total_val = 0
        total_train = 0
        correct_train = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
                outputs = self.model(images)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
                outputs = self.model(images)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        accuracy_train = 100 * correct_train / total_train
        accuracy_val = 100 * correct_val / total_val
        print(f"Train Accuracy: {accuracy_train:.2f}%")
        print(f"Val Accuracy: {accuracy_val:.2f}%")

    def train_model(self, train_loader, val_loader, epochs=10, learning_rate=1e-4):
        """
        Trains the vgg16 model on the provided data loaders.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            epochs (int): Number of epochs to train.
            learning_rate (float): Learning rate for optimizer.

        Returns:
            self (torch.nn.Module): Trained model.
        """

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
                optimizer.zero_grad()

                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

            # Validation step
            self._evaluate(train_loader, val_loader)

        return self

    def calculate_metrics(self, data_loader, output_dir='output', iteration=None, print_metrics=False):
        """
        Evaluates the model on a given dataset with additional metrics and saves results to files.

        Args:
            data_loader (DataLoader): DataLoader for the dataset.
            output_dir (str): Directory to save output files.
            iteration (int, optional): Iteration number for saving files with unique names.
            print_metrics (boolean): If to print the metrics or not

        Returns:
            dict: A dictionary containing the evaluation metrics.
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        self.model.eval()
        correct = 0
        total = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
                outputs = self.model(images)
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate metrics
        accuracy = correct / total
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        roc_auc = roc_auc_score(all_labels, all_probs)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        if print_metrics:
            print(f"Test Accuracy: {accuracy:.2f}%")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1-Score: {f1:.2f}")
            print(f"ROC-AUC: {roc_auc:.2f}")

        # Save metrics to a JSON file
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": conf_matrix.tolist()}  # Convert to list for JSON serialization

        metrics_filename = f"metrics_iteration_{iteration}.json" if iteration is not None else "metrics.json"
        with open(os.path.join(output_dir, metrics_filename), 'w') as f:
            json.dump(metrics, f, indent=4)

        # Save confusion matrix as an image
        ConfusionMatrixDisplay(conf_matrix).plot()
        conf_matrix_filename = f"confusion_matrix_iteration_{iteration}.png" if iteration is not None else "confusion_matrix.png"
        plt.savefig(os.path.join(output_dir, conf_matrix_filename))
        plt.close()  # Close the plot to avoid display overlap in loops

        return metrics


class ActiveLearningResnet18:
    def __init__(self):
        """ Initializes a pre-trained resnet18 model downloaded via `_load_or_download_resnet18`. """
        resnet18 = self._load_or_download_resnet18()
        num_features = resnet18.fc.in_features
        resnet18.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 1)
        )
        self.model = resnet18

    def _load_or_download_resnet18(self, model_dir: str = "models"):
        """
        Downloads a pre-trained resnet18 model if not already present in the 'models' directory.

        Args:
            model_dir (str): Directory to store the downloaded models.

        Returns:
            model (torch.nn.Module): The requested pre-trained model.
        """
        model_path = os.path.join(model_dir, "resnet18.pth")

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Check if already exists
        if not os.path.exists(model_path):
            print(f"Downloading resnet18...")
            model = models.resnet18(pretrained=True)
            torch.save(model.state_dict(), model_path)

        else:
            print(f"Loading resnet18 from {model_path}...")
            model = models.resnet18()
            model.load_state_dict(torch.load(model_path))

        return model

    def _evaluate(self, train_loader, val_loader):
        """
        Evaluates the model on a given dataset.

        Args:
            train_loader (DataLoader): DataLoader for the train dataset
            val_loader (DataLoader): DataLoader for the validation dataset.

        Returns:
            None
        """
        self.model.eval()
        correct_val = 0
        total_val = 0
        total_train = 0
        correct_train = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
                outputs = self.model(images)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
                outputs = self.model(images)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        accuracy_train = 100 * correct_train / total_train
        accuracy_val = 100 * correct_val / total_val
        print(f"Train Accuracy: {accuracy_train:.2f}%")
        print(f"Val Accuracy: {accuracy_val:.2f}%")

    def train_model(self, train_loader, val_loader, epochs=10, learning_rate=1e-4):
        """
        Trains the resnet18 model on the provided data loaders.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            epochs (int): Number of epochs to train.
            learning_rate (float): Learning rate for optimizer.

        Returns:
            resnet18 (torch.nn.Module): Trained ResNet18.
        """

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
                optimizer.zero_grad()

                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

            # Validation step
            self._evaluate(train_loader, val_loader)

        return self

    def calculate_metrics(self, data_loader, output_dir='output', iteration=None, print_metrics=False):
        """
        Evaluates the resnet18 on a given dataset with additional metrics and saves results to files.

        Args:
            data_loader (DataLoader): DataLoader for the dataset.
            output_dir (str): Directory to save output files.
            iteration (int, optional): Iteration number for saving files with unique names.
            print_metrics (boolean): If to print the metrics or not

        Returns:
            dict: A dictionary containing the evaluation metrics.
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        self.model.eval()
        correct = 0
        total = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
                outputs = self.model(images)
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate metrics
        accuracy = correct / total
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        roc_auc = roc_auc_score(all_labels, all_probs)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        if print_metrics:
            print(f"Test Accuracy: {accuracy:.2f}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1-Score: {f1:.2f}")
            print(f"ROC-AUC: {roc_auc:.2f}")

        # Save metrics to a JSON file
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": conf_matrix.tolist()}  # Convert to list for JSON serialization

        metrics_filename = f"metrics_iteration_{iteration}.json" if iteration is not None else "metrics.json"
        with open(os.path.join(output_dir, metrics_filename), 'w') as f:
            json.dump(metrics, f, indent=4)

        # Save confusion matrix as an image
        ConfusionMatrixDisplay(conf_matrix).plot()
        conf_matrix_filename = f"confusion_matrix_iteration_{iteration}.png" if iteration is not None else "confusion_matrix.png"
        plt.savefig(os.path.join(output_dir, conf_matrix_filename))
        plt.close()  # Close the plot to avoid display overlap in loops

        return metrics