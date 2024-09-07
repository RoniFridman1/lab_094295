import os
import torch
import torchvision.models as models


def download_model(model_name: str, model_dir: str = "models"):
    """
    Downloads a pre-trained model if not already present in the 'models' directory.

    Args:
        model_name (str): Name of the model to download (e.g., 'resnet18', 'vgg16').
        model_dir (str): Directory to store the downloaded models.

    Returns:
        model (torch.nn.Module): The requested pre-trained model.
    """
    model_path = os.path.join(model_dir, model_name + ".pth")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Check if the model already exists
    if not os.path.exists(model_path):
        print(f"Downloading {model_name} model...")
        if model_name == "resnet18":
            model = models.resnet18(pretrained=True)
        elif model_name == "vgg16":
            model = models.vgg16(pretrained=True)
        elif model_name == "yolov3":
            # For YOLO, you may need a separate library like 'yolov3' from Ultralytics
            # This is a placeholder; replace with actual code to download YOLO models
            raise NotImplementedError("YOLO model download requires a custom implementation.")
        else:
            raise ValueError(f"Model {model_name} is not supported.")

        # Save the model
        torch.save(model.state_dict(), model_path)
    else:
        print(f"Loading {model_name} model from {model_path}...")
        if model_name == "resnet18":
            model = models.resnet18()
        elif model_name == "vgg16":
            model = models.vgg16()
        elif model_name == "yolov3":
            raise NotImplementedError("YOLO model download requires a custom implementation.")
        else:
            raise ValueError(f"Model {model_name} is not supported.")

        model.load_state_dict(torch.load(model_path))

    return model
