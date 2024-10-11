import os
import torch
from torchvision import datasets, transforms
import PIL.Image


class AugmentedImageFolder(datasets.ImageFolder):
    def __init__(self, original_data_dir_path, augmented_data_dir_path, transform=None, num_augmentations=50):
        super().__init__(original_data_dir_path, transform)

        augmented_samples = []  # init the new augmented samples list

        # if the dir already exists no need to do the entire augmentation again
        if os.path.isdir(augmented_data_dir_path):
            # just need to update the samples variable to the augmented
            self.samples = datasets.ImageFolder(augmented_data_dir_path, transform=None).samples

        # need to run the augmentation
        else:
            os.mkdir(augmented_data_dir_path)
            os.mkdir(augmented_data_dir_path + "/NORMAL")
            os.mkdir(augmented_data_dir_path + "/PNEUMONIA")

            print(f"Augmenting each image in {original_data_dir_path} {num_augmentations} times")
            # go over the samples and augment each one 50 times
            for index, (img_path, label) in enumerate(self.samples):

                if index % 50 == 0:
                    print(f"Augmented so far {index} / {len(self.samples)} images")

                img = PIL.Image.open(img_path)
                for i in range(num_augmentations):
                    augmented_img = self.transform(img)
                    augmented_img = transforms.ToPILImage()(augmented_img)
                    label_str = "NORMAL" if label == 0 else "PNEUMONIA"
                    img_name = img_path.split("/")[-1].split(".")[0]  # change to above line of in local computer
                    # img_name = img_path.split("\\")[-1].split(".")[0]
                    augmented_img.save(f"{augmented_data_dir_path}/{label_str}/{img_name}_{i}.jpeg")
                    augmented_samples.append((f"{augmented_data_dir_path}/{label_str}/{img_name}_{i}.jpeg", label))

            self.samples = augmented_samples  # update the samples variable


if __name__ == '__main__':
    torch.manual_seed(42)

    transform = transforms.Compose([
        transforms.Resize((312, 312)),
        transforms.RandomRotation(degrees=(-15, 15)),  # Randomly rotate images by -15 to 15 degrees
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Randomly translate and scale
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    original_data_dir_path = "../chest_xray/train"
    augmented_data_dir_path = "chest_xray/train_augmented"
    train_dataset = AugmentedImageFolder(original_data_dir_path=original_data_dir_path,
                                         augmented_data_dir_path=augmented_data_dir_path,
                                         transform=transform)




