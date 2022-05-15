from matplotlib import transforms
from custom_dataset import CustomImageDataset
from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms


"""
    Test performed to observe whether CustomImageDataset (dataset to use with CattleNet) is able to work together with
    the DataLoader class from pytorch appropriately.
"""
if __name__ == "__main__":
    dataset = CustomImageDataset(dataset_folder='../../dataset/Raw/RGB (320 x 240)/',img_dir='../../dataset/Raw/Combined/',transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]))
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for data in data_loader:
        print(data[3])
        exit()
        # print("Batch of images has shape: ",imgs.shape)
        # print("Batch of labels has shape: ", labels.shape)
