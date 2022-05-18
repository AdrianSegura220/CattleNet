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
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    equal = 0
    total = 64
    for data in data_loader:
        print(data[0][0][0])
        exit()
        for j in range(0,data[3].size(dim=0)):
            if data[3][j] == data[2][j]:
                equal += 1
        print(equal/total)
        # if data[2] == data[3]:
        #     print('True')
        # else:
        #     print('False')
        exit()
        # print("Batch of images has shape: ",imgs.shape)
        # print("Batch of labels has shape: ", labels.shape)
